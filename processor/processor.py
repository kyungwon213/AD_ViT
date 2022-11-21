import logging
import os, cv2
import numpy as np
import time
import math
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    best_eoch = 1
    prev_rank1_best = 0.0
    prev_mAP_best = 0.0
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, kp_img, vid, target_cam, target_view, target_cloth, target_att) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            kp_img = kp_img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            target_att = target_att.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, kp_img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_att, target_cam, epoch/epochs)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))
        
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        # save chekpoint at the defined period
        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, kp_img, vid, camid, camids, target_view, target_cloth, target_att) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            kp_img = kp_img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            target_att = target_att.to(device)
                            feat = model(img, kp_img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid, target_cloth))
                    cmc, mAP = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10, 20, 50]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
                    
            else:
                model.eval()
                for n_iter, (img, kp_img, vid, camid, camids, target_view, target_cloth, target_att) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        kp_img = kp_img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        target_att = target_att.to(device)
                        feat = model(img, kp_img, cam_label=camids, view_label=target_view) # feat.shape: [256, 1536]
                        evaluator.update((feat, vid, camid, target_cloth))
                cmc, mAP = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10, 20, 50]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

                curr_rank1 = cmc[0]
                curr_mAP = mAP

                if curr_rank1 > prev_rank1_best:
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'checkpoint_best.pth'))
                    prev_rank1_best = curr_rank1
                    prev_mAP_best = curr_mAP
                    best_eoch = epoch
                    logger.info("Saved Best Checkpoint at {} epoch rank-1 {:.1%} cmc, {:.1%} mAP".format(best_eoch, cmc[0], mAP))
                logger.info("Saved Best Checkpoint at {} epoch".format(best_eoch))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()

    for n_iter, (img, kp_img, vid, camid, camids, target_view, target_cloth, target_att) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            kp_img = kp_img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, kp_img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, vid, camid, target_cloth))

    cmc, mAP = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10, 20, 50]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


