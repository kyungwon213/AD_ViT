import os
import glob
import os.path as osp
from config import cfg
from .bases import BaseImageDataset

class LTCC(BaseImageDataset):
    """
    Two test settings: (1) standard clothes setting, (2) cloth-changing setting
    
    (1) stardard clothes setting (sc)
    Dataset statistics:
    ---------------------------------------------------
    subset   | # ids | # images | # cameras | # clothes
    ---------------------------------------------------
    train    |    31 |     1209 |        12 |         1
    query    |    75 |      493 |        12 |        13
    gallery  |    30 |     1031 |        12 |         1
    ---------------------------------------------------

    (2) cloth-changing setting (cc)
    Dataset statistics:
    ---------------------------------------------------
    subset   | # ids | # images | # cameras | # clothes
    ---------------------------------------------------
    train    |    46 |     8367 |        12 |        14
    query    |    75 |      493 |        12 |        13
    gallery  |    45 |     6019 |        12 |        13
    ---------------------------------------------------
    """
    dataset_dir = 'LTCC_ReID'

    def __init__(self, root='', verbose=True, **kwargs):
        super(LTCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'test')
        self.unique_clothes_dict = {}
        self.unique_clothes_cnt = 0

        if not cfg.MODEL.ATTR:
            if cfg.MODEL.Evaluate == "ClothChangingSetting":
                self.list_train_path = self._init_path(self.train_dir, 'cloth-change_id_train.txt')
                self.list_query_path = [os.path.join(self.query_dir, path) for path in os.listdir(self.query_dir)]
                self.list_gallery_path = self._init_path(self.gallery_dir, 'cloth-change_id_test.txt')
            elif cfg.MODEL.Evaluate == "StandardSetting":
                self.list_train_path = self._init_path(self.train_dir, 'cloth-unchange_id_train.txt')
                self.list_query_path = [os.path.join(self.query_dir, path) for path in os.listdir(self.query_dir)]
                self.list_gallery_path = self._init_path(self.gallery_dir, 'cloth-unchange_id_test.txt')
            
            self._check_before_run()
            train = self._process_dir(self.list_train_path, relabel=True)
            query = self._process_dir(self.list_query_path, relabel=False)
            gallery = self._process_dir(self.list_gallery_path, relabel=False)

        else:
            if cfg.MODEL.Evaluate == "ClothChangingSetting":
                train = self._process_dir('cloth-change_id_train.txt', par=cfg.MODEL.ATTR, relabel=True)
                query = self._process_dir('cloth-change_id_query.txt', par=cfg.MODEL.ATTR, relabel=False)
                gallery = self._process_dir('cloth-change_id_test.txt', par=cfg.MODEL.ATTR, relabel=False)
            elif cfg.MODEL.Evaluate == "StandardSetting":
                train = self._process_dir('cloth-unchange_id_train.txt', par=cfg.MODEL.ATTR, relabel=True)
                query = self._process_dir('cloth-unchange_id_query.txt', par=cfg.MODEL.ATTR, relabel=False)
                gallery = self._process_dir('cloth-unchange_id_test.txt', par=cfg.MODEL.ATTR, relabel=False)
            
        if verbose:
            print("=> LTCC dataset loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        if cfg.MODEL.ATTR:
            self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids, self.num_train_clothids, self.len_train_attributes = self.get_imagedata_info(self.train)
            self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids, self.num_query_clothids, self.len_query_attributes = self.get_imagedata_info(self.query)
            self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids, self.num_gallery_clothids, self.len_gallery_attributes = self.get_imagedata_info(self.gallery)
        else:
            self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids, self.num_train_clothids = self.get_imagedata_info(self.train)
            self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids, self.num_query_clothids = self.get_imagedata_info(self.query)
            self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids, self.num_gallery_clothids = self.get_imagedata_info(self.gallery)

    def _init_path(self, dir_path, file_name):
        info_dir = osp.join(self.dataset_dir, 'info')
        list_file = osp.join(info_dir, file_name)
        pids = []
        with open(list_file, 'r') as f:
            for fn in f:
                pids.append(fn.strip())

        image_paths = []
        for pid in pids:
            image_paths += glob.glob(os.path.join(dir_path, pid)+'*.png')

        return image_paths

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir): 
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir): 
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir): 
            raise RuntimeError("'{}' is not available".format(self.query_dir))

    def _process_dir(self, list_path, par=False, relabel=False):
        datasets = []
        pid_container_orig = set()
        pid_container_after = set()

        if par:
            file_paths = []
            attributes = []
            with open(os.path.join(self.dataset_dir, 'PAR', 'baseline', cfg.MODEL.PAR_MODEL, cfg.MODEL.PAR_DATASET, list_path), 'r') as f:
                for fn in f:
                    file_paths.append(fn.strip().split(' ')[0])
                    attributes.append(list(map(int, fn.strip().split('[')[1].split(']')[0].split(' '))))
            
            for img_idx, file_info in enumerate(file_paths):
                img_info = file_info.split('/')[-1]
                pid = int(img_info.split('_')[0])
                pid_container_orig.add(pid)
            pid_container_orig = sorted(pid_container_orig)
            pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}

            for file_info, attribute in zip(file_paths, attributes):
                img_info = file_info.split('/')[-1]
                pid = int(img_info.split('_')[0])
                camid = int(img_info.split('_')[2].split("c")[1])
                clothid = int(img_info.split('_')[1])
                assert 0 <= pid <= 151
                assert 1 <= camid <= 12
                camid -= 1  # index starts from 0
                clothid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]  # train ids must be relabelled from zero
                unique_clothid = pid*14 + clothid 
                if unique_clothid not in self.unique_clothes_dict:
                    self.unique_clothes_dict[unique_clothid] = self.unique_clothes_cnt
                    self.unique_clothes_cnt += 1
                attributes_info = [float(a) for a in attribute]
                datasets.append((file_info, pid, camid, self.unique_clothes_dict[unique_clothid], self.unique_clothes_dict[unique_clothid], attributes_info))
                pid_container_after.add(pid)
        else:     
            for img_idx, file_info in enumerate(list_path):
                img_info = file_info.split('/')[-1]
                pid = int(img_info.split('_')[0])
                pid_container_orig.add(pid)
            pid_container_orig = sorted(pid_container_orig)
            pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}

            for img_idx, file_info in enumerate(list_path):
                img_info = file_info.split('/')[-1]
                pid = int(img_info.split('_')[0])
                camid = int(img_info.split('_')[2].split("c")[1])
                clothid = int(img_info.split('_')[1])
                assert 0 <= pid <= 151
                assert 1 <= camid <= 12
                camid -= 1  # index starts from 0
                clothid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]  # train ids must be relabelled from zero
                unique_clothid = pid*14 + clothid
                datasets.append((file_info, pid, camid, unique_clothid, clothid))
                pid_container_after.add(pid)

        if relabel:
            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"

        return datasets
