from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp
import random
import torch
import numpy as np
from config import cfg
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks, cloths, atts = [], [], [], [], []

        for _, _, pid, camid, trackid, clothid, att in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
            cloths += [clothid]
            atts += [att]

        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        cloths = set(cloths)
        
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        num_cloths = len(cloths)
        len_attributes = len(atts[0])
        
        return num_pids, num_imgs, num_cams, num_views, num_cloths, len_attributes

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views, num_train_cloths, _ = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_query_views, num_query_cloths,  _ = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_views, num_gallery_cloths,  _ = self.get_imagedata_info(gallery)
        
        print("Dataset statistics:")
        print("  -----------------------------------------------------")
        print("  subset   | # ids | # images | # cameras | # clothes")
        print("  -----------------------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams, num_train_cloths))
        print("  query    | {:5d} | {:8d} | {:9d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams, num_query_cloths))
        print("  gallery  | {:5d} | {:8d} | {:9d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_gallery_cloths))
        print("  -----------------------------------------------------")


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, kp_file_info, pid, camid, trackid, clothid, attribute = self.dataset[index]

        img = read_image(img_path)
        kp_img = read_image(kp_file_info)

        if self.transform is not None:
            img = self.transform(img)
            kp_img = self.transform(kp_img)

        return img, kp_img, pid, camid, trackid, clothid, attribute