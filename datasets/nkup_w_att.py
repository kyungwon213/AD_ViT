import os
import os.path as osp
from config import cfg
from .bases import BaseImageDataset

class NKUP(BaseImageDataset):
    """
    For knowing how to read annotations please go to this link
    https://github.com/nkicsl/NKUP-dataset/issues/2

    Three test settings: (1) standard clothes setting, (2) cloth-changing setting, (3) both
    
    (1) stardard clothes setting (sc)

    (2) cloth-changing setting (cc)
    Dataset statistics:
    ---------------------------------------------------
    subset   | # ids | # images | # cameras | # clothes
    ---------------------------------------------------
    train    |    40 |     5336 |        15 |        11
    query    |    39 |      332 |        14 |        11
    gallery  |    67 |     4070 |        15 |        11
    ---------------------------------------------------

    (3) both
    
    """
    
    dataset_dir = 'NKUP'

    def __init__(self, root='', verbose=True, **kwargs):
        super(NKUP, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.list_train_path = os.listdir(self.train_dir)
        self.list_gallery_path = os.listdir(self.gallery_dir)
        self.list_query_path = os.listdir(self.query_dir)
        
        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path, relabel=True)
        query = self._process_dir(self.query_dir, self.list_query_path, relabel=False)
        gallery = self._process_dir(self.gallery_dir, self.list_gallery_path, relabel=False)
        
        if verbose:
            print("=> the original dataset is loaded")
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

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.train_dir): 
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.gallery_dir): 
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
        if not osp.exists(self.query_dir): 
            raise RuntimeError("'{}' is not available".format(self.query_dir))

    def _process_dir(self, dir_path, list_path, relabel=False):
        datasets = []
        pid_container_orig = set()
        pid_container_after = set()

        for img_idx, img_info in enumerate(list_path):
            pid = img_info.split('_')[0]
            pid_container_orig.add(pid)
        pid_container_orig = sorted(pid_container_orig)
        pid2label = {pid: label for label, pid in enumerate(pid_container_orig)}

        file_name = ''
        dataset_type = dir_path.split('/')[-1]
        if dataset_type == 'bounding_box_train':
            file_name = 'attributes_train.txt'
        elif dataset_type == 'query':
            file_name = 'attributes_query.txt'
        elif dataset_type == 'bounding_box_test':
            file_name = 'attributes_test.txt'

        attributes = []
        file_paths = []
        if cfg.MODEL.ATTR:
            with open(os.path.join(self.dataset_dir, 'PAR', 'baseline', cfg.MODEL.PAR_MODEL, cfg.MODEL.PAR_DATASET, file_name), 'r') as f:
                for fn in f:
                    file_paths.append(fn.strip().split(' ')[0])
                    attributes.append(list(map(int, fn.strip().split('[')[1].split(']')[0].split(' '))))

            for img_info, attribute in zip(list_path, attributes):
                img_path = os.path.join(dir_path, img_info)
                pid = img_info.split('_')[0]
                camid = int(img_info.split('_')[2].split("S")[0].split("C")[1])
                clothid = int(img_info.split('_')[1].split("D")[1])    # represents recording day which can also be considered clothes id
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]  # train ids must be relabelled from zero
                datasets.append((img_path, pid, camid, 1, clothid, [float(a) for a in attribute]))
                pid_container_after.add(pid)
        else:
            for img_idx, img_info in enumerate(list_path):
                img_path = os.path.join(dir_path, img_info)
                pid = img_info.split('_')[0]
                camid = int(img_info.split('_')[2].split("S")[0].split("C")[1])
                clothid = int(img_info.split('_')[1].split("D")[1])    # represents recording day which can also be considered clothes id
                camid -= 1  # index starts from 0
                if relabel: pid = pid2label[pid]  # train ids must be relabelled from zero
                datasets.append((img_path, pid, camid, 1, clothid))
                pid_container_after.add(pid)

        if relabel:
            # check if pid starts from 0 and increments with 1
            for idx, pid in enumerate(pid_container_after):
                assert idx == pid, "See code comment for explanation"

        return datasets
