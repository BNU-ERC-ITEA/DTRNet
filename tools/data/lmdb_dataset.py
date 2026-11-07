import os

import cv2
import lmdb
import numpy as np
from torch.utils.data import Dataset

from openrec.preprocess import create_operators, transform


class LMDBDataSet(Dataset):

    def __init__(self, config, mode, logger, seed=None, epoch=1, task='rec'):
        super(LMDBDataSet, self).__init__()

        self.logger = logger
        global_config = config['Global']
        dataset_config = config[mode]['dataset']
        loader_config = config[mode]['loader']
        loader_config['batch_size_per_card']
        data_dir = dataset_config['data_dir']
        self.do_shuffle = loader_config['shuffle']
        self.max_retry = int(dataset_config.get('max_retry', 50))
        self.ext_data_max_retry = int(dataset_config.get('ext_data_max_retry',
                                                        max(50, self.max_retry)))
        self.debug_error_samples = int(dataset_config.get('debug_error_samples', 3))

        self.lmdb_sets = self.load_hierarchical_lmdb_dataset(data_dir)
        logger.info(f'Initialize indexs of datasets: {data_dir}')
        self.data_idx_order_list = self.dataset_traversal()
        if self.do_shuffle:
            np.random.shuffle(self.data_idx_order_list)
        self.ops = create_operators(dataset_config['transforms'],
                                    global_config)
        self.ext_op_transform_idx = dataset_config.get('ext_op_transform_idx',
                                                       1)

        ratio_list = dataset_config.get('ratio_list', [1.0])
        self.need_reset = True in [x < 1 for x in ratio_list]

    def load_hierarchical_lmdb_dataset(self, data_dir):
        lmdb_sets = {}
        dataset_idx = 0
        # change by lrr 20251228： 修复 os.walk 无法遍历符号链接目录的问题
        for dirpath, dirnames, filenames in os.walk(data_dir + '/', followlinks=True):
            if not dirnames:
                env = lmdb.open(
                    dirpath,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                txn = env.begin(write=False)
                num_samples = int(txn.get('num-samples'.encode()))
                lmdb_sets[dataset_idx] = {
                    'dirpath': dirpath,
                    'env': env,
                    'txn': txn,
                    'num_samples': num_samples,
                }
                dataset_idx += 1
        # print(f"lmdb_sets: {lmdb_sets}") # debug
        return lmdb_sets

    def dataset_traversal(self):
        lmdb_num = len(self.lmdb_sets)
        total_sample_num = 0
        for lno in range(lmdb_num):
            total_sample_num += self.lmdb_sets[lno]['num_samples']
        data_idx_order_list = np.zeros((total_sample_num, 2))
        beg_idx = 0
        for lno in range(lmdb_num):
            tmp_sample_num = self.lmdb_sets[lno]['num_samples']
            end_idx = beg_idx + tmp_sample_num
            data_idx_order_list[beg_idx:end_idx, 0] = lno
            data_idx_order_list[beg_idx:end_idx,
                                1] = list(range(tmp_sample_num))
            data_idx_order_list[beg_idx:end_idx, 1] += 1
            beg_idx = beg_idx + tmp_sample_num
        return data_idx_order_list

    def get_img_data(self, value):
        """get_img_data."""
        if not value:
            return None
        imgdata = np.frombuffer(value, dtype='uint8')
        if imgdata is None:
            return None
        imgori = cv2.imdecode(imgdata, 1)
        if imgori is None:
            return None
        return imgori

    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, 'ext_data_num'):
                ext_data_num = getattr(op, 'ext_data_num')
                break
        load_data_ops = self.ops[:self.ext_op_transform_idx]
        ext_data = []
        try_count = 0

        while len(ext_data) < ext_data_num and try_count < self.ext_data_max_retry:
            try_count += 1
            lmdb_idx, file_idx = self.data_idx_order_list[np.random.randint(
                len(self))]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            sample_info = self.get_lmdb_sample_info(
                self.lmdb_sets[lmdb_idx]['txn'], file_idx)
            if sample_info is None:
                continue
            img, label = sample_info
            data = {'image': img, 'label': label}
            data, _ = self.apply_ops_with_trace(data, load_data_ops)
            if data is None:
                continue
            ext_data.append(data)
        if len(ext_data) < ext_data_num:
            self.logger.warning(
                f'Only collected {len(ext_data)}/{ext_data_num} ext_data samples '
                f'after {try_count} tries')
        return ext_data

    def get_lmdb_sample_info(self, txn, index):
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key)
        if label is None:
            return None
        label = label.decode('utf-8')
        img_key = 'image-%09d'.encode() % index
        imgbuf = txn.get(img_key)
        return imgbuf, label

    def apply_ops_with_trace(self, data, ops):
        for op_idx, op in enumerate(ops):
            op_name = type(op).__name__
            try:
                data = op(data)
            except Exception as exc:
                label = data.get('label', '') if isinstance(data, dict) else ''
                raise RuntimeError(
                    f'op[{op_idx}]={op_name} raised {type(exc).__name__}: {exc}; '
                    f'label={label!r}'
                ) from exc
            if data is None:
                return None, f'op[{op_idx}]={op_name} returned None'
        return data, None

    def __getitem__(self, idx):
        current_idx = idx
        errors = []

        for _ in range(self.max_retry):
            lmdb_idx, file_idx = self.data_idx_order_list[current_idx]
            lmdb_idx = int(lmdb_idx)
            file_idx = int(file_idx)
            dataset_path = self.lmdb_sets[lmdb_idx]['dirpath']
            sample_info = self.get_lmdb_sample_info(
                self.lmdb_sets[lmdb_idx]['txn'], file_idx)
            if sample_info is None:
                error = (
                    f'missing sample_info; dir={dataset_path}, '
                    f'lmdb_idx={lmdb_idx}, file_idx={file_idx}'
                )
            else:
                img, label = sample_info
                data = {'image': img, 'label': label}
                data['ext_data'] = self.get_ext_data()
                outs, fail_reason = self.apply_ops_with_trace(data, self.ops)
                if outs is not None:
                    return outs
                error = (
                    f'{fail_reason}; dir={dataset_path}, '
                    f'lmdb_idx={lmdb_idx}, file_idx={file_idx}, '
                    f'label={label!r}'
                )

            if len(errors) < self.debug_error_samples:
                errors.append(error)
            current_idx = np.random.randint(self.__len__())

        raise RuntimeError(
            f'Failed to fetch a valid sample after {self.max_retry} retries. '
            f'Likely all sampled items are filtered by preprocess ops. '
            f'Examples: {" | ".join(errors)}'
        )

    def __len__(self):
        return self.data_idx_order_list.shape[0]
