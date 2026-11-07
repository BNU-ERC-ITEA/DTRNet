import numbers
from collections import defaultdict

import numpy as np
import torch


class DictCollator(object):
    """data batch."""

    def __call__(self, batch):
        data_dict = defaultdict(list)
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if isinstance(v, (np.ndarray, torch.Tensor, numbers.Number)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.from_numpy(data_dict[k])
        return data_dict


class ListCollator(object):
    """data batch."""

    def __call__(self, batch):
        data_dict = defaultdict(list)
        to_tensor_idxs = []
        for sample in batch:
            for idx, v in enumerate(sample):
                if isinstance(v, (np.ndarray, torch.Tensor, numbers.Number)):
                    if idx not in to_tensor_idxs:
                        to_tensor_idxs.append(idx)
                data_dict[idx].append(v)
        for idx in to_tensor_idxs:
            data_dict[idx] = torch.from_numpy(data_dict[idx])
        return list(data_dict.values())


class SSLRotateCollate(object):
    """
    bach: [
        [(4*3xH*W), (4,)]
        [(4*3xH*W), (4,)]
        ...
    ]
    """

    def __call__(self, batch):
        output = [np.concatenate(d, axis=0) for d in zip(*batch)]
        return output


class RecWithRawCollator(object):
    """Collate for rec with extra raw image (variable size)."""

    def __call__(self, batch):
        from torch.utils.data._utils.collate import default_collate

        if not batch:
            return []
        
        num_fields = len(batch[0])
        output = []
        for i in range(num_fields):
            column = [b[i] for b in batch]
            sample = column[0]
            # Heuristic: try to collate numeric/tensor/array types
            if isinstance(sample, (np.ndarray, torch.Tensor, numbers.Number)):
                try:
                    output.append(default_collate(column))
                except Exception:
                    # Fallback if dimensions mismatch or other collate error
                    output.append(column)
            else:
                # Keep as list (e.g. raw bytes, strings)
                output.append(column)
        return output


class DyMaskCollator(object):
    """
    batch: [
        image [batch_size, channel, maxHinbatch, maxWinbatch]
        image_mask [batch_size, channel, maxHinbatch, maxWinbatch]
        label [batch_size, maxLabelLen]
        label_mask [batch_size, maxLabelLen]
        ...
    ]
    """

    def __call__(self, batch):
        max_width, max_height, max_length = 0, 0, 0
        bs, channel = len(batch), batch[0][0].shape[0]
        proper_items = []
        for item in batch:
            if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[
                    2] * max_height > 1600 * 320:
                continue
            max_height = item[0].shape[
                1] if item[0].shape[1] > max_height else max_height
            max_width = item[0].shape[
                2] if item[0].shape[2] > max_width else max_width
            max_length = len(
                item[1]) if len(item[1]) > max_length else max_length
            proper_items.append(item)

        images, image_masks = np.zeros(
            (len(proper_items), channel, max_height, max_width),
            dtype='float32'), np.zeros(
                (len(proper_items), 1, max_height, max_width), dtype='float32')
        labels, label_masks = np.zeros((len(proper_items), max_length),
                                       dtype='int64'), np.zeros(
                                           (len(proper_items), max_length),
                                           dtype='int64')

        for i in range(len(proper_items)):
            _, h, w = proper_items[i][0].shape
            images[i][:, :h, :w] = proper_items[i][0]
            image_masks[i][:, :h, :w] = 1
            l = len(proper_items[i][1])
            labels[i][:l] = proper_items[i][1]
            label_masks[i][:l] = 1

        return images, image_masks, labels, label_masks


class AutoLabelMatchCollate(object):
    """Flatten per-image positive/negative pairs and return torch tensors."""

    def __call__(self, batch):
        output = []
        for items in zip(*batch):
            arr = np.concatenate(items, axis=0)
            output.append(torch.from_numpy(arr))
        return output
