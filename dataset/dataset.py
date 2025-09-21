import os
import json
import base64
import random
from dataclasses import dataclass

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import pycocotools.mask as mask_util
from skimage.draw import disk


def annToRLE(h, w, ann):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = mask_util.frPyObjects(segm, h, w)
        rle = mask_util.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = mask_util.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


@dataclass
class DataCollator_MT_EVAL(object):
    """
    Data collator for evaluation and inference.
    Handles various vision tasks including segmentation, depth estimation, 
    normal estimation, semantic segmentation, and pose estimation.
    """
    resize: transforms.Resize
    norm: transforms.Normalize
    width: int = 768
    height: int = 768
    data_root_path: str = '/'
    go_through_all_seg_labels: bool = False

    @torch.no_grad()
    def __call__(self, samples):
        input_image_ = []
        target_image_ = []
        point_labels_ = []
        coor_point_ = []
        original_size_ = []
        target_size_ = []
        prompt_ = []
        input_to_viz_ = []
        path_ = []

        MAX_POINT = 5

        for s in samples:
            sample = self.process_example(s)
            input_image = sample['input']
            target_image = sample['target']
            path = sample['path']

            prompt = sample['caption']

            coor_point = torch.zeros((5, 2))
            point_label = torch.zeros((5, 1))

            cur_input_image, cur_target_image, crop_coords_top_left, original_size, target_size = self.process(
                input_image, target_image, True
            )

            input_to_viz = cur_input_image.clone()

            if '[[image2segmentation]]' in prompt:
                if self.go_through_all_seg_labels:
                    num_labels = cur_target_image.shape[0]

                    coor_point = torch.zeros((num_labels, 5, 2))
                    point_label = torch.zeros((num_labels, 5, 1))

                    for cur_idx in range(cur_target_image.shape[0]):
                        cur_path = path.replace('.jpg', f'_{cur_idx}.jpg')
                        mask_indices = torch.nonzero(cur_target_image[cur_idx] == 1)

                        indexes = self.generate_random_numbers(0, mask_indices.shape[0] - 1, MAX_POINT)

                        for i in range(len(indexes)):
                            idx = indexes[i]
                            center_x, center_y = mask_indices[idx, 1], mask_indices[idx, 2]

                            rr, cc = disk((center_y.item(), center_x.item()), 5, shape=(self.height, self.width))

                            input_to_viz[cur_idx][0, cc, rr] = 1.0
                            input_to_viz[cur_idx][1, cc, rr] = 0.0
                            input_to_viz[cur_idx][2, cc, rr] = 0.0

                            # cautious!
                            center_x = center_x / self.height
                            center_y = center_y / self.width

                            coor_point[cur_idx][i] = torch.tensor([center_x, center_y])
                            point_label[cur_idx][i] = torch.tensor([1])
                        path_.append(cur_path)
                        prompt_.append(prompt)

                    cur_target_image = cur_target_image.repeat(1, 3, 1, 1)

                    return_dict = {
                        "input_to_viz": input_to_viz,
                        "input_images": cur_input_image,
                        "target_images": cur_target_image,
                        "coor_point": coor_point,
                        "original_size": torch.stack([original_size] * num_labels),
                        "target_size": torch.stack([target_size] * num_labels),
                        "point_labels": point_label,
                        "prompt": prompt_,
                        "path": path_,
                    }

                    return return_dict

                else:
                    mask_indices = torch.nonzero(cur_target_image == 1)

                    indexes = self.generate_random_numbers(0, mask_indices.shape[0] - 1, MAX_POINT)

                    for i in range(len(indexes)):
                        idx = indexes[i]
                        center_x, center_y = mask_indices[idx, 1], mask_indices[idx, 2]

                        rr, cc = disk((center_y.item(), center_x.item()), 5, shape=(self.height, self.width))

                        input_to_viz[0, cc, rr] = 1.0
                        input_to_viz[1, cc, rr] = 0.0
                        input_to_viz[2, cc, rr] = 0.0

                        center_x = center_x / self.height
                        center_y = center_y / self.width

                        coor_point[i] = torch.tensor([center_x, center_y])
                        point_label[i] = torch.tensor([1])
                    cur_target_image = cur_target_image.repeat(3, 1, 1)

            input_to_viz_.append(input_to_viz)

            input_image_.append(cur_input_image)
            if cur_target_image is not None:
                target_image_.append(cur_target_image)
            point_labels_.append(point_label)
            prompt_.append(prompt)
            coor_point_.append(coor_point)
            original_size_.append(original_size)
            target_size_.append(target_size)
            path_.append(path)

        if len(target_image_) == 0:
            return {
                "input_to_viz": torch.stack(input_to_viz_),
                "input_images": torch.stack(input_image_),
                "coor_point": torch.stack(coor_point_),
                "original_size": torch.stack(original_size_),
                "target_size": torch.stack(target_size_),
                "point_labels": torch.stack(point_labels_),
                "prompt": prompt_,
                "path": path_,
            }
        else:
            return {
                "input_to_viz": torch.stack(input_to_viz_),
                "input_images": torch.stack(input_image_),
                "target_images": torch.stack(target_image_),
                "coor_point": torch.stack(coor_point_),
                "original_size": torch.stack(original_size_),
                "target_size": torch.stack(target_size_),
                "point_labels": torch.stack(point_labels_),
                "prompt": prompt_,
                "path": path_,
            }

    @torch.no_grad()
    def crop(self, input_image, target_image):
        w = input_image.shape[-1]
        h = input_image.shape[-2]
        input_image = self.resize(input_image)

        original_size = torch.tensor([w, h])
        target_size = torch.tensor([self.width, self.height])

        top = 0
        left = 0

        crop_coords_top_left = torch.tensor([top, left])

        return input_image, target_image, crop_coords_top_left, original_size, target_size

    @torch.no_grad()
    def process(self, input_image, target_image=None, do_crop=False):
        if do_crop:
            w = input_image.shape[-1]
            h = input_image.shape[-2]
            input_image = self.resize(input_image)

            if target_image is not None:
                target_image = self.resize(target_image)

            original_size = torch.tensor([w, h])
            target_size = torch.tensor([self.width, self.height])

            top = 0
            left = 0

            crop_coords_top_left = torch.tensor([top, left])

        input_image = torch.clamp(input_image, 0, 1)

        input_image = self.norm(input_image)
        if target_image is not None:
            target_image = self.norm(target_image)

        if not do_crop:
            return input_image, target_image

        return input_image, target_image, crop_coords_top_left, original_size, target_size

    def process_example(self, example):
        input_path = os.path.join(self.data_root_path, example['input'])
        input_image = cv2.imread(input_path)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        tensor_image = torch.tensor(input_image, dtype=torch.float32) / 255.0
        tensor_image = tensor_image.permute(2, 0, 1)

        prompt = example['caption']

        if 'target' in example.keys() and isinstance(example['target'], dict):
            m = example['target']

            if 'counts' not in m.keys():
                target_path = os.path.join(self.data_root_path, m['path'])

                if '[[image2depth]]' in example['caption']:
                    target_image = np.load(target_path)
                    target_tensor = torch.tensor(target_image, dtype=torch.float32)

                    _min, _max = torch.quantile(
                        target_tensor,
                        torch.tensor([0.02, 0.98]),
                    )

                    disp_linear = 1 / target_tensor
                    disp_min = 1 / _max
                    disp_max = 1 / _min
                    disp_norm_linear = (disp_linear - disp_min) / (disp_max - disp_min)

                    target_tensor = (disp_norm_linear - disp_norm_linear.min()) / (
                        disp_norm_linear.max() - disp_norm_linear.min()
                    )
                    target_tensor = target_tensor.unsqueeze(0).repeat(3, 1, 1)

                elif '[[image2normal]]' in example['caption'] or '[[image2pose]]' in example['caption']:
                    target = cv2.imread(target_path)
                    image = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

                    if image.max() > 1:
                        image = image / 255
                    target_tensor = torch.tensor(image, dtype=torch.float32)
                    target_tensor = target_tensor.permute(2, 0, 1)

                elif '[[image2segmentation]]' in example['caption']:
                    with open(target_path, 'r') as f:
                        annos = json.load(f)

                        if self.go_through_all_seg_labels:
                            target_tensor = []
                            for anno in annos['annotations']:
                                seg = anno['segmentation']
                                target = mask_util.decode(seg)
                                cur_target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
                                target_tensor.append(cur_target_tensor)
                            target_tensor = torch.stack(target_tensor)
                            num_labels = target_tensor.shape[0]
                            tensor_image = torch.stack([tensor_image] * num_labels)
                        else:
                            anno = random.choice(annos['annotations'])
                            seg = anno['segmentation']
                            target = mask_util.decode(seg)
                            target_tensor = torch.tensor(target, dtype=torch.float32)
                            target_tensor = target_tensor.unsqueeze(0)

                elif '[[image2semantic]]' in example['caption']:
                    with open(target_path, 'r') as f:
                        annos = json.load(f)
                        anno = random.choice(annos)
                        _, h, w = tensor_image.shape
                        rle = annToRLE(h, w, anno)
                        target = mask_util.decode(rle)
                        add_prompt = anno['prompt']
                        prompt = f'{prompt} {add_prompt}'
                    target_tensor = torch.tensor(target, dtype=torch.float32)
                    target_tensor = target_tensor.unsqueeze(0)
                    target_tensor = target_tensor.repeat(3, 1, 1)
            else:
                try:
                    if 'sa1b/' not in input_path:
                        m['counts'] = base64.b64decode(m['counts'])
                    target = mask_util.decode(m)
                    target_tensor = torch.tensor(target, dtype=torch.float32)
                except:
                    pass

            return {
                'input': tensor_image,
                'target': target_tensor,
                'caption': prompt,
                'path': input_path,
                'addition': example.get('addition', {}),
            }
        else:
            return {
                'input': tensor_image,
                'target': None,
                'caption': prompt,
                'path': input_path,
                'addition': example.get('addition', {}),
            }

    def generate_random_numbers(self, start, end, max_point):
        selected_numbers = []

        if start >= end + 1:
            return selected_numbers

        remaining_numbers = set(range(start, end + 1))
        remaining_numbers = list(remaining_numbers)

        max_point = min(max_point, end + 1)

        try:
            selected_numbers = random.sample(remaining_numbers, k=max_point)
        except Exception:
            selected_numbers = remaining_numbers[:max_point]

        return selected_numbers


