from __future__ import division
import cv2
import json
import numpy as np
import os
import PIL
import torch
import torch.nn as nn
import torchvision
import types
import math
import time

from . import utils
from . import model as lightpose


def build(config_path: str = '') -> nn.Module:
    """Builds a pytorch compatible deep learning model

    The model can be used as any other pytorch model. Additional methods
    for `preprocessing`, `postprocessing` have been added to ease the handling of the model
    and simplify interchangeability of different models.
    """
    # Load Config file
    if not config_path:  # If no config path then load default one
        config_path = os.path.join(os.path.realpath(
            os.path.dirname(__file__)), "config.json")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Set up model
    model = lightpose.PoseEstimationWithMobileNet()
    weights_path = os.path.join(os.path.realpath(
        os.path.dirname(__file__)), config['weights_path'])
    checkpoint = torch.load(weights_path, map_location='cpu')
    utils.load_state(model, checkpoint)

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)
    setattr(model, 'config', config)

    return model


def preprocess(self, img: PIL.Image.Image) -> torch.Tensor:
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.
    Labels must have the following format: `x1, y1, x2, y2, category_id`

    Args:
        img (PIL.Image): input image
        labels (list): list of bounding boxes and class labels
    """
    start = time.time()
    if not isinstance(img, PIL.Image.Image):
        raise TypeError(
            'wrong input type: got {} but expected PIL.Image.Image.'.format(type(img)))

    originalMethod = False

    if originalMethod:
        img = np.array(img)

        stride = self.config['stride']
        pad_value = tuple(self.config['pad_color_RGB'])
        img_mean = tuple(self.config['mean_RGB'])
        img_scale = 1.0 / self.config['input_size'][0]
        net_input_height_size = self.config['input_size'][0]

        np_img = np.array(img)
        # Converting the image from RGB to BGR

        img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        # Scale the input image
        height, width, _ = img.shape
        scale = net_input_height_size / height
        scaled_img = cv2.resize(img, (0, 0), fx=scale,
                                fy=scale, interpolation=cv2.INTER_CUBIC)

        # Normalize the input image
        scaled_img = utils.normalize(
                            scaled_img,
                            img_mean,
                            img_scale
                        )

        # Pad the input image
        min_dims = [net_input_height_size, max(
            scaled_img.shape[1], net_input_height_size)]
        
        padded_img, pad = utils.pad_width(
                                scaled_img,
                                stride,
                                pad_value,
                                min_dims
                            )

        out_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    else:
        # Initialize list of transformations necessary for the preprocessing
        list_transformations = []

        # ---RESIZE---
        # Resize the image with the height to the input size
        scale = self.config['input_size'][0] / img.size[1]
        new_h = self.config['input_size'][0]
        new_w = int(round(img.size[0] * scale))

        list_transformations.append(
            torchvision.transforms.Resize(
                (new_h, new_w),
                PIL.Image.BICUBIC
            )
        )

        # --- PADDING ---
        # Get optimal width
        optimal_width = max(new_w, self.config['input_size'][1])

        # Compute the padding for the width to be a multiple of the stride
        optimal_width = math.ceil(optimal_width / self.config['stride']) * self.config['stride']

        pad_left = int(math.floor((optimal_width - new_w) / 2.0))
        pad_right = int(optimal_width - new_w - pad_left)
        # No padding for the top and bottom
        pad_top = 0
        pad_bottom = 0

        list_transformations.append(
            torchvision.transforms.Pad(
                padding=(pad_left, pad_top, pad_right, pad_bottom),
                fill=tuple(self.config['pad_color_RGB']),
                padding_mode='constant'
            )
        )

        # ---CONVERT TO TENSOR---
        list_transformations.append(
            torchvision.transforms.ToTensor()
        )

        # ---NORMALIZE---
        # Load the mean color to normalize the images
        mean_rgb = tuple(self.config['mean_RGB'])
        if max(mean_rgb) > 1: # convert on
            mean_rgb = [v / 255 for v in mean_rgb]

        list_transformations.append(
            torchvision.transforms.Normalize(
                mean_rgb,
                (1.0, 1.0, 1.0)
            )
        )

        # --- RGB TO BGR ---
        list_transformations.append(
            torchvision.transforms.Lambda(
                lambda x: x[[2, 1, 0], : ,:]
            )
        )

        # ---APPLY TRANSFORMS ---
        transform_input = torchvision.transforms.Compose(
            list_transformations
        )
        # Apply transformations
        out_tensor = transform_input(img)
        # Add the batch size dimension
        out_tensor = out_tensor.unsqueeze(0)

    print(time.time() - start)
    return out_tensor


def postprocess(self, pose_output: torch.Tensor, input_img: PIL.Image, visualize: bool = False):
    """Converts pytorch tensor into interpretable format

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the rocket family there might be additional options.
    This model supports either outputting a list of bounding boxes of the format
    (x0, y0, w, h) or outputting a `PIL.Image` with the bounding boxes
    and (class name, class confidence, object confidence) indicated.

    Args:
        detections (Tensor): Output Tensor to postprocess
        input_img (PIL.Image): Original input image which has not been preprocessed yet
        visualize (bool): If True outputs image with annotations else a list of bounding boxes
    """
    color = self.config['color']
    stride = self.config['stride']
    upsample_ratio = self.config['upsample_ratio']
    pad_value = tuple(self.config['pad_color_RGB'])
    img_mean = tuple(self.config['mean_RGB'])
    img_scale = 1.0 / self.config['input_size'][0]
    net_input_height_size = self.config['input_size'][0]

    np_img = np.array(input_img)
    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(img, (0, 0), fx=scale,
                            fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = utils.normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(
        scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = utils.pad_width(scaled_img, stride, pad_value, min_dims)

    stage2_heatmaps = pose_output[-2]
    heatmaps = np.transpose(
        stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio,
                          fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = pose_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio,
                      fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    if visualize:
        orig_img = img.copy()
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += utils.extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = utils.group_keypoints(
            all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (
                all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (
                all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            for part_id in range(len(utils.BODY_PARTS_PAF_IDS) - 2):
                kpt_a_id = utils.BODY_PARTS_KPT_IDS[part_id][0]
                global_kpt_a_id = pose_entries[n][kpt_a_id]
                if global_kpt_a_id != -1:
                    x_a, y_a = all_keypoints[int(global_kpt_a_id), 0:2]
                    cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1)
                kpt_b_id = utils.BODY_PARTS_KPT_IDS[part_id][1]
                global_kpt_b_id = pose_entries[n][kpt_b_id]
                if global_kpt_b_id != -1:
                    x_b, y_b = all_keypoints[int(global_kpt_b_id), 0:2]
                    cv2.circle(img, (int(x_b), int(y_b)), 3, color, -1)
                if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                    cv2.line(img, (int(x_a), int(y_a)),
                             (int(x_b), int(y_b)), color, 2)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        cv2_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(cv2_im)
    return pafs
