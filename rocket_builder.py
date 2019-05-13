""" Function to create the Rocket

The rocket_builder.py file contains the build function to
build the Rocket. Moreover, it contains the preprocess
and postprocess functions that will be added to the model.
"""
from __future__ import division
import json
import types
import os

import cv2
import numpy as np
import PIL
from PIL import ImageDraw
import torch
import torch.nn as nn

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

    with open(config_path, 'r') as json_file:
        config = json.load(json_file)

    # Load Classes
    classes_path = os.path.join(os.path.realpath(
        os.path.dirname(__file__)), config['classes_path'])
    with open(classes_path, 'r') as json_file:
        classes = json.load(json_file)

    # Set up model
    model = lightpose.PoseEstimationWithMobileNet()
    weights_path = os.path.join(os.path.realpath(
        os.path.dirname(__file__)), config['weights_path'])
    checkpoint = torch.load(weights_path, map_location='cpu')
    utils.load_state(model, checkpoint)

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)
    setattr(model, 'config', config)
    setattr(model, 'classes', classes)

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
    # Test if the input is a PIL image
    if not isinstance(img, PIL.Image.Image):
        raise TypeError(
            'wrong input type: got {} but expected PIL.Image.Image.'.format(type(img)))

    # Load the parameters from config.json
    stride = self.config['stride']
    pad_value = tuple(self.config['pad_color_RGB'])
    img_mean = tuple(self.config['mean_RGB'])
    img_scale = 1.0 / 255 # Typo in the initial repo was 1/256
    net_input_height_size = self.config['input_size'][0]

    # Conver the PIL image to a numpy array
    np_img = np.array(img)

    # Converting the image from RGB to BGR
    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    # Scale the input image
    height, _, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(img, (0, 0), fx=scale,
                            fy=scale, interpolation=cv2.INTER_CUBIC)

    # Normalize the input image
    normalized_img = utils.normalize(
        scaled_img,
        img_mean,
        img_scale
    )

    # Pad the input image
    min_dims = [net_input_height_size, max(
        normalized_img.shape[1], net_input_height_size)]

    padded_img, _ = utils.pad_width(
        normalized_img,
        stride,
        pad_value,
        min_dims
    )

    # Convert numpy to tensor + final modification
    out_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()

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
    # Test if the input is a PIL image
    if not isinstance(input_img, PIL.Image.Image):
        raise TypeError(
            'wrong input type: got {} but expected PIL.Image.Image.'.format(type(input_img)))

    # Load parameters from config.json
    stride = self.config['stride']
    upsample_ratio = self.config['upsample_ratio']
    pad_value = tuple(self.config['pad_color_RGB'])
    img_mean = tuple(self.config['mean_RGB'])
    img_scale = 1.0 / 255
    net_input_height_size = self.config['input_size'][0]

    # Convert PIL image to numpy array
    np_img = np.array(input_img)
    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    height, _, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(img, (0, 0), fx=scale,
                            fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = utils.normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(
        scaled_img.shape[1], net_input_height_size)]
    _, pad = utils.pad_width(scaled_img, stride, pad_value, min_dims)

    # Extract the keypoint heatmaps (heatmaps)
    stage2_heatmaps = pose_output[-2]
    heatmaps = np.transpose(
        stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio,
                          fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    # Extract the Part Affinity Fields (pafs)
    stage2_pafs = pose_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio,
                      fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    # Extract the keypoints
    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(18):  # 19th for bg
        total_keypoints_num += utils.extract_keypoints(
            heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    # Group the keypoints
    pose_entries, all_keypoints = utils.group_keypoints(
        all_keypoints_by_type, pafs, demo=True)

    # Convert the position of the keypoints to the original image
    for kpt in all_keypoints:
        kpt[0] = (kpt[0] * stride / upsample_ratio - pad[1]) / scale
        kpt[1] = (kpt[1] * stride / upsample_ratio - pad[0]) / scale

    # Convert the list of keypoints to a list of dictionary:
    # [
    #   {"name_kpt":
    #       {"x": x_pos, "y": y_pos, "confidence": confidence_score},
    #   ...},
    # ...]

    list_humans_poses = []
    for human in pose_entries:
        human_pose = {}
        for kpt_id, kpt_location in enumerate(human[:-2]):
            if not kpt_location == -1:
                kpt_info = all_keypoints[int(kpt_location)]
                kpt_name = self.classes[str(int(kpt_id))]
                x_pos = kpt_info[0]
                y_pos = kpt_info[1]
                confidence_score = kpt_info[2]
                human_pose[kpt_name] = {
                    'x': x_pos,
                    'y': y_pos,
                    'confidence': confidence_score
                }
        list_humans_poses.append(human_pose)

    if visualize:
        # Visualization parameters
        line_width = 2
        line_color = (0, 225, 225, 255)
        point_radius = 4
        point_color = (255, 255, 255, 255)

        # Initialize the context to draw on the image
        img_out = input_img.copy()
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        # Draw the skeleton of each human in the picture
        for human in list_humans_poses:
            # Draw every connection defined in classes.json
            for connection in self.classes['connections']:
                # Test if both keypoints have been found
                human_has_kpt_a = connection[0] in human.keys()
                human_has_kpt_b = connection[1] in human.keys()
                if human_has_kpt_a and human_has_kpt_b:
                    # Get the coordinates of the two keypoints
                    kpt_a_x = int(round(human[connection[0]]['x']))
                    kpt_a_y = int(round(human[connection[0]]['y']))
                    kpt_b_x = int(round(human[connection[1]]['x']))
                    kpt_b_y = int(round(human[connection[1]]['y']))

                    # Draw the line between the two keypoints
                    ctx.line(
                        [(kpt_a_x, kpt_a_y), (kpt_b_x, kpt_b_y)],
                        fill=line_color,
                        width=line_width,
                        joint=None)

            # Draw Keypoints
            for _, item in human.items():
                # Create bounding box of the point
                top_left = (
                    int(round(item['x'] - point_radius)),
                    int(round(item['y'] - point_radius))
                )
                bottom_right = (
                    int(round(item['x'] + point_radius)),
                    int(round(item['y'] + point_radius))
                )

                # Draw the point at the keypoint position
                ctx.ellipse(
                    [top_left, bottom_right],
                    fill=point_color,
                    outline=None,
                    width=0
                )

        del ctx
        return img_out

    return list_humans_poses
