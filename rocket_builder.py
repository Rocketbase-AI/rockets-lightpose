import os
import cv2
import torch
import types
import json
import numpy as np
import torch.nn as nn
from PIL import Image
from PIL import ImageDraw
from torchvision import transforms
from .model import PoseEstimationWithMobileNet
from .utils import normalize, pad_width, load_state, extract_keypoints, group_keypoints, BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS


def build(config_path: str = '') -> nn.Module:
    """Builds a pytorch compatible deep learning model

    The model can be used as any other pytorch model. Additional methods
    for `preprocessing`, `postprocessing`, `label_to_class` have been added to ease handling of the model
    and simplify interchangeability of different models.
    """
    # Load Config file
    if not config_path: # If no config path then load default one
        config_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), "config.json")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Set up model
    model = PoseEstimationWithMobileNet()
    weights_path = os.path.join(os.path.realpath(os.path.dirname(__file__)), config['weights_path'])
    checkpoint = torch.load(weights_path, map_location='cpu')
    load_state(model, checkpoint)

    
    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)

    return model


def preprocess(self, img: Image) -> torch.Tensor:
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.
    Labels must have the following format: `x1, y1, x2, y2, category_id`

    Args:
        img (PIL.Image): input image
        labels (list): list of bounding boxes and class labels
    """
    if type(img) == Image.Image:
        # PIL.Image
        # Extract image
        img = np.array(img)
    elif type(img) == torch.Tensor:
        # list of tensors
        img = img[0].cpu()
        img = transforms.ToPILImage()(img)
        img = np.array(img)
    elif "PIL" in str(type(img)): # type if file just has been opened
        img = np.array(img.convert("RGB"))
    else:
        raise TypeError("wrong input type: got {} but expected list of PIL.Image, "
                        "single PIL.Image or torch.Tensor".format(type(img)))
    pad_value=(0, 0, 0)
    img_mean=(128, 128, 128)
    img_scale=1/256
    net_input_height_size = 256
    stride = 8
    np_img = np.array(img)
    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    return torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()


def postprocess(self, pose_output: torch.Tensor, input_img: Image, visualize: bool = False):
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
    net_input_height_size = 256
    pad_value=(0, 0, 0)
    img_mean=(128, 128, 128)
    img_scale=1/256
    stride = 8
    upsample_ratio = 4
    color = [0, 224, 255]
    np_img = np.array(input_img)
    img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    height, width, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    stage2_heatmaps = pose_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = pose_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
    
    if visualize:
      orig_img = img.copy()
      total_keypoints_num = 0
      all_keypoints_by_type = []
      for kpt_idx in range(18):  # 19th for bg
          total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

      pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
      for kpt_id in range(all_keypoints.shape[0]):
          all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
          all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
      for n in range(len(pose_entries)):
          if len(pose_entries[n]) == 0:
              continue
          for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
              kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
              global_kpt_a_id = pose_entries[n][kpt_a_id]
              if global_kpt_a_id != -1:
                  x_a, y_a = all_keypoints[int(global_kpt_a_id), 0:2]
                  cv2.circle(img, (int(x_a), int(y_a)), 3, color, -1)
              kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
              global_kpt_b_id = pose_entries[n][kpt_b_id]
              if global_kpt_b_id != -1:
                  x_b, y_b = all_keypoints[int(global_kpt_b_id), 0:2]
                  cv2.circle(img, (int(x_b), int(y_b)), 3, color, -1)
              if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                  cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2)
      img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
      cv2_im = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      return Image.fromarray(cv2_im)
    return pafs