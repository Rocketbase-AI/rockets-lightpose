""" Main Model

The model.py file contains the main model used in the Rocket.
All the other classes and functions relative to the model,
should be in the layers.py file.

"""
import torch
import torch.nn as nn

import .layers


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            layers.conv(3, 32, stride=2, bias=False),
            layers.conv_dw(32, 64),
            layers.conv_dw(64, 128, stride=2),
            layers.conv_dw(128, 128),
            layers.conv_dw(128, 256, stride=2),
            layers.conv_dw(256, 256),
            layers.conv_dw(256, 512),  # conv4_2
            layers.conv_dw(512, 512, dilation=2, padding=2),
            layers.conv_dw(512, 512),
            layers.conv_dw(512, 512),
            layers.conv_dw(512, 512),
            layers.conv_dw(512, 512)   # conv5_5
        )
        self.cpm = layers.Cpm(512, num_channels)

        self.initial_stage = layers.InitialStage(
            num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for _ in range(num_refinement_stages):
            self.refinement_stages.append(
                layers.RefinementStage(
                    num_channels + num_heatmaps + num_pafs,
                    num_channels,
                    num_heatmaps,
                    num_pafs
                )
            )

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(
                    torch.cat(
                        [
                            backbone_features,
                            stages_output[-2],
                            stages_output[-1]
                        ], dim=1
                    )
                )
            )

        return stages_output
