import layers
import torch
import torch.nn as nn


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = layers.conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            layers.conv_dw_no_bn(out_channels, out_channels),
            layers.conv_dw_no_bn(out_channels, out_channels),
            layers.conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = layers.conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            layers.conv(num_channels, num_channels, bn=False),
            layers.conv(num_channels, num_channels, bn=False),
            layers.conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            layers.conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            layers.conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            layers.conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            layers.conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = layers.conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            layers.conv(out_channels, out_channels),
            layers.conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            layers.conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            layers.conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            layers.conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            layers.conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            layers.conv(     3,  32, stride=2, bias=False),
            layers.conv_dw( 32,  64),
            layers.conv_dw( 64, 128, stride=2),
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
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output
