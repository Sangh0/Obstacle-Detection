import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *

from .config import NUM_CLASSES, NUM_ANCHORS_PER_SCALE, ANCHORS, NUM_ATTRIB, LAST_LAYER_DIM


class ConvLayer(nn.Module):
    """
    Convolutional Layer, Conv + BN + Leaky ReLU
    """
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        kernel_size: int, 
        stride: int=1, 
        slope: float=0.1
    ):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.leaky_relu = nn.LeakyReLU(slope)
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class ResBlock(nn.Module):
    """
    Residual Block to build Darknet backbone
    """
    def __init__(self, in_dim: int):
        super(ResBlock, self).__init__()
        assert in_dim % 2 == 0, f'{in_dim} is not even'
        hidden_dim = in_dim // 2
        self.conv1 = ConvLayer(in_dim, hidden_dim, kernel_size=1)
        self.conv2 = ConvLayer(hidden_dim, in_dim, kernel_size=3)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class YOLOLayer(nn.Module):
    """
    A last layer in yolo network to detect objects
    """
    def __init__(self, scale: str, stride: int):
        super(YOLOLayer, self).__init__()
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            idx = None

        self.anchors = torch.tensor([ANCHORS[i] for i in idx])
        self.stride = stride

    def forward(self, x: torch.Tensor):
        num_batch = x.size(0)
        num_grid = x.size(2)

        if self.training:
            output_raw = x.view(
                num_batch, NUM_ANCHORS_PER_SCALE, NUM_ATTRIB, num_grid, num_grid,
            ).permute(0, 1, 3, 4, 2).contiguous()
            output_raw = output_raw.view(
                num_batch, -1, NUM_ATTRIB,
            )
            
            return output_raw

        else:
            prediction_raw = x.view(
                num_batch, NUM_ANCHORS_PER_SCALE, NUM_ATTRIB, num_grid, num_grid,
            ).permute(0, 1, 3, 4, 2).contiguous()

            self.anchors = self.anchors.to(x.device).float()
            
            # calculate offsets for each grid
            grid_tensor = torch.arange(num_grid, dtype=torch.float, device=x.device).repeat(num_grid, 1)
            grid_x = grid_tensor.view([1, 1, num_grid, num_grid])
            grid_y = grid_tensor.t().view([1, 1, num_grid, num_grid])
            anchor_w = self.anchors[:, 0:1].view((1, -1, 1, 1))
            anchor_h = self.anchors[:, 1:2].view((1, -1, 1, 1))

            # get outputs
            x_center_pred = (torch.sigmoid(prediction_raw[..., 0]) + grid_x) * self.stride
            y_center_pred = (torch.sigmoid(prediction_raw[..., 1]) + grid_y) * self.stride
            w_pred = torch.exp(prediction_raw[..., 2]) * anchor_w
            h_pred = torch.exp(prediction_raw[..., 3]) * anchor_h
            bbox_pred = torch.stack((x_center_pred, y_center_pred, w_pred, h_pred), dim=4).view(num_batch, -1, 4)
            conf_pred = torch.sigmoid(prediction_raw[..., 4]).view(num_batch, -1, 1)
            cls_pred = torch.sigmoid(prediction_raw[..., 5:]).view(num_batch, -1, NUM_CLASSES)

            output = torch.cat((bbox_pred, conf_pred, cls_pred), dim=-1)
            return output


class DarkNet53(nn.Module):
    """
    A backbone network to extract features
    """
    def __init__(
        self,
        in_dim: int=3, 
        num_filters: int=32, 
        repeat_list: List[int]=[1,2,8,8,4],
    ):
        super(DarkNet53, self).__init__()
        self.conv1 = ConvLayer(in_dim, num_filters, kernel_size=3)
        self.block1 = self._make_residual_block(num_filters, num_filters*2, res_repeat=repeat_list[0])
        self.block2 = self._make_residual_block(num_filters*2, num_filters*4, res_repeat=repeat_list[1])
        self.block3 = self._make_residual_block(num_filters*4, num_filters*8, res_repeat=repeat_list[2])
        self.block4 = self._make_residual_block(num_filters*8, num_filters*16, res_repeat=repeat_list[3])
        self.block5 = self._make_residual_block(num_filters*16, num_filters*32, res_repeat=repeat_list[4])

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        out3 = self.block3(x)
        out2 = self.block4(out3)
        out1 = self.block5(out2)
        return out1, out2, out3

    def _make_residual_block(self, in_dim, out_dim, res_repeat):
        layers = []
        layers.append(ConvLayer(in_dim, out_dim, kernel_size=3, stride=2))
        for _ in range(res_repeat):
            layers.append(ResBlock(out_dim))
        return nn.Sequential(*layers)


class DetectionBlock(nn.Module):
    """
    Detection Block for detecting objects
    """
    def __init__(self, in_dim: int, out_dim: int, scale: str, stride: int):
        super(DetectionBlock, self).__init__()
        assert out_dim % 2 == 0, f'out dim {out_dim} is not even'
        hidden_dim = out_dim // 2
        self.conv1 = ConvLayer(in_dim, hidden_dim, kernel_size=1)
        self.conv2 = ConvLayer(hidden_dim, out_dim, kernel_size=3)
        self.conv3 = ConvLayer(out_dim, hidden_dim, kernel_size=1)
        self.conv4 = ConvLayer(hidden_dim, out_dim, kernel_size=3)
        self.conv5 = ConvLayer(out_dim, hidden_dim, kernel_size=1)
        self.conv6 = ConvLayer(hidden_dim, out_dim, kernel_size=3)
        self.conv7 = nn.Conv2d(out_dim, LAST_LAYER_DIM, kernel_size=1, bias=True)
        self.yolo = YOLOLayer(scale, stride)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        self.branch = self.conv5(x)
        x = self.conv6(self.branch)
        x = self.conv7(x)
        x = self.yolo(x)
        return x


class Upsample(nn.Module):
    """
    Upsampling Layer to detect smaller objects in feature map
    """
    def __init__(self, scale_factor: float, mode: str='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class YOLOTail(nn.Module):

    def __init__(self):
        super(YOLOTail, self).__init__()
        self.detect1 = DetectionBlock(in_dim=1024, out_dim=1024, scale='l', stride=32)
        self.conv1 = ConvLayer(in_dim=512, out_dim=256, kernel_size=1)
        self.upsample1 = Upsample(scale_factor=2)

        self.detect2 = DetectionBlock(in_dim=768, out_dim=512, scale='m', stride=16)
        self.conv2 = ConvLayer(in_dim=256, out_dim=128, kernel_size=1)
        self.upsample2 = Upsample(scale_factor=2)

        self.detect3 = DetectionBlock(in_dim=384, out_dim=256, scale='s', stride=8)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        out1 = self.detect1(x1)
        branch1 = self.detect1.branch
        tmp = self.conv1(branch1)
        tmp = self.upsample1(tmp)
        tmp = torch.cat((tmp, x2), dim=1)

        out2 = self.detect2(tmp)
        branch2 = self.detect2.branch
        tmp = self.conv2(branch2)
        tmp = self.upsample2(tmp)
        tmp = torch.cat((tmp, x3), dim=1)
        
        out3 = self.detect3(tmp)
        
        return out1, out2, out3


class YOLOv3(nn.Module):

    def __init__(self, nms: bool=False, post: bool=True):
        super(YOLOv3, self).__init__()
        self.backbone = DarkNet53()
        self.yolo_tail = YOLOTail()
        self.nms = nms
        self.post_process = post

    def forward(self, x: torch.Tensor):
        tmp1, tmp2, tmp3 = self.backbone(x)
        out1, out2, out3 = self.yolo_tail(tmp1, tmp2, tmp3)
        out = torch.cat((out1, out2, out3), dim=1)
        return out