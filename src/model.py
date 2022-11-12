import torch
import torch.nn as nn
import torch.nn.functional as F

from config import NUM_CLASSES, NUM_ANCHORS_PER_SCALE, ANCHORS, NUM_ATTRIB, LAST_LAYER_DIM


class ConvLayer(nn.Module):
    """
    Convolutional Layer
    """
    def __init__(self, in_dim, out_dim, kernel_size, stride, slope=0.1):
        super(ConvLayer, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.leaky_relu = nn.LeakyReLU(slope)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class ResBlock(nn.Module):
    """
    Residual Block to build Darknet backbone
    """
    
    def __init__(self, in_dim):
        super(ResBlock, self).__init__()
        assert in_dim % 2 == 0, f'{in_dim} is not even'
        hidden_dim = in_dim // 2
        self.conv1 = ConvLayer(in_dim, hidden_dim, kernel_size=1)
        self.conv2 = ConvLayer(hidden_dim, in_dim, kernel_size=3)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x += residual
        return x


class YOLOLayer(nn.Module):

    def __init__(self, scale, stride):
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

    def forward(self, x):
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

    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv1 = ConvLayer(3, 32, kernel_size=3)
        self.block1 = self.make_residual_block(32, 64, res_repeat=1)
        self.block2 = self.make_residual_block(64, 128, res_repeat=2)
        self.block3 = self.make_residual_block(128, 256, res_repeat=8)
        self.block4 = self.make_residual_block(256, 512, res_repeat=8)
        self.block5 = self.make_residual_block(512, 1024, res_repeat=4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        out3 = self.block3(x)
        out2 = self.block4(x)
        out1 = self.block5(x)
        return out1, out2, out3

    def make_residual_block(self, in_dim, out_dim, res_repeat):
        layers = []
        layers.append(ConvLayer(in_dim, out_dim, kernel_size=3, stride=2))
        for _ in range(res_repeat):
            layers.append(ResBlock(out_dim))
        return nn.Sequential(*layers)


class DetectionBlock(nn.Module):

    def __init__(self, in_dim, out_dim, scale, stride):
        super(DetectionBlock, self).__init__()
        hidden_dim = in_dim // 2
        self.conv1 = ConvLayer(in_dim, hidden_dim, kernel_size=1)
        self.conv2 = ConvLayer(hidden_dim, out_dim, kernel_size=3)
        self.conv3 = ConvLayer(out_dim, hidden_dim, kernel_size=1)
        self.conv4 = ConvLayer(hidden_dim, out_dim, kernel_size=3)
        self.conv5 = ConvLayer(out_dim, hidden_dim, kernel_size=1)
        self.conv6 = ConvLayer(hidden_dim, out_dim, kernel_size=3)
        self.conv7 = nn.Conv2d(out_dim, LAST_LAYER_DIM, kernel_size=1, bias=True)
        self.yolo = YOLOLayer(scale, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        branch = self.conv5(x)
        x = self.conv6(branch)
        x = self.conv7(x)
        x = self.yolo(x)
        return x


class Upsample(nn.Module):

    def __init__(self, scale_factor, mode='nearest'):
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
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

    def forward(self, x1, x2, x3):
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

    def __init__(self, nms=False, post=True):
        self.backbone = DarkNet53()
        self.yolo_tail = YOLOTail()
        self.nms = nms
        self.post = post

    def forward(self, x):
        tmp1, tmp2, tmp3 = self.backbone(x)
        out1, out2, out3 = self.yolo_tail(tmp1, tmp2, tmp3)
        out = torch.cat((out1, out2, out3), dim=1)
        return out