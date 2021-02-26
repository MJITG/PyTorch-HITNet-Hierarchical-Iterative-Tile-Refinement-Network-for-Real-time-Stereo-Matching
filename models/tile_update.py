import torch
import torch.nn as nn
import torch.nn.functional as F
from .FE import BasicConv2d
from .tile_warping import TileWarping, TileWarping1
from .submodules import DispUpsampleBySlantedPlane, SlantDUpsampleBySlantedPlaneT4T4, SlantD2xUpsampleBySlantedPlaneT4T2
import pdb
from utils.write_pfm import write_pfm_tensor


class ResBlock(nn.Module):
    """
    Residual Block without BN but with dilation
    """

    def __init__(self, inplanes, out_planes, hid_planes, add_relu=True):
        super(ResBlock, self).__init__()
        self.add_relu = add_relu
        self.conv1 = nn.Sequential(nn.Conv2d(inplanes, hid_planes, 3, 1, 1, 1),
                                   nn.LeakyReLU(inplace=True, negative_slope=0.2))

        self.conv2 = nn.Conv2d(hid_planes, out_planes, 3, 1, 1, 1)
        if add_relu:
            self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += x
        if self.add_relu:
            out = self.relu(out)
        return out


class TileUpdate(nn.Module):
    """
    Tile Update for a single resolution
    forward input: fea duo from current resolution, tile hypothesis from current and previous resolution
    forward output: refined tile hypothesis and confidence (if available)
    """
    def __init__(self, in_c, out_c, hid_c, resblk_num, args):
        super(TileUpdate, self).__init__()
        self.disp_upsample = SlantDUpsampleBySlantedPlaneT4T4(2)
        self.tile_warping = TileWarping(args)
        self.prop_warp0 = BasicConv2d(48, 16, 1, 1, 0, 1)
        self.prop_warp1 = BasicConv2d(48, 16, 1, 1, 0, 1)
        self.conv0 = BasicConv2d(in_c, hid_c, 1, 1, 0, 1)
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(ResBlock(hid_c, hid_c, hid_c))
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, 1, 1, 0, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        # For final disparity and each supervision signal to be positive
        self.relu = nn.ReLU()

    def forward(self, fea_l, fea_r, current_hypothesis, previous_hypothesis=None):
        current_tile_local_cv = self.tile_warping(current_hypothesis[:, :3, :, :], fea_l, fea_r)
        current_tile_local_cv = self.prop_warp0(current_tile_local_cv)
        aug_current_tile_hypothesis = torch.cat([current_hypothesis, current_tile_local_cv], 1)
        if previous_hypothesis is None:
            aug_hypothesis_set = aug_current_tile_hypothesis
        else:
            previous_tile_d = previous_hypothesis[:, 0, :, :].unsqueeze(1)  # multiply 2 when passing to slant upsampling
            previous_tile_dx = previous_hypothesis[:, 1, :, :].unsqueeze(1)   # h direction
            previous_tile_dy = previous_hypothesis[:, 2, :, :].unsqueeze(1)  # w direction
            up_previous_tile_d = self.disp_upsample(previous_tile_d, previous_tile_dx, previous_tile_dy)
            # pdb.set_trace()
            up_previous_tile_dx_dy = self.upsample(previous_hypothesis[:, 1:3, :, :])
            up_previous_tile_dscrpt = self.upsample(previous_hypothesis[:, 3:, :, :])
            up_previous_tile_dx_dy_dscrpt = torch.cat([up_previous_tile_dx_dy, up_previous_tile_dscrpt], dim=1)
            up_previous_tile_plane = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy_dscrpt[:, :2, :, :]], 1)
            up_previous_tile_local_cv = self.tile_warping(up_previous_tile_plane, fea_l, fea_r)
            up_previous_tile_local_cv = self.prop_warp1(up_previous_tile_local_cv)
            aug_up_previous_tile_hypothesis = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy_dscrpt, up_previous_tile_local_cv], 1)
            aug_hypothesis_set = torch.cat([aug_current_tile_hypothesis, aug_up_previous_tile_hypothesis], 1)
        tile_hypothesis_update = self.conv0(aug_hypothesis_set)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        if previous_hypothesis is None:
            refined_hypothesis = current_hypothesis + tile_hypothesis_update
            # pdb.set_trace()
            refined_hypothesis[:, :1, :, :] = F.relu(refined_hypothesis[:, :1, :, :].clone())
            # pdb.set_trace()
            return [refined_hypothesis]
        else:
            conf = tile_hypothesis_update[:, :2, :, :]  # [:, 0, :, :] is for pre
            previous_delta_hypothesis = tile_hypothesis_update[:, 2:18, :, :]
            current_delta_hypothesis = tile_hypothesis_update[:, 18:34, :, :]
            _, hypothesis_select_mask = torch.max(conf, dim=1, keepdim=True)
            hypothesis_select_mask = hypothesis_select_mask.float()
            # 1: current is larger, this mask is used to select current
            inverse_hypothesis_select_mask = 1 - hypothesis_select_mask
            # 1: previous is larger, this mask is used to select previous
            update_current_hypothesis = current_hypothesis + current_delta_hypothesis
            # tmp = F.relu(update_current_hypothesis[:, :1, :, :])
            update_current_hypothesis[:, :1, :, :] = F.relu(update_current_hypothesis[:, :1, :, :].clone()) # Force disp to be positive
            update_previous_hypothesis = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy_dscrpt], 1) + previous_delta_hypothesis
            # tmp = F.relu(update_previous_hypothesis[:, :1, :, :])
            update_previous_hypothesis[:, :1, :, :] = F.relu(update_previous_hypothesis[:, :1, :, :].clone())  # Force disp to be positive
            refined_hypothesis = hypothesis_select_mask * update_current_hypothesis + inverse_hypothesis_select_mask * update_previous_hypothesis

            pre_conf = conf[:, :1, :, :]
            cur_conf = conf[:, 1:2, :, :]
            update_current_disp = update_current_hypothesis[:, :1, :, :]
            update_previous_disp = update_previous_hypothesis[:, :1, :, :]
            update_current_dx = update_current_hypothesis[:, 1:2, :, :]
            update_previous_dx = update_previous_hypothesis[:, 1:2, :, :]
            update_current_dy = update_current_hypothesis[:, 2:3, :, :]
            update_previous_dy = update_previous_hypothesis[:, 2:3, :, :]
            # pdb.set_trace()
            return [
                refined_hypothesis,
                update_current_disp, update_previous_disp,
                update_current_dx, update_previous_dx,
                update_current_dy, update_previous_dy,
                cur_conf, pre_conf,
            ]


class PostTileUpdateNoUp(nn.Module):
    """
    No hyp upsampling, equal to pure refinement, for 1/4 res
    """
    def __init__(self, in_c, out_c, hid_c, resblk_num, args):
        super(PostTileUpdateNoUp, self).__init__()
        self.conv0 = BasicConv2d(in_c, hid_c, 1, 1, 0, 1)
        self.conv1 = BasicConv2d(hid_c, hid_c, 3, 1, 1, 1)
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(ResBlock(hid_c, hid_c, hid_c))
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, 3, 1, 1, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        # For final disparity and each supervision signal to be positive
        self.relu = nn.ReLU()

    def forward(self, fea_l, previous_hypothesis):
        # pdb.set_trace()
        guided_up_previous_tile_hypothesis = torch.cat([previous_hypothesis, fea_l], 1)
        tile_hypothesis_update = self.conv0(guided_up_previous_tile_hypothesis)
        tile_hypothesis_update = self.conv1(tile_hypothesis_update)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        refined_hypothesis = previous_hypothesis + tile_hypothesis_update
        refined_hypothesis[:, :1, :, :] = F.relu(refined_hypothesis[:, :1, :, :].clone())

        # pdb.set_trace()
        return refined_hypothesis


class PostTileUpdate(nn.Module):
    """
    Post Tile Update for a single resolution: decrease tile size, e.g. upsampling tile hypothesis, and do tile warping
    forward input: fea duo from the largest resolution, tile hypothesis from previous resolution
    forward output: refined tile hypothesis
    """
    def __init__(self, in_c, out_c, hid_c, resblk_num, slant_disp_up, args):
        super(PostTileUpdate, self).__init__()
        self.disp_upsample = slant_disp_up
        self.conv0 = BasicConv2d(in_c, hid_c, 1, 1, 0, 1)
        self.conv1 = BasicConv2d(hid_c, hid_c, 3, 1, 1, 1)
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(ResBlock(hid_c, hid_c, hid_c))
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, 3, 1, 1, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        # For final disparity and each supervision signal to be positive
        self.relu = nn.ReLU()

    def forward(self, fea_l, previous_hypothesis):
        previous_tile_d = previous_hypothesis[:, 0, :, :].unsqueeze(1)
        previous_tile_dx = previous_hypothesis[:, 1, :, :].unsqueeze(1)   # h direction
        previous_tile_dy = previous_hypothesis[:, 2, :, :].unsqueeze(1)  # w direction
        up_previous_tile_d = self.disp_upsample(previous_tile_d, previous_tile_dx, previous_tile_dy)
        up_previous_tile_dx_dy = self.upsample(previous_hypothesis[:, 1:3, :, :])
        up_previous_tile_dscrpt = self.upsample(previous_hypothesis[:, 3:, :, :])
        up_previous_tile_hypothesis = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy, up_previous_tile_dscrpt], 1)
        # pdb.set_trace()
        guided_up_previous_tile_hypothesis = torch.cat([up_previous_tile_hypothesis, fea_l], 1)
        tile_hypothesis_update = self.conv0(guided_up_previous_tile_hypothesis)
        tile_hypothesis_update = self.conv1(tile_hypothesis_update)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        refined_hypothesis = up_previous_tile_hypothesis + tile_hypothesis_update
        # tmp = F.relu(refined_hypothesis[:, :1, :, :])
        refined_hypothesis[:, :1, :, :] = F.relu(refined_hypothesis[:, :1, :, :].clone())  # Force disp to be positive

        # pdb.set_trace()
        return refined_hypothesis


class FinalTileUpdate(nn.Module):
    """
    Final Tile Update: only predicts disp
    forward input: fea duo from the largest resolution, tile hypothesis from previous resolution
    forward output: refined tile hypothesis
    """
    def __init__(self, in_c, out_c, hid_c, resblk_num, slant_disp_up, args):
        super(FinalTileUpdate, self).__init__()
        self.disp_upsample = slant_disp_up
        self.conv0 = BasicConv2d(in_c, hid_c, 1, 1, 0, 1)
        self.conv1 = BasicConv2d(hid_c, hid_c, 3, 1, 1, 1)
        resblks = nn.ModuleList()
        for i in range(resblk_num):
            resblks.append(ResBlock(hid_c, hid_c, hid_c))
        self.resblocks = nn.Sequential(*resblks)
        self.lastconv = nn.Conv2d(hid_c, out_c, 3, 1, 1, 1, bias=False)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        # For final disparity and each supervision signal to be positive
        self.relu = nn.ReLU()

    def forward(self, fea_l, previous_hypothesis):
        previous_tile_d = previous_hypothesis[:, 0, :, :].unsqueeze(1)
        previous_tile_dx = previous_hypothesis[:, 1, :, :].unsqueeze(1)   # h direction
        previous_tile_dy = previous_hypothesis[:, 2, :, :].unsqueeze(1)  # w direction
        up_previous_tile_d = self.disp_upsample(previous_tile_d, previous_tile_dx, previous_tile_dy)
        up_previous_tile_dx_dy = self.upsample(previous_hypothesis[:, 1:3, :, :])
        up_previous_tile_dscrpt = self.upsample(previous_hypothesis[:, 3:, :, :])
        up_previous_tile_hypothesis = torch.cat([up_previous_tile_d, up_previous_tile_dx_dy, up_previous_tile_dscrpt], 1)
        # pdb.set_trace()
        guided_up_previous_tile_hypothesis = torch.cat([up_previous_tile_hypothesis, fea_l], 1)
        tile_hypothesis_update = self.conv0(guided_up_previous_tile_hypothesis)
        tile_hypothesis_update = self.conv1(tile_hypothesis_update)
        tile_hypothesis_update = self.resblocks(tile_hypothesis_update)
        tile_hypothesis_update = self.lastconv(tile_hypothesis_update)
        refined_hypothesis = up_previous_tile_d+ tile_hypothesis_update
        refined_hypothesis = F.relu(refined_hypothesis.clone())  # Force disp to be positive

        # pdb.set_trace()
        return refined_hypothesis
