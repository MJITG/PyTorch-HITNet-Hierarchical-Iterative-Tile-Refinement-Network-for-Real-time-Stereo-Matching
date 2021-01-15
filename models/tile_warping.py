import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from .submodules import DispUpsampleBySlantedPlane, BuildVolume2dChaos


class TileWarping(nn.Module):
    def __init__(self, args):
        super(TileWarping, self).__init__()
        self.disp_up = DispUpsampleBySlantedPlane(4)
        self.build_l1_volume_chaos = BuildVolume2dChaos()

    def forward(self, tile_plane: torch.Tensor, fea_l: torch.Tensor, fea_r: torch.Tensor):
        """
        local cost volume
        :param tile_plane: d, dx, dy
        :param fea_l:
        :param fea_r:
        :return: local cost volume
        """
        tile_d = tile_plane[:, 0, :, :].unsqueeze(1)
        tile_dx = tile_plane[:, 1, :, :].unsqueeze(1)   # h direction
        tile_dy = tile_plane[:, 2, :, :].unsqueeze(1)  # w direction
        local_cv = []
        for disp_d in range(-1, 2):
            flatten_local_disp_ws_disp_d = self.disp_up(tile_d + disp_d, tile_dx, tile_dy)
            cv_ws_disp_d = self.build_l1_volume_chaos(fea_l, fea_r, flatten_local_disp_ws_disp_d)
            local_cv_ws_disp_d = []  # local cost volume in one disp hypothesis [B, 16, H/4, W/4]
            for i in range(4):
                for j in range(4):
                    local_cv_ws_disp_d.append(cv_ws_disp_d[:, :, i::4, j::4])
            local_cv_ws_disp_d = torch.cat(local_cv_ws_disp_d, 1)
            local_cv.append(local_cv_ws_disp_d)  # local cost volume containing all the disp hypothesis[B, 48, H/4, W/4]
            # pdb.set_trace()
        local_cv = torch.cat(local_cv, 1)
        return local_cv


class TileWarping1(nn.Module):
    """
    Functionality same as TileWarping but with variable tile size
    """
    def __init__(self, tile_size, args):
        super(TileWarping1, self).__init__()
        self.tile_size = tile_size
        self.center = (tile_size - 1) / 2
        self.disp_up = DispUpsampleBySlantedPlane(tile_size)
        self.build_l1_volume_chaos = BuildVolume2dChaos()

    def forward(self, tile_plane: torch.Tensor, fea_l: torch.Tensor, fea_r: torch.Tensor):
        """
        local cost volume
        :param tile_plane: d, dx, dy
        :param fea_l:
        :param fea_r:
        :return: local cost volume
        """
        tile_d = tile_plane[:, 0, :, :].unsqueeze(1)
        tile_dx = tile_plane[:, 1, :, :].unsqueeze(1)   # h direction
        tile_dy = tile_plane[:, 2, :, :].unsqueeze(1)  # w direction
        local_cv = []
        for disp_d in range(-1, 2):
            flatten_local_disp_ws_disp_d = self.disp_up(tile_d + disp_d, tile_dx, tile_dy)
            cv_ws_disp_d = self.build_l1_volume_chaos(fea_l, fea_r, flatten_local_disp_ws_disp_d)
            local_cv_ws_disp_d = []  # local cost volume in one disp hypothesis [B, 16, H/4, W/4]
            for i in range(self.tile_size):
                for j in range(self.tile_size):
                    local_cv_ws_disp_d.append(cv_ws_disp_d[:, :, i::self.tile_size, j::self.tile_size])
            local_cv_ws_disp_d = torch.cat(local_cv_ws_disp_d, 1)
            local_cv.append(local_cv_ws_disp_d)  # local cost volume containing all the disp hypothesis[B, 48, H/4, W/4]
            # pdb.set_trace()
        local_cv = torch.cat(local_cv, 1)
        return local_cv
