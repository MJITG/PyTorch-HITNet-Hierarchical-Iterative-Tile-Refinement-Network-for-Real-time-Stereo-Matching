import torch
import torch.nn as nn
import torch.nn.functional as F
from .FE import feature_extraction_conv
from .initialization import INIT
from .tile_warping import TileWarping
from .tile_update import TileUpdate, PostTileUpdate
from models.submodules import DispUpsampleBySlantedPlane, SlantDUpsampleBySlantedPlaneT4T4, SlantD2xUpsampleBySlantedPlaneT4T2
import pdb
from utils.write_pfm import write_pfm_tensor


class HITNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.feature_extractor = feature_extraction_conv(args)
        self.tile_init = INIT(args)
        self.tile_warp = TileWarping(args)
        self.tile_update0 = TileUpdate(64, 16, 32, 2, args)  # 1/16 tile refine
        self.tile_update1 = TileUpdate(128, 34, 32, 2, args)  # 1/8 tile refine
        self.tile_update2 = TileUpdate(128, 34, 32, 4, args)  # 1/4 tile refine
        self.tile_update3 = TileUpdate(128, 34, 32, 4, args)  # 1/2 tile refine
        self.tile_update4 = TileUpdate(128, 34, 32, 2, args)  # 1/1 tile refine
        self.tile_update5 = PostTileUpdate(28, 16, 32, 2, 2, SlantD2xUpsampleBySlantedPlaneT4T2(), args)  # 2/1 tile refine tile_size=2
        self.tile_update6 = PostTileUpdate(19, 16, 16, 2, 1, DispUpsampleBySlantedPlane(2, 2), args)  # 2/1 tile refine tile_size=1

        # For training phase, we need to upsample disps using slant equation
        self.prop_disp_upsample64x = DispUpsampleBySlantedPlane(64)
        self.prop_disp_upsample32x = DispUpsampleBySlantedPlane(32)
        self.prop_disp_upsample16x = DispUpsampleBySlantedPlane(16)
        self.prop_disp_upsample8x = DispUpsampleBySlantedPlane(8)
        self.prop_disp_upsample4x = DispUpsampleBySlantedPlane(4)
        self.prop_disp_upsample2x = DispUpsampleBySlantedPlane(2, 2)
        # For training phase, we need to upsample dx and dy using nearest interpolation
        self.dxdy_upsample64x = nn.UpsamplingNearest2d(scale_factor=64)
        self.dxdy_upsample32x = nn.UpsamplingNearest2d(scale_factor=32)
        self.dxdy_upsample16x = nn.UpsamplingNearest2d(scale_factor=16)
        self.dxdy_upsample8x = nn.UpsamplingNearest2d(scale_factor=8)
        self.dxdy_upsample4x = nn.UpsamplingNearest2d(scale_factor=4)
        self.dxdy_upsample2x = nn.UpsamplingNearest2d(scale_factor=2)
        # For final disparity and each supervision signal to be positive
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, left_img, right_img):
        left_fea_pyramid = self.feature_extractor(left_img)
        right_fea_pyramid = self.feature_extractor(right_img)
        init_cv_pyramid, init_tile_pyramid = self.tile_init(left_fea_pyramid, right_fea_pyramid)
        refined_tile16x = self.tile_update0(left_fea_pyramid[0], right_fea_pyramid[0], init_tile_pyramid[0])[0]
        tile_update8x = self.tile_update1(left_fea_pyramid[1], right_fea_pyramid[1], init_tile_pyramid[1], refined_tile16x)
        tile_update4x = self.tile_update2(left_fea_pyramid[2], right_fea_pyramid[2], init_tile_pyramid[2], tile_update8x[0])
        tile_update2x = self.tile_update3(left_fea_pyramid[3], right_fea_pyramid[3], init_tile_pyramid[3], tile_update4x[0])
        tile_update1x = self.tile_update4(left_fea_pyramid[4], right_fea_pyramid[4], init_tile_pyramid[4], tile_update2x[0])
        refined_tile05x = self.tile_update5(left_fea_pyramid[4], right_fea_pyramid[4], tile_update1x[0])
        refined_tile025x = self.tile_update6(left_fea_pyramid[4], right_fea_pyramid[4], refined_tile05x)
        final_disp = refined_tile025x[:, :1, :, :]

        if self.training:
            prop_disp16_fx = self.prop_disp_upsample64x(refined_tile16x[:, :1, :, :], refined_tile16x[:, 1:2, :, :], refined_tile16x[:, 2:3, :, :])
            prop_disp8_fx_cur = self.prop_disp_upsample32x(tile_update8x[1], tile_update8x[3], tile_update8x[5])
            prop_disp8_fx_pre = self.prop_disp_upsample32x(tile_update8x[2], tile_update8x[4], tile_update8x[6])
            prop_disp4_fx_cur = self.prop_disp_upsample16x(tile_update4x[1], tile_update4x[3], tile_update4x[5])
            prop_disp4_fx_pre = self.prop_disp_upsample16x(tile_update4x[2], tile_update4x[4], tile_update4x[6])
            prop_disp2_fx_cur = self.prop_disp_upsample8x(tile_update2x[1], tile_update2x[3], tile_update2x[5])
            prop_disp2_fx_pre = self.prop_disp_upsample8x(tile_update2x[2], tile_update2x[4], tile_update2x[6])
            prop_disp1_fx_cur = self.prop_disp_upsample4x(tile_update1x[1], tile_update1x[3], tile_update1x[5])
            prop_disp1_fx_pre = self.prop_disp_upsample4x(tile_update1x[2], tile_update1x[4], tile_update1x[6])
            prop_disp05_fx = self.prop_disp_upsample2x(refined_tile05x[:, :1, :, :], refined_tile05x[:, 1:2, :, :], refined_tile05x[:, 2:3, :, :])
            prop_disp_pyramid = [
                prop_disp16_fx,
                prop_disp8_fx_cur,
                prop_disp8_fx_pre,
                prop_disp4_fx_cur,
                prop_disp4_fx_pre,
                prop_disp2_fx_cur,
                prop_disp2_fx_pre,
                prop_disp1_fx_cur,
                prop_disp1_fx_pre,
                prop_disp05_fx,
                final_disp
            ]
            # WARNING: EACH PYRAMID MUST ALIGN ACCORDING TO PRE-CUR ORDER AND RESOLUTION ORDER SINCE SUPERVISION WOULDN'T SEE THE ORDER

            dx16_fx = self.dxdy_upsample64x(refined_tile16x[:, 1:2, :, :])
            dx8_fx_cur = self.dxdy_upsample32x(tile_update8x[3])
            dx8_fx_pre = self.dxdy_upsample32x(tile_update8x[4])
            dx4_fx_cur = self.dxdy_upsample16x(tile_update4x[3])
            dx4_fx_pre = self.dxdy_upsample16x(tile_update4x[4])
            dx2_fx_cur = self.dxdy_upsample8x(tile_update2x[3])
            dx2_fx_pre = self.dxdy_upsample8x(tile_update2x[4])
            dx1_fx_cur = self.dxdy_upsample4x(tile_update1x[3])
            dx1_fx_pre = self.dxdy_upsample4x(tile_update1x[4])
            dx05_fx = self.dxdy_upsample2x(refined_tile05x[:, 1:2, :, :])
            dx_pyramid = [
                dx16_fx,
                dx8_fx_cur,
                dx8_fx_pre,
                dx4_fx_cur,
                dx4_fx_pre,
                dx2_fx_cur,
                dx2_fx_pre,
                dx1_fx_cur,
                dx1_fx_pre,
                dx05_fx,
                refined_tile025x[:, 1:2, :, :]
            ]

            dy16_fx = self.dxdy_upsample64x(refined_tile16x[:, 2:3, :, :])
            dy8_fx_cur = self.dxdy_upsample32x(tile_update8x[5])
            dy8_fx_pre = self.dxdy_upsample32x(tile_update8x[6])
            dy4_fx_cur = self.dxdy_upsample16x(tile_update4x[5])
            dy4_fx_pre = self.dxdy_upsample16x(tile_update4x[6])
            dy2_fx_cur = self.dxdy_upsample8x(tile_update2x[5])
            dy2_fx_pre = self.dxdy_upsample8x(tile_update2x[6])
            dy1_fx_cur = self.dxdy_upsample4x(tile_update1x[5])
            dy1_fx_pre = self.dxdy_upsample4x(tile_update1x[6])
            dy05_fx = self.dxdy_upsample2x(refined_tile05x[:, 2:3, :, :])
            dy_pyramid = [
                dy16_fx,
                dy8_fx_cur,
                dy8_fx_pre,
                dy4_fx_cur,
                dy4_fx_pre,
                dy2_fx_cur,
                dy2_fx_pre,
                dy1_fx_cur,
                dy1_fx_pre,
                dy05_fx,
                refined_tile025x[:, 2:3, :, :]
            ]

            conf8_fx_cur = self.dxdy_upsample32x(tile_update8x[7])
            conf8_fx_pre = self.dxdy_upsample32x(tile_update8x[8])
            conf4_fx_cur = self.dxdy_upsample16x(tile_update4x[7])
            conf4_fx_pre = self.dxdy_upsample16x(tile_update4x[8])
            conf2_fx_cur = self.dxdy_upsample8x(tile_update2x[7])
            conf2_fx_pre = self.dxdy_upsample8x(tile_update2x[8])
            conf1_fx_cur = self.dxdy_upsample4x(tile_update1x[7])
            conf1_fx_pre = self.dxdy_upsample4x(tile_update1x[8])
            w_pyramid = [
                conf8_fx_cur,
                conf8_fx_pre,
                conf4_fx_cur,
                conf4_fx_pre,
                conf2_fx_cur,
                conf2_fx_pre,
                conf1_fx_cur,
                conf1_fx_pre,
            ]

            outputs = {
                "init_cv_pyramid": init_cv_pyramid,
                "prop_disp_pyramid": prop_disp_pyramid,
                "dx_pyramid": dx_pyramid,
                "dy_pyramid": dy_pyramid,
                "w_pyramid": w_pyramid,
            }
            # pdb.set_trace()

            return outputs

        else:
            prop_disp_pyramid = [final_disp]
            return {
                "prop_disp_pyramid": prop_disp_pyramid,
            }
        # pdb.set_trace()

