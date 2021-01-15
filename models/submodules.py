import torch
import torch.nn as nn
import torch.nn.functional as F


class DispUpsampleBySlantedPlane(nn.Module):
    def __init__(self, upscale, ts=4):
        super(DispUpsampleBySlantedPlane, self).__init__()
        self.upscale = upscale
        self.center = (upscale - 1) / 2
        self.DUC = nn.PixelShuffle(upscale)
        self.ts = ts

    def forward(self, tile_disp, tile_dx, tile_dy):
        tile_disp = tile_disp * (self.upscale / self.ts)
        disp0 = []  # for each pixel, upsampled disps are stored in channel dimension
        for i in range(self.upscale):
            for j in range(self.upscale):
                disp0.append(tile_disp + (i - self.center) * tile_dx + (j - self.center) * tile_dy)
        disp0 = torch.cat(disp0, 1)  # [B, upscale**2, H/upscale, W/upscale]
        disp1 = self.DUC(disp0)  # [B, 1, H/1, W/1]
        return disp1


class SlantDUpsampleBySlantedPlaneT4T4(nn.Module):
    """
    Slant map upsampling, input tile size = 4x4, output tile size = 4x4
    """
    def __init__(self, upscale):
        super(SlantDUpsampleBySlantedPlaneT4T4, self).__init__()
        self.upscale = upscale
        self.center = 4 * (upscale - 1) / 2
        self.DUC = nn.PixelShuffle(upscale)

    def forward(self, tile_disp, tile_dx, tile_dy):
        tile_disp = tile_disp * self.upscale
        disp0 = []  # for each pixel, upsampled disps are stored in channel dimension
        for i in range(self.upscale):
            for j in range(self.upscale):
                disp0.append(tile_disp + (i * 4 - self.center) * tile_dx + (j * 4 - self.center) * tile_dy)
        disp0 = torch.cat(disp0, 1)  # [B, upscale**2, H/upscale, W/upscale]
        disp1 = self.DUC(disp0)  # [B, 1, H/1, W/1]
        return disp1


class SlantD2xUpsampleBySlantedPlaneT4T2(nn.Module):
    """
    Slant map upsampling 2x, input tile size = 4x4, output tile size = 2x2
    """
    def __init__(self):
        super(SlantD2xUpsampleBySlantedPlaneT4T2, self).__init__()
        self.DUC = nn.PixelShuffle(2)

    def forward(self, tile_disp, tile_dx, tile_dy):
        disp0 = []  # for each pixel, upsampled disps are stored in channel dimension
        for i in range(2):
            for j in range(2):
                disp0.append(tile_disp + (i * 2 - 1) * tile_dx + (j * 2 - 1) * tile_dy)
        disp0 = torch.cat(disp0, 1)  # [B, upscale**2, H/upscale, W/upscale]
        disp1 = self.DUC(disp0)  # [B, 1, H/1, W/1]
        return disp1


class BuildVolume2d(nn.Module):
    def __init__(self, maxdisp):
        super(BuildVolume2d, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, feat_l, feat_r):
        padded_feat_r = F.pad(feat_r, [self.maxdisp-1, 0, 0, 0])
        cost = torch.zeros((feat_l.size()[0], self.maxdisp, feat_l.size()[2], feat_l.size()[3]), device='cuda')
        for i in range(0, self.maxdisp):
            if i > 0:
                # pdb.set_trace()
                cost[:, i, :, :] = torch.norm(feat_l[:, :, :, :] - padded_feat_r[:, :, :, self.maxdisp-1-i:-i:4], 1, 1)
            else:
                # pdb.set_trace()
                cost[:, i, :, :] = torch.norm(feat_l[:, :, :, :] - padded_feat_r[:, :, :, self.maxdisp-1::4], 1, 1)

        return cost.contiguous()  # B*D*H*W


class BuildVolume2dChaos(nn.Module):
    def __init__(self):
        super(BuildVolume2dChaos, self).__init__()

    def forward(self, refimg_fea, targetimg_fea, disps):
        B, C, H, W = refimg_fea.shape
        batch_disp = torch.unsqueeze(disps, dim=2).view(-1, 1, H, W)
        batch_feat_l = refimg_fea[:, None, :, :, :].repeat(1, disps.shape[1], 1, 1, 1).view(-1, C, H, W)
        batch_feat_r = targetimg_fea[:, None, :, :, :].repeat(1, disps.shape[1], 1, 1, 1).view(-1, C, H, W)
        warped_batch_feat_r = warp(batch_feat_r, batch_disp)
        volume = torch.norm(batch_feat_l - warped_batch_feat_r, 1, 1).view(B, disps.shape[1], H, W)
        volume = volume.contiguous()
        return volume


def warp(x, disp):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    vgrid = torch.cat((xx, yy), 1).float()

    # vgrid = Variable(grid)
    vgrid[:,:1,:,:] = vgrid[:,:1,:,:] - disp

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid)
    return output
