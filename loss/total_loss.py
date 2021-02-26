import torch
import torch.nn.functional as F
from loss.initialization_loss import init_loss
from loss.propagation_loss import prop_loss, slant_loss, w_loss


def global_loss(init_cv_cost_pyramid, prop_disp_pyramid, dx_pyramid, dy_pyramid, w_pyramid,
                d_gt, dx_gt, dy_gt, maxdisp,
                lambda_init=1, lambda_prop=1, lambda_slant=1, lambda_w=1):
    """

    :param init_cv_cost_pyramid:
    :param prop_disp_pyramid:
    :param slant_pyramid:
    :param w_pyramid:
    :param d_gt:
    :param maxdisp:
    :param loss_init:
    :param loss_prop:
    :param loss_slant:
    :param loss_w:
    :param lambda_init:
    :param lambda_prop:
    :param lambda_slant:
    :param lambda_w:
    :return:
    """
    # if len(d_gt.shape) == 3:
    #     d_gt = d_gt.unsqueeze(1)
    # if len(dx_gt.shape) == 3:
    #     dx_gt = dx_gt.unsqueeze(1)
    # if len(dy_gt.shape) == 3:
    #     dy_gt = dy_gt.unsqueeze(1)

    d_gt_pyramid = []
    for i in range(len(init_cv_cost_pyramid)):
        scale = 4 * (2 ** i)  # 4,8,16,32,64
        d_gt_pyramid.append(torch.nn.MaxPool2d(scale, scale)(d_gt)/(scale/4))
    d_gt_pyramid.reverse()  # disp ground truth generation. From small to large.

    init_loss_pyramid = []
    for i, cv in enumerate(init_cv_cost_pyramid):
        # pdb.set_trace()
        mask = (d_gt_pyramid[i] > 0) & (d_gt_pyramid[i] < maxdisp/(2**(len(init_cv_cost_pyramid)-1-i)))
        init_loss_pyramid.append(
            lambda_init * init_loss(cv, d_gt_pyramid[i])[mask]
        )
        # pdb.set_trace()
    init_loss_vec = torch.cat(init_loss_pyramid, dim=0)  # 1-dim vector
    # pdb.set_trace()

    prop_loss_pyramid = []  # masked
    prop_diff_pyramid = []  # not masked
    mask = (d_gt > 0) & (d_gt < maxdisp)
    prop_loss_weights = [1/64, 1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1]
    for i, disp in enumerate(prop_disp_pyramid):
        prop_diff_pyramid.append(
            torch.abs(d_gt - disp)
        )
        prop_loss_pyramid.append(
            lambda_prop * prop_loss_weights[i] * prop_loss(prop_diff_pyramid[-1], 10000)[mask]
        )
        # pdb.set_trace()
    prop_loss_vec = torch.cat(prop_loss_pyramid, dim=0)
    # pdb.set_trace()

    slant_loss_pyramid = []
    slant_loss_weights = [1/64, 1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2]
    for i in range(len(dx_pyramid)):
        # print(i)
        slant_loss_pyramid.append(
            lambda_slant * slant_loss_weights[i] * slant_loss(dx_pyramid[i], dy_pyramid[i], dx_gt, dy_gt, prop_diff_pyramid[i], mask)
        )
    slant_loss_vec = torch.cat(slant_loss_pyramid, dim=0)
    # pdb.set_trace()

    w_loss_pyramid = []
    w_loss_weights = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/4, 1/4]
    for i, w in enumerate(w_pyramid):
        w_loss_pyramid.append(
            lambda_w * w_loss_weights[i] * w_loss(w, prop_diff_pyramid[i+1], mask)  # index for prop_diff_pyramid plus 1 since there is no confidence at 1st level
        )
    w_loss_vec = torch.cat(w_loss_pyramid, dim=0)
    # pdb.set_trace()

    total_loss_vec = torch.cat([init_loss_vec, prop_loss_vec, slant_loss_vec, w_loss_vec], dim=0)
    # pdb.set_trace()

    return torch.mean(total_loss_vec)


# def total_loss(init_cv_pyramid, prop_disp_pyramid, slant_pyramid, w_pyramid, d_gt_pyramid):
#     """
#     calculate final loss
#
#     :param init_cv_pyramid: output of init module of network. None in the post-prop (16x, 8x, 4x, 2x, 1x)
#     :param prop_disp_pyramid: output hypothesis disparity of prop module(64x, 32x, 16x, 8x, 4x, 2x, 1x)
#     :param slant_pyramid: dx and dy of slants(64x, 32x, 16x, 8x, 4x, 2x, 1x)
#     :param w_pyramid: confidence. none in the first and post-prop(32x, 16x, 8x, 4x)
#     :param d_gt_pyramid: disparity groundtruth pyramid (from small to large)(64x, 32x, 16x, 8x, 4x, 2x, 1x)
#     :return: scalar, weighted sum of all loss
#     """
#     for i, hyp in enumerate(d_gt_pyramid):
#         if i == 0:
#             loss_init = ini
#         loss_vec = loss.reshape(-1)
#         if i == 0:
#             all_loss_vec = loss_vec
#         else:
#             all_loss_vec = torch.cat([all_loss_vec, loss_vec], 0)
#
#     return torch.mean(all_loss_vec)


if __name__ == '__main__':
    import pdb
    import os
    import time

    os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    maxdisp = 256
    img_h = 256
    img_w = 512
    bs = 4
    M = 5  # corresponding to M=4 in paper
    init_cv_cost_pyramid = []
    for i in range(M):
        scale = 2**(M-i-1)
        init_cv_cost_pyramid.append(
            torch.rand(bs, maxdisp//scale, img_h//(scale*4), img_w//(scale*4)).cuda() * 5.
        )

    prop_disp_pyramid = []
    dx_pyramid = []
    dy_pyramid = []
    for i in range(M+2):
        prop_disp_pyramid.append(
            (torch.rand(bs, 1, img_h, img_w).cuda() - 0.5) * 10.
        )
        dx_pyramid.append(
            torch.rand(bs, 1, img_h, img_w).cuda() - 0.5
        )
        dy_pyramid.append(
            torch.rand(bs, 1, img_h, img_w).cuda() - 0.5
        )

    w_pyramid = []
    for i in range(M-1):
        w_pyramid.append(
            torch.rand(bs, 1, img_h, img_w).cuda()
        )

    d_gt = (torch.rand(bs, 1, img_h, img_w).cuda() - 0.5) * 10.
    dx_gt = torch.rand(bs, 1, img_h, img_w).cuda() - 0.5
    dy_gt = torch.rand(bs, 1, img_h, img_w).cuda() - 0.5
    for _ in range(1):
        st_time = time.time()
        loss = global_loss(init_cv_cost_pyramid, prop_disp_pyramid, dx_pyramid, dy_pyramid, w_pyramid,
                           d_gt, dx_gt, dy_gt, maxdisp)
        print('Time: {:.3f}'.format(time.time() - st_time))
    pdb.set_trace()

