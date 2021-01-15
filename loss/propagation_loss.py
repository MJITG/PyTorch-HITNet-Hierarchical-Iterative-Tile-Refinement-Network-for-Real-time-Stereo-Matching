import torch
import torch.nn.functional as F
import pdb
import math


def prop_loss(d_diff, A=1, alpha=1, c=0.1):
    """
    Loss from HITNet eqt(12
    :param d_diff: d^gt - d^
    :param A: The truncation value
    :param alpha: shape param
    :param c > 0: scale param
    :return: torch.Tensor: L^prop  [B*1*H*W]
    """
    rho = echo_loss(d_diff, alpha, c)
    A = torch.ones_like(rho) * A
    loss = torch.where(rho < A, rho, A)
    # pdb.set_trace()
    return loss


def echo_loss(x, alpha, c):
    """
    An amazing loss function presented in paper: A General and Adaptive Robust Loss Function (CVPR 2019).
    The name prefix 'echo' is the name of a hero in Overwatch who can become any other hero during her ultimate
    :param x: torch.Tensor
    :param alpha: shape param
    :param c > 0: scale param
    :return: torch.Tensor: loss
    """
    loss = (abs(alpha - 2) / alpha) * ((((x / c)**2) / abs(alpha - 2) + 1)**(alpha / 2) - 1)
    return loss


def slant_loss(dx, dy, dx_gt, dy_gt, d_diff, mask, B=1):
    closer_mask = d_diff < B
    mask = mask * closer_mask  # mask and
    slant_diff = torch.cat([dx_gt-dx, dy_gt-dy], dim=1)
    loss = torch.norm(slant_diff, p=1, dim=1, keepdim=True)[mask]
    return loss  # 1-dim vector


def fitting_plane(disp_gt, window):
    pass


def w_loss(conf, diff, mask, C1=1, C2=1.5):
    """

    :param conf: aka omega
    :param diff: d^gt - d^
    :param C1:
    :param C2:
    :return: torch.Tensor: loss
    """
    closer_mask = diff < C1
    further_mask = diff > C2
    mask = mask * (closer_mask + further_mask)  # mask and
    closer_item = F.relu(1 - conf)
    further_item = F.relu(conf)
    # pdb.set_trace()
    loss = closer_item * closer_mask.float() + further_item * further_mask.float()
    return loss[mask]  # 1-dim vector


# if __name__ == '__main__':
#     cost = torch.rand(1, 12, 2, 2)
#     conf = (torch.rand(1, 1, 2, 2).cuda() - 1) * 4
#     d_gt = torch.rand(1, 1, 2, 2) * 4
#     d_pred = torch.rand(1, 1, 2, 2) * 4
#     d_diff = d_gt.cuda() - d_pred.cuda()
#     prop_loss = w_loss(conf, d_diff, 1, 1)
#     pdb.set_trace()
