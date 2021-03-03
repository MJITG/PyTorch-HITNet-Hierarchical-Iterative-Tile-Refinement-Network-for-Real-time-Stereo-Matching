import torch
import torch.nn.functional as F
import pdb


def init_loss(pred_init_cost: torch.Tensor, d_gt: torch.Tensor, maxdisp, beta=1):
    """
    Initialization loss, HITNet paper eqt(10
    :param pred_init_cost:
    :param d_gt:
    :param beta:
    :return: init loss [B*1*H*W]
    """
    cost_gt = subpix_cost(pred_init_cost, d_gt, maxdisp)
    cost_nm = torch.gather(pred_init_cost, 1, get_non_match_disp(pred_init_cost, d_gt))
    loss = cost_gt + F.relu(beta - cost_nm)
    # pdb.set_trace()
    return loss


def subpix_cost(cost: torch.Tensor, disp: torch.Tensor, maxdisp: int):
    """
    phi, e.g. eqt(9) in HITNet paper
    :param cost:
    :param disp:
    :return:
    """
    # pdb.set_trace()
    disp[disp >= maxdisp - 1] = maxdisp - 2
    disp[disp < 0] = 0
    disp_floor = disp.floor()
    sub_cost = (disp - disp_floor) * torch.gather(cost, 1, disp_floor.long()+1) + (disp_floor + 1 - disp) * torch.gather(cost, 1, disp_floor.long())
    # pdb.set_trace()
    return sub_cost


def get_non_match_disp(pred_init_cost: torch.Tensor, d_gt: torch.Tensor):
    """
    HITNet paper, eqt (11)
    :param pred_init_cost: B, D, H, W
    :param d_gt: B, 1, H, W
    :return: LongTensor: min_non_match_disp: B, 1, H, W
    """
    B, D, H, W = pred_init_cost.size()
    disp_cand = torch.arange(0, D, step=1, device=d_gt.device).view(1, -1, 1, 1).repeat(B, 1, H, W).float()
    match_disp_lower_bound = d_gt - 1.5
    match_disp_upper_bound = d_gt + 1.5
    INF = torch.Tensor([float("Inf")]).view(1, 1, 1, 1).repeat(B, D, H, W).to(d_gt.device)
    tmp_cost = torch.where((disp_cand < match_disp_lower_bound) | (disp_cand > match_disp_upper_bound), pred_init_cost, INF)
    # pdb.set_trace()
    __, min_non_match_disp = torch.min(tmp_cost, dim=1, keepdim=True)
    # pdb.set_trace()
    return min_non_match_disp

#
# if __name__ == '__main__':
#     cost = torch.rand(1, 12, 2, 2)
#     d_gt = torch.rand(1, 1, 2, 2)*4
#     output_cost = init_loss(cost, d_gt)
#     pdb.set_trace()


