
import torch
import torch.nn as nn


def compute_distillation_output_loss(p, t_p, model, dist_loss="l2", T=20, reg_norm=None, distill_ratio=0.5):

    t_ft = torch.cuda.FloatTensor if t_p[0].is_cuda else torch.Tensor
    t_lcls, t_lbox, t_lobj = t_ft([0]), t_ft([0]), t_ft([0])
    h = model.hyp  # hyperparameters
    red = 'mean'  # Loss reduction (sum or mean)
    if red != "mean":
        raise NotImplementedError(
            "reduction must be mean in distillation mode!")

    DboxLoss = nn.MSELoss(reduction="none")
    if dist_loss == "l2":
        DclsLoss = nn.MSELoss(reduction="none")
    elif dist_loss == "kl":
        DclsLoss = nn.KLDivLoss(reduction="none")
    else:
        DclsLoss = nn.BCEWithLogitsLoss(reduction="none")
    DobjLoss = nn.MSELoss(reduction="none")
    # per output
    for i, pi in enumerate(p):  # layer index, layer predictions
        t_pi = t_p[i]
        t_obj_scale = t_pi[..., 4].sigmoid()

        # BBox
        b_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1, 1, 1, 1, 4)
        if not reg_norm:
            t_lbox += torch.mean(DboxLoss(pi[..., :4],
                                          t_pi[..., :4]) * b_obj_scale)
        else:
            wh_norm_scale = reg_norm[i].unsqueeze(
                0).unsqueeze(-2).unsqueeze(-2)
            t_lbox += torch.mean(DboxLoss(pi[..., :2].sigmoid(),
                                          t_pi[..., :2].sigmoid()) * b_obj_scale)
            t_lbox += torch.mean(DboxLoss(pi[..., 2:4].sigmoid(),
                                          t_pi[..., 2:4].sigmoid() * wh_norm_scale) * b_obj_scale)

        # Class
        if model.nc > 1:  # cls loss (only if multiple classes)
            c_obj_scale = t_obj_scale.unsqueeze(-1).repeat(1,
                                                           1, 1, 1, model.nc)
            if dist_loss == "kl":
                kl_loss = DclsLoss(F.log_softmax(pi[..., 5:]/T, dim=-1),
                                   F.softmax(t_pi[..., 5:]/T, dim=-1)) * (T * T)
                t_lcls += torch.mean(kl_loss * c_obj_scale)
            else:
                t_lcls += torch.mean(DclsLoss(pi[..., 5:],
                                              t_pi[..., 5:]) * c_obj_scale)

        t_lobj += torch.mean(DobjLoss(pi[..., 4], t_pi[..., 4]) * t_obj_scale)
    t_lbox *= h['box'] * distill_ratio
    t_lobj *= h['obj'] * distill_ratio
    t_lcls *= h['cls'] * distill_ratio
    bs = p[0].shape[0]  # batch size
    dloss = (t_lobj + t_lbox + t_lcls) * bs
    return dloss