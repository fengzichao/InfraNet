import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):
    return (1.0 - 0.5 * eps, 0.5 * eps)


class BCEBlurWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)
        dx = pred - true
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 0.0001))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(nn.Module):

    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ComputeLoss:
    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        h = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = (FocalLoss(BCEcls, g), FocalLoss(BCEobj, g))
        m = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (BCEcls, BCEobj, 1.0, h, autobalance)
        self.nc = m.nc
        self.nl = m.nl
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):
        bs = p[0].shape[0]
        loss = torch.zeros(3, device=self.device)
        tcls, tbox, indices = self.build_targets(p, targets)
        for i, pi in enumerate(p):
            b, gj, gi = indices[i]
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)
            n_labels = b.shape[0]
            if n_labels:
                pxy, pwh, _, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 1)
                pxy = pxy.sigmoid() * 1.6 - 0.3
                pwh = (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                loss[0] += (1.0 - iou).mean()
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, gj, gi, iou = (b[j], gj[j], gi[j], iou[j])
                if self.gr < 1:
                    iou = 1.0 - self.gr + self.gr * iou
                tobj[b, gj, gi] = iou
                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(n_labels), tcls[i]] = self.cp
                    loss[2] += self.BCEcls(pcls, t)
            obji = self.BCEobj(pi[:, 4], tobj)
            loss[1] += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        loss[0] *= self.hyp['box']
        loss[1] *= self.hyp['obj']
        loss[2] *= self.hyp['cls']
        return (loss.sum() * bs, loss.detach())

    def build_targets(self, p, targets):
        nt = targets.shape[0]
        tcls, tbox, indices = ([], [], [])
        gain = torch.ones(6, device=self.device)
        g = 0.3
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float() * g
        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            t = targets * gain
            if nt:
                r = t[..., 4:6] / self.anchors[i]
                j = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']
                t = t[j]
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            bc, gxy, gwh = t.chunk(3, 1)
            b, c = bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            tcls.append(c)
        return (tcls, tbox, indices)


class ComputeLoss_NEW:
    sort_obj_iou = False

    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device
        h = model.hyp
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))
        g = h['fl_gamma']
        if g > 0:
            BCEcls, BCEobj = (FocalLoss(BCEcls, g), FocalLoss(BCEobj, g))
        m = de_parallel(model).model[-1]
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = (BCEcls, BCEobj, 1.0, h, autobalance)
        self.nc = m.nc
        self.nl = m.nl
        self.anchors = m.anchors
        self.device = device
        self.BCE_base = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, p, targets):
        tcls, tbox, indices = self.build_targets(p, targets)
        bs = p[0].shape[0]
        n_labels = targets.shape[0]
        loss = torch.zeros(3, device=self.device)
        all_loss = []
        for i, pi in enumerate(p):
            b, gj, gi = indices[i]
            if n_labels:
                pxy, pwh, pobj, pcls = pi[b, :, gj, gi].split((2, 2, 1, self.nc), 2)
                pbox = torch.cat((pxy.sigmoid() * 1.6 - 0.3, (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i]), 2)
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()
                obj_target = iou.detach().clamp(0).type(pi.dtype)
                all_loss.append([(1.0 - iou) * self.hyp['box'], self.BCE_base(pobj.squeeze(), torch.ones_like(obj_target)) * self.hyp['obj'], self.BCE_base(pcls, F.one_hot(tcls[i], self.nc).float()).mean(2) * self.hyp['cls'], obj_target, tbox[i][..., 2] > 0.0])
        n_assign = 4
        cat_loss = [torch.cat(x, 1) for x in zip(*all_loss)]
        ij = torch.zeros_like(cat_loss[0]).bool()
        sum_loss = cat_loss[0] + cat_loss[2]
        for col in torch.argsort(sum_loss, dim=1).T[:n_assign]:
            ij[range(n_labels), col] = cat_loss[4][range(n_labels), col]
        loss[0] = cat_loss[0][ij].mean() * self.nl
        loss[2] = cat_loss[2][ij].mean() * self.nl
        for i, (h, pi) in enumerate(zip(ij.chunk(self.nl, 1), p)):
            b, gj, gi = indices[i]
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)
            if n_labels:
                tobj[b[h], gj[h], gi[h]] = all_loss[i][3][h]
            loss[1] += self.BCEobj(pi[:, 4], tobj) * (self.balance[i] * self.hyp['obj'])
        return (loss.sum() * bs, loss.detach())

    def build_targets(self, p, targets):
        nt = targets.shape[0]
        tcls, tbox, indices = ([], [], [])
        gain = torch.ones(6, device=self.device)
        g = 0.3
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=self.device).float()
        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]
            t = targets * gain
            if nt:
                r = t[..., 4:6] / self.anchors[i]
                a = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']
                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m)) & a
                t = t.repeat((5, 1, 1))
                offsets = torch.zeros_like(gxy)[None] + off[:, None]
                t[..., 4:6][~j] = 0.0
            else:
                t = targets[0]
                offsets = 0
            bc, gxy, gwh = t.chunk(3, 2)
            b, c = bc.long().transpose(0, 2).contiguous()
            gij = (gxy - offsets).long()
            gi, gj = gij.transpose(0, 2).contiguous()
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 2).permute(1, 0, 2).contiguous())
            tcls.append(c)
        return (tcls, tbox, indices)
