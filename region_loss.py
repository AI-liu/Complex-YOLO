from __future__ import division
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

from utils import *


def build_targets(pred_boxes,pred_conf, pred_cls, target, anchors, num_anchors, num_classes, nH, nW, ignore_thres):
    nB = target.size(0)
    nA = num_anchors   #5
    nC = num_classes   #8
    mask = torch.zeros(nB,nA,nH,nW)
    conf_mask  = torch.ones(nB, nA, nH, nW)
    tx         = torch.zeros(nB, nA, nH, nW)
    ty         = torch.zeros(nB, nA, nH, nW) 
    tw         = torch.zeros(nB, nA, nH, nW) 
    tl         = torch.zeros(nB, nA, nH, nW)
    tim        = torch.zeros(nB, nA, nH, nW)
    tre        = torch.zeros(nB, nA, nH, nW)
    tconf      = torch.ByteTensor(nB, nA, nH, nW).fill_(0)
    tcls       = torch.ByteTensor(nB, nA, nH, nW , nC).fill_(0)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b][t].sum() == 0:
                continue

            nGT += 1
            # Convert to position relative to box
            gx = target[b, t, 1] * nW
            gy = target[b, t, 2] * nH
            gw = target[b, t, 3] * nW
            gl = target[b, t, 4] * nH
            #gim = target[b][t][5]
            #gre = target[b][t][6]

            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gl])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shapes
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Where the overlap is larger than threshold set mask to zero (ignore)
            conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            # Get ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gl])).unsqueeze(0)
            # Get the best prediction
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
            # Masks
            mask[b, best_n, gj, gi] = 1
            conf_mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            tl[b, best_n, gj, gi] = math.log(gl / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            target_label = int(target[b, t, 0])
            tcls[b, best_n, gj, gi, target_label] = 1
            tconf[b, best_n, gj, gi] = 1

            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
            score = pred_conf[b, best_n, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                nCorrect += 1

    return nGT, nCorrect, mask, conf_mask, tx, ty, tw, tl, tconf, tcls


class RegionLoss(nn.Module):
    def __init__(self, num_classes=8, num_anchors=5):
        super(RegionLoss, self).__init__()

        self.anchors = anchors
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.bbox_attrs = 7+num_classes
        self.ignore_thres = 0.6
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss



    def forward(self, x, targets):
        #x : batch_size*num_anchorsx(6+1+num_classes)*H*W    [12,75,16,32]
        #targets :   targets define in utils.py  get_target function   [12,50,7]

        nA = self.num_anchors     # num_anchors = 5
        nB = x.data.size(0)  # batch_size
        nH = x.data.size(2)  # nH  16
        nW = x.data.size(3)  # nW  32


        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nH, nW).permute(0, 1, 3, 4, 2).contiguous()  # prediction [12,5,16,32,15]

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 6])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 7:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nW).repeat(nH, 1).view([1, 1, nH, nW]).type(FloatTensor)
        grid_y = torch.arange(nH).repeat(nW, 1).t().view([1, 1, nH, nW]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w , a_h ) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h



        if x.is_cuda:
            self.mse_loss = self.mse_loss.cuda()
            self.bce_loss = self.bce_loss.cuda()
            self.ce_loss = self.ce_loss.cuda()

        nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
            pred_boxes=pred_boxes.cpu().data,
            pred_conf=pred_conf.cpu().data,
            pred_cls=pred_cls.cpu().data,
            target=targets.cpu().data,
            anchors=scaled_anchors.cpu().data,
            num_anchors=nA,
            num_classes=self.num_classes,
            nH=nH,
            nW=nW,
            ignore_thres=self.ignore_thres
        )

        nProposals = int((pred_conf > 0.5).sum().item())
        recall = float(nCorrect / nGT) if nGT else 1
        precision = float(nCorrect / nProposals)

        # Handle masks
        mask = Variable(mask.type(ByteTensor))
        conf_mask = Variable(conf_mask.type(ByteTensor))

        # Handle target variables
        tx = Variable(tx.type(FloatTensor), requires_grad=False)
        ty = Variable(ty.type(FloatTensor), requires_grad=False)
        tw = Variable(tw.type(FloatTensor), requires_grad=False)
        th = Variable(th.type(FloatTensor), requires_grad=False)
        tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
        tcls = Variable(tcls.type(LongTensor), requires_grad=False)

        # Get conf mask where gt and where there is no gt
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask

        # Mask outputs to ignore non-existing objects
        loss_x = self.mse_loss(x[mask], tx[mask])
        loss_y = self.mse_loss(y[mask], ty[mask])
        loss_w = self.mse_loss(w[mask], tw[mask])
        loss_h = self.mse_loss(h[mask], th[mask])
        loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
            pred_conf[conf_mask_true], tconf[conf_mask_true]
        )
        loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
        loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

        print('nGT %d, recall %f, precision %f, proposals %d, loss: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f' % \
                 (nGT, recall,  precision,  nProposals, loss_x.data, loss_y.data, loss_w.data, loss_h.data, loss_conf.data, loss_cls.data,loss.data))

        return loss

