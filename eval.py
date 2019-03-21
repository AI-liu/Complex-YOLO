from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
from scipy import misc

from utils import *


def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)


def get_region_boxes(x, conf_thresh, num_classes, anchors, num_anchors):
    if x.dim() == 3:
        x = x.unsqueeze(0)

    assert (x.size(1) == (7 + num_classes) * num_anchors)

    nA = num_anchors  # num_anchors = 5
    nB = x.data.size(0)
    nC = num_classes  # num_classes = 8
    nH = x.data.size(2)  # nH  16
    nW = x.data.size(3)  # nW  32

    # Tensors for cuda support
    FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
    ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

    prediction = x.view(nB, nA, 7+num_classes, nH, nW).permute(0, 1, 3, 4, 2).contiguous()

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
    scaled_anchors = FloatTensor([(a_w , a_h ) for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

    # Add offset and scale with anchors
    pred_boxes = FloatTensor(prediction.shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

    pred_boxes[..., 6] = pred_conf
    pred_boxes[..., 7:(7 + nC) ] = pred_cls

    pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, (7 + nC)))  # torch.Size([2560, 15])

    all_boxes = []
    for i in range(2560):
        if pred_boxes[i][6] > conf_thresh:
            all_boxes.append(pred_boxes[i])
            # print(pred_boxes[i])
    return all_boxes


# classes
# class_list = ['Car', 'Van' , 'Truck' , 'Pedestrian' , 'Person_sitting' , 'Cyclist' , 'Tram' ]


bc = {}
bc['minX'] = 0;
bc['maxX'] = 80;
bc['minY'] = -40;
bc['maxY'] = 40
bc['minZ'] = -2;
bc['maxZ'] = 1.25

for file_i in range(6030,6230):
    test_i = str(file_i).zfill(6)

    lidar_file = '/home/ai/KITTI/training/velodyne/' + test_i + '.bin'
    calib_file = '/home/ai/KITTI/training/calib/' + test_i + '.txt'
    label_file = '/home/ai/KITTI/training/label_2/' + test_i + '.txt'

    # load target data
    calib = load_kitti_calib(calib_file)
    target = get_target(label_file, calib['Tr_velo2cam'])
    # print(target)

    # load point cloud data
    a = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
    b = removePoints(a, bc)
    rgb_map = makeBVFeature(b, bc, 40 / 512)
    misc.imsave('eval_bv.png', rgb_map)

    # load trained model  and  forward
    input = torch.from_numpy(rgb_map)  # (512, 1024, 3)
    input = input.reshape(1, 3, 512, 1024)
    model = torch.load('ComplexYOLO_epoch100')
    model.cuda()
    output = model(input.float().cuda())  # torch.Size([1, 75, 16, 32])

    # eval result
    conf_thresh = 0.7
    nms_thresh = 0.4
    num_classes = int(8)
    num_anchors = int(5)
    img = cv2.imread('eval_bv.png')

    all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)

    for i in range(len(all_boxes)):
        pred_img_y = int(all_boxes[i][0] * 1024.0 / 32.0)  # 32 cell = 1024 pixels
        pred_img_x = int(all_boxes[i][1] * 512.0 / 16.0)  # 16 cell = 512 pixels
        pred_img_width = int(all_boxes[i][2] * 1024.0 / 32.0)  # 32 cell = 1024 pixels
        pred_img_height = int(all_boxes[i][3] * 512.0 / 16.0)  # 16 cell = 512 pixels

        rect_top1 = int(pred_img_y - pred_img_width / 2)
        rect_top2 = int(pred_img_x - pred_img_height / 2)
        rect_bottom1 = int(pred_img_y + pred_img_width / 2)
        rect_bottom2 = int(pred_img_x + pred_img_height / 2)
        cv2.rectangle(img, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (255, 0, 0), 1)

    # for j in range(50):
    #     if target[j][1] == 0:
    #         break
    #     img_y = int(target[j][1] * 1024.0)  # 32 cell = 1024 pixels
    #     img_x = int(target[j][2] * 512.0)  # 16 cell = 512 pixels
    #     img_width = int(target[j][3] * 1024.0)  # 32 cell = 1024 pixels
    #     img_height = int(target[j][4] * 512.0)  # 16 cell = 512 pixels
    #
    #     rect_top1 = int(img_y - img_width / 2)
    #     rect_top2 = int(img_x - img_height / 2)
    #     rect_bottom1 = int(img_y + img_width / 2)
    #     rect_bottom2 = int(img_x + img_height / 2)
    #     cv2.rectangle(img, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (0, 0, 255), 1)

    misc.imsave('eval_bv' + test_i + '.png', img)
