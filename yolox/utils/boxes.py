#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math
import shapely.geometry
import numpy as np
import shapely
import torch
import torchvision
import cv2
__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # box_corner = prediction.new(prediction.shape)
    # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    j = 0
    for i, image_pred in enumerate(prediction):
        conf_thre = 0.01

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 9: 9 + num_classes], 1, keepdim=True)
        # image_pred[:, 0], image_pred[:, 2], image_pred[:, 4], image_pred[:, 6], = image_pred[:, 0].sigmoid()*640, image_pred[:, 2].sigmoid()*640, image_pred[:, 4].sigmoid()*640, image_pred[:, 6].sigmoid()*640
        # image_pred[:, 1], image_pred[:, 3], image_pred[:, 5], image_pred[:, 7], = image_pred[:, 1].sigmoid()*480, image_pred[:, 3].sigmoid()*480, image_pred[:, 5].sigmoid()*480, image_pred[:, 7].sigmoid()*480

        conf_mask = (image_pred[:, 8] * class_conf.squeeze() >= conf_thre).squeeze()

        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :9], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        if not detections.size(0):
            continue
        detection = detections.cpu().numpy().tolist()
        best_box = polygon_nms(detection, 0.1)
        # best_box = best_box.tolist()
        img_path1 = '/home/zq/DynamicX/POLYGON-YOLOX-3090/polygon_data/JPEGImages/2337.jpg'
        img_path2 = '/home/zq/DynamicX/POLYGON-YOLOX-3090/polygon_data/JPEGImages/5927.jpg'
        img_path3 = '/home/zq/DynamicX/POLYGON-YOLOX-3090/polygon_data/JPEGImages/5993.jpg'
        img_path4 = '/home/zq/DynamicX/POLYGON-YOLOX-3090/polygon_data/JPEGImages/4535.jpg'
        if j == 0:
            img_path = img_path1
        if j == 1:
            img_path = img_path2
        if j == 2:
            img_path = img_path3
        if j == 3:
            img_path = img_path4


        # img_path += str(k+1)
        save_path = 'pred_img' + str(j) + '.jpg'
        img = cv2.imread(img_path)
        r = 0.325
        for i in range(len(best_box)):
            cv2.putText(img, str(best_box[i][9]), (int(best_box[i][0]), int(best_box[i][3])+20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 2)
            cv2.line(img, (int(best_box[i][0]/0.325), int(best_box[i][1]/0.325)), (int(best_box[i][2]/0.325), int(best_box[i][3]/0.325)), (0, 0, 255), 3)
            cv2.line(img, (int(best_box[i][2]/0.325), int(best_box[i][3]/0.325)), (int(best_box[i][4]/0.325), int(best_box[i][5]/0.325)), (0, 0, 255), 3)
            cv2.line(img, (int(best_box[i][4]/0.325), int(best_box[i][5]/0.325)), (int(best_box[i][6]/0.325), int(best_box[i][7]/0.325)), (0, 0, 255), 3)
            cv2.line(img, (int(best_box[i][6]/0.325), int(best_box[i][7]/0.325)), (int(best_box[i][0]/0.325), int(best_box[i][1]/0.325)), (0, 0, 255), 3)
        cv2.imwrite(save_path, img)
        j += 1
        continue
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

def polygon_nms(detection, nms_thre):

    detection = np.array(detection)
    prediction = detection


    score = prediction[:, 8]*prediction[:, 9]


    mask = score > nms_thre
    detection = prediction[mask]




    class_conf = np.expand_dims(np.max(detection[:, 9:10], axis=-1), axis=-1)

    # class_pred = np.expand_dims(np.argmax(detection[:, 9:10], axis=-1), axis=-1)
    class_pred = detection[:, 10:]


    detections = np.concatenate([detection[:, :8], class_conf, class_pred], axis=-1)
    unique_class = np.unique(detection[:, -1])
    if (len(unique_class) == 0):
        print(66)

    best_box = []


    output = []
    for c in unique_class:

        cls_mask = detections[:, -1] == c
        detection = detections[cls_mask]
        scores = detection[:, 8]
        arg_sort = np.argsort(scores)[::-1]
        detection = detection[arg_sort]
        cls_box1 = []
        cls_box2 = []

        cls_box1.append(detection[0])

        while len(detection) != 0:

            best_box.append(detection[0])
            if len(detection) == 1:
                break
            ious = []
            for i in range(len(detection)-1):
                iou = polygon_box_iou(torch.from_numpy(detection[0][:8]), torch.from_numpy(detection[i+1][:8])).squeeze(0).squeeze(0).numpy()
                ious.append(iou)
                # if iou < 0.65:
                #     cls_box2.append(detection[i+1])
            detection = detection[1:][np.array(ious) < 0.65]

    print(best_box)
    return(best_box)
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes

def polygon_inter_union_cpu(boxes1, boxes2):
    """
        Reference: https://github.com/ming71/yolov3-polygon/blob/master/utils/utils.py ;
        iou computation (polygon) with cpu;
        Boxes have shape nx8 and Anchors have mx8;
        Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
    """

    n, m = boxes1.shape[0], boxes2.shape[0]
    inter = torch.zeros(n, m)
    union = torch.zeros(n, m)
    for i in range(n):
        polygon1 = shapely.geometry.Polygon(boxes1[i, :].view(4, 2)).convex_hull
        for j in range(m):
            polygon2 = shapely.geometry.Polygon(boxes2[j, :].view(4, 2)).convex_hull
            if polygon1.intersects(polygon2):
                try:
                    inter[i, j] = polygon1.intersection(polygon2).area
                    union[i, j] = polygon1.union(polygon2).area
                except shapely.geos.TopologicalError:
                    print('shapely.geos.TopologicalError occured')
    return inter, union


def polygon_box_iou(boxes1, boxes2, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu", ordered=False):
    """
        Compute iou of polygon boxes via cpu or cuda;
        For cuda code, please refer to files in ./iou_cuda
        Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
    """
    # For testing this function, please use ordered=False
    if not ordered:
        boxes1, boxes2 = order_corners(boxes1.clone().to(device)), order_corners(boxes2.clone().to(device))
    else:
        boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)

    # if torch.cuda.is_available() and and boxes1.is_cuda:
        # using cuda extension to compute
        # the boxes1 and boxes2 go inside polygon_inter_union_cuda must be torch.cuda.float, not double type
        # boxes1_ = boxes1.float().contiguous().view(-1)
        # boxes2_ = boxes2.float().contiguous().view(-1)
        # inter, union = polygon_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.
        #
        # inter_nan, union_nan = inter.isnan(), union.isnan()
        # if inter_nan.any() or union_nan.any():
        #     inter2, union2 = polygon_inter_union_cuda(boxes1_,
        #                                               boxes2_)  # Careful that order should be: boxes1_, boxes2_.
        #     inter2, union2 = inter2.T, union2.T
        #     inter = torch.where(inter_nan, inter2, inter)
        #     union = torch.where(union_nan, union2, union)

        # using shapely (cpu) to compute
    inter, union = polygon_inter_union_cpu(boxes1, boxes2)
    union += eps
    iou = inter / union
    iou[torch.isnan(inter)] = 0.0
    iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
    iou[torch.isnan(iou)] = 0.0

    if GIoU or DIoU or CIoU:
        # minimum bounding box of boxes1 and boxes2
        b1_x1, b1_x2 = boxes1[:, 0::2].min(dim=1)[0], boxes1[:, 0::2].max(dim=1)[0]  # 1xn
        b1_y1, b1_y2 = boxes1[:, 1::2].min(dim=1)[0], boxes1[:, 1::2].max(dim=1)[0]  # 1xn
        b2_x1, b2_x2 = boxes2[:, 0::2].min(dim=1)[0], boxes2[:, 0::2].max(dim=1)[0]  # 1xm
        b2_y1, b2_y2 = boxes2[:, 1::2].min(dim=1)[0], boxes2[:, 1::2].max(dim=1)[0]  # 1xm
        for i in range(boxes1.shape[0]):
            cw = torch.max(b1_x2[i], b2_x2) - torch.min(b1_x1[i], b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2[i], b2_y2) - torch.min(b1_y1[i], b2_y1)  # convex height
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1[i] - b1_x2[i]) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1[i] - b1_y2[i]) ** 2) / 4  # center distance squared
                if DIoU:
                    iou[i, :] -= rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
                    w1, h1 = b1_x2[i] - b1_x1[i], b1_y2[i] - b1_y1[i] + eps
                    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou[i, :] + (1 + eps))
                    iou[i, :] -= (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                iou[i, :] -= (c_area - union[i, :]) / c_area  # GIoU
    return iou  # IoU

def order_corners(boxes):
    """
        Return sorted corners for loss.py::class Polygon_ComputeLoss::build_targets
        Sorted corners have the following restrictions:
                                y3, y4 >= y1, y2; x1 <= x2; x4 <= x3
    """

    boxes = boxes.view(-1, 4, 2)
    x = boxes[..., 0]
    y = boxes[..., 1]
    y_sorted, y_indices = torch.sort(y)  # sort y
    x_sorted = torch.zeros_like(x, dtype=x.dtype)
    for i in range(x.shape[0]):
        x_sorted[i] = x[i, y_indices[i]]
    x_sorted[:, :2], x_bottom_indices = torch.sort(x_sorted[:, :2])
    x_sorted[:, 2:4], x_top_indices = torch.sort(x_sorted[:, 2:4], descending=True)
    for i in range(y.shape[0]):
        y_sorted[i, :2] = y_sorted[i, :2][x_bottom_indices[i]]
        y_sorted[i, 2:4] = y_sorted[i, 2:4][x_top_indices[i]]
    return torch.stack((x_sorted, y_sorted), dim=2).view(-1, 8).contiguous()