#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger
import shapely
import numpy as np
import shapely.geometry
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=8,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            # print(reg_output[0, :, 0][0])

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                # print(output[7])
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 9 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        # output[..., :2] = (output[..., :2] + grid) * stride
        # output[..., 2:4] = torch.exp(output[..., 2:4]) * stride

        # output[:, 0], output[:, 2], output[:, 4], output[:, 6], = output[:, 0].sigmoid()*640, output[:, 2].sigmoid()*640, output[:, 4].sigmoid()*640, output[:, 6].sigmoid()*640
        # output[:, 1], output[:, 3], output[:, 5], output[:, 7], = output[:, 1].sigmoid()*480, output[:, 3].sigmoid()*480, output[:, 5].sigmoid()*480, output[:, 7].sigmoid()*480
        # output[..., :8] = output[..., :8].sigmoid()

        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = (output[..., 2:4] + grid) * stride
        output[..., 4:6] = (output[..., 4:6] + grid) * stride
        output[..., 6:8] = (output[..., 6:8] + grid) * stride

        # output[:, 0], output[:, 2], output[:, 4], output[:, 6] = output[:, 0] * 640, output[:, 2] * 640, output[:, 4] * 640, output[:, 6] * 640
        # output[:, 1], output[:, 3], output[:, 5], output[:, 7] = output[:, 1] * 480, output[:, 3] * 480, output[:, 5] * 480, output[:, 7] * 480
        # output[..., 0:1], output[..., 2:3], output[..., 4:5], output[..., 6:7], = output[..., 0:1]*640, output[..., 2:3]*640, output[..., 4:5]*640, output[..., 6:7]*640
        # output[..., 1:2], output[..., 3:4], output[..., 5:6], output[..., 7:8], = output[..., 1:2]*480, output[..., 3:4]*480, output[..., 5:6]*480, output[..., 7:8]*480
        # print(output)
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        # outputs[:, 0], outputs[:, 2], outputs[:, 4], outputs[:, 6], = outputs[:, 0].sigmoid()*640, outputs[:, 2].sigmoid()*640, outputs[:, 4].sigmoid()*640, outputs[:, 6].sigmoid()*640
        # outputs[:, 1], outputs[:, 3], outputs[:, 5], outputs[:, 7], = outputs[:, 1].sigmoid()*480, outputs[:, 3].sigmoid()*480, outputs[:, 5].sigmoid()*480, outputs[:, 7].sigmoid()*480
        # outputs[..., :8] = outputs[..., :8].sigmoid()
        #
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = (outputs[..., 2:4] + grids) * strides
        outputs[..., 4:6] = (outputs[..., 4:6] + grids) * strides
        outputs[..., 6:8] = (outputs[..., 6:8] + grids) * strides

        # outputs[:, 0], outputs[:, 2], outputs[:, 4], outputs[:, 6], = outputs[:, 0] * 640, outputs[:, 2] * 640, outputs[:, 4] * 640, outputs[:, 6] * 640
        # outputs[:, 1], outputs[:, 3], outputs[:, 5], outputs[:, 7], = outputs[:, 1] * 480, outputs[:, 3] * 480, outputs[:, 5] * 480, outputs[:, 7] * 480

        # outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides

        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :8]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 8].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 9:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:9]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "gpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        # loss_iou = (
        #     self.iou_loss(bbox_preds.view(-1, 8)[fg_masks], reg_targets)
        # ).sum() / num_fg
        #polygon_changes
        loss_iou = torch.zeros(1, device='cuda:0')
        zero = torch.tensor(0., device='cuda:0')
        bbox_preds = bbox_preds.view(-1, 8)[fg_masks]
        # print(bbox_preds)
        # bbox_preds = torch.from_numpy(bbox_preds.view(-1, 8)[fg_masks].detach().cpu().numpy())
        # bbox_pred /= 640
        # reg_targets /= 640
        for i in range(len(bbox_preds)):
            # print(bbox_preds[i],reg_targets[i])
            bbox_preds[i][0], bbox_preds[i][2], bbox_preds[i][4], bbox_preds[i][6] = bbox_preds[i][0] / 640, bbox_preds[i][2] / 640, \
                                                                         bbox_preds[i][4] / 640, bbox_preds[i][6] / 640
            bbox_preds[i][1], bbox_preds[i][3], bbox_preds[i][5], bbox_preds[i][7] = bbox_preds[i][1] / 640, bbox_preds[i][3] / 640, \
                                                                         bbox_preds[i][5] / 640, bbox_preds[i][7] / 640
            reg_targets[i][0], reg_targets[i][2], reg_targets[i][4], reg_targets[i][6] = reg_targets[i][0] / 640, reg_targets[i][2] / 640, \
                                                                             reg_targets[i][4] / 640, reg_targets[i][6] / 640
            reg_targets[i][1], reg_targets[i][3], reg_targets[i][5], reg_targets[i][7] = reg_targets[i][1] / 640, reg_targets[i][3] / 640, \
                                                                             reg_targets[i][5] / 640, reg_targets[i][7] / 640

            loss_iou += (torch.max(zero, bbox_preds[i][1] - bbox_preds[i][5]) ** 2).mean() / 6 + (
                        torch.max(zero,  bbox_preds[i][7] - bbox_preds[i][5]) ** 2).mean() / 6 + \
                    (torch.max(zero, bbox_preds[i][1] - bbox_preds[i][3]) ** 2).mean() / 6 + (
                                torch.max(zero, bbox_preds[i][7] - bbox_preds[i][3]) ** 2).mean() / 6 + \
                    (torch.max(zero, bbox_preds[i][0] - bbox_preds[i][6]) ** 2).mean() / 6 + (
                                torch.max(zero, bbox_preds[i][2] - bbox_preds[i][4]) ** 2).mean() / 6
            # include the values of each vertice of poligon into loss function

            loss_iou += nn.SmoothL1Loss(beta=0.11)(bbox_preds[i], reg_targets[i])
        #     loss_iou += self.l1_loss(bbox_preds[i], reg_targets[i]).sum()
        # loss_iou = loss_iou/len(bbox_preds)

            # loss_iou += self.bcewithlog_loss(bbox_preds[i], reg_targets[i]).sum()

        # loss_iou = lbox/num_fg

        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        # print(obj_preds.view(-1, 1), obj_targets)
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        # print(cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets)
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        #polygon_changes
        try:
            pair_wise_ious = torch.cat([self.polygon_box_iou(gt_bboxes_per_image[i].unsqueeze(0), bboxes_preds_per_image) for i in range(num_gt)], 0)
        except:
            print(batch_idx)
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss.cuda()
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious.cuda(), gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()
        # print(pred_ious_this_matching)
        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        # gt_bboxes_per_image_l = (
        #     (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
        #     .unsqueeze(1)
        #     .repeat(1, total_num_anchors)
        # )
        # gt_bboxes_per_image_r = (
        #     (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
        #     .unsqueeze(1)
        #     .repeat(1, total_num_anchors)
        # )
        # gt_bboxes_per_image_t = (
        #     (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
        #     .unsqueeze(1)
        #     .repeat(1, total_num_anchors)
        # )
        # gt_bboxes_per_image_b = (
        #     (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
        #     .unsqueeze(1)
        #     .repeat(1, total_num_anchors)
        # )
        #polygon_changes
        gt_bboxes_per_image_l = (
            (torch.cat([min(gt_bboxes_per_image[i, 0], gt_bboxes_per_image[i, 2]).unsqueeze(0) for i in range(num_gt)], 0))
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (torch.cat([max(gt_bboxes_per_image[i, 4], gt_bboxes_per_image[i, 6]).unsqueeze(0) for i in range(num_gt)], 0))
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (torch.cat([min(gt_bboxes_per_image[i, 1], gt_bboxes_per_image[i, 7]).unsqueeze(0) for i in range(num_gt)], 0))
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (torch.cat([max(gt_bboxes_per_image[i, 3], gt_bboxes_per_image[i, 5]).unsqueeze(0) for i in range(num_gt)], 0))
                .unsqueeze(1)
                .repeat(1, total_num_anchors)
        )
        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        # gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
        #     1, total_num_anchors
        # ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        # gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
        #     1, total_num_anchors
        # ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        # gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
        #     1, total_num_anchors
        # ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        # gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
        #     1, total_num_anchors
        # ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        #polygon_changes
        polygon_center_x = (gt_bboxes_per_image[:, 0]+gt_bboxes_per_image[:, 2]+gt_bboxes_per_image[:, 4]+gt_bboxes_per_image[:, 6])/4
        polygon_center_y = (gt_bboxes_per_image[:, 1]+gt_bboxes_per_image[:, 3]+gt_bboxes_per_image[:, 5]+gt_bboxes_per_image[:, 7])/4
        gt_bboxes_per_image_l = (polygon_center_x).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (polygon_center_x).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (polygon_center_y).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (polygon_center_y).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def order_corners(self, boxes):
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

    def polygon_inter_union_cpu(self, boxes1, boxes2):
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

    def polygon_box_iou(self, gt_box, pred_boxes, GIoU=False, DIoU=False, CIoU=False, eps=1e-7, device="cpu", ordered=False):
        """
            Compute iou of polygon boxes via cpu or cuda;
            For cuda code, please refer to files in ./iou_cuda
            Returns the IoU of shape (n, m) between boxes1 and boxes2. boxes1 is nx8, boxes2 is mx8
        """
        if gt_box[0] == 'inf':
            print(66)
        ious = []
        boxes1 = gt_box
        pred_boxes = pred_boxes.cpu().numpy().tolist()
        for i in range(len(pred_boxes)):
            boxes2 = pred_boxes[i]
            boxes2 = np.array(boxes2)
            boxes2 = torch.from_numpy(boxes2).unsqueeze(0)
            # For testing this function, please use ordered=False
            # if not ordered:
            #     boxes1, boxes2 = self.order_corners(boxes1.clone().to(device)), self.order_corners(boxes2.clone().to(device))
            # else:
            boxes1, boxes2 = boxes1.clone().to(device), boxes2.clone().to(device)

            if torch.cuda.is_available() and boxes1.is_cuda:
                # using cuda extension to compute
                # the boxes1 and boxes2 go inside polygon_inter_union_cuda must be torch.cuda.float, not double type
                boxes1_ = boxes1.float().contiguous().view(-1)
                boxes2_ = boxes2.float().contiguous().view(-1)
                inter, union = self.polygon_inter_union_cuda(boxes2_, boxes1_)  # Careful that order should be: boxes2_, boxes1_.

                inter_nan, union_nan = inter.isnan(), union.isnan()
                if inter_nan.any() or union_nan.any():
                    inter2, union2 = self.polygon_inter_union_cuda(boxes1_,
                                                              boxes2_)  # Careful that order should be: boxes1_, boxes2_.
                    inter2, union2 = inter2.T, union2.T
                    inter = torch.where(inter_nan, inter2, inter)
                    union = torch.where(union_nan, union2, union)
            else:
                # using shapely (cpu) to compute
                inter, union = self.polygon_inter_union_cpu(boxes1, boxes2)
            union += eps
            iou = inter / union
            iou[torch.isnan(inter)] = 0.0
            iou[torch.logical_and(torch.isnan(inter), torch.isnan(union))] = 1.0
            iou[torch.isnan(iou)] = 0.0
            iou = iou.cpu().numpy().tolist()
            ious.append(iou)  # IoU
        ious = np.array(ious)
        ious = torch.from_numpy(ious)
        ious = ious.squeeze(1)
        ious = ious.permute(1, 0)
        return ious