# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Part of the code is from https://github.com/tztztztztz/eql.detectron2/blob/master/projects/EQL/eql/fast_rcnn.py
import logging
import math
import json
from typing import Dict, Union

import numpy as np
from numpy import dtype
import numpy
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.utils.comm import get_world_size
from .fed_loss import load_class_freq, get_fed_loss_inds
from detectron2.modeling.backbone.fpn import FPN
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.data import build
#import FocalLoss


from centernet.modeling.roi_heads.FocalLoss import FocalLoss

__all__ = ["CustomFastRCNNOutputLayers"]

logger = logging.getLogger("detectron2")

class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(
        self, 
        cfg, 
        input_shape: ShapeSpec,
        **kwargs
    ):
        #self.num_objects= np.sum(build.class_freq) 
        super().__init__(cfg, input_shape, **kwargs)
        self.use_sigmoid_ce = cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE
        self.use_focal_loss=cfg.MODEL.ROI_BOX_HEAD.USE_FOCAL_LOSS
        self.use_softmax_difficulty= cfg.MODEL.ROI_BOX_HEAD.USE_SOFTMAX_DIFFICULTY
        self.max_diff=-10000000
        self.min_diff=100000000
        self.ub=1.0
        self.lb=0.5

        if self.use_focal_loss:
            self.focal_loss=FocalLoss(2.0)
        if self.use_sigmoid_ce:
            prior_prob = cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_score.bias, bias_value)
        
        self.cfg = cfg
        self.use_fed_loss = cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS
        if self.use_fed_loss:
            self.fed_loss_num_cat = cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT
            self.register_buffer(
                'freq_weight', 
                load_class_freq(
                    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH, 
                    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
                )
            )

    def losses(self, predictions, proposals, object_per_image, name, num_iter):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        num_classes = self.num_classes
        #print("Number of classes : ", build.class_freq)
        _log_classification_stats(scores, gt_classes)

        #print("Inside CUstom Loss :",len(proposals))
        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        #print("###### Helllllo ##########", gt_classes, "    :::::", gt_classes.shape)
        #print("###### Helllllo ##########", scores[0], "    :::::", scores.shape)
        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        elif self.use_focal_loss:
            loss_cls=  self.focal_loss(scores, gt_classes)  #self.softmax_focal_loss(scores, gt_classes, num_classes+1, num_objects= build.num_objects, alpha= build.alpha, gamma=2, size_average=True)
            print(loss_cls)
        elif self.use_softmax_difficulty:
            loss_cls=self.softmax_cross_entropy_difficulty_loss(scores, gt_classes, object_per_image, name , num_iter)
        else:
            loss_cls = self.softmax_cross_entropy_loss(scores, gt_classes, alpha= None) 

        return {
            "loss_cls": loss_cls, 
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes)
        }

    def softmax_focal_loss(self, inputs, targets, class_num, alpha=None, gamma=2, size_average=False):
        #print("############Inside Focal Loss############ : ", alpha)

        # print("Sum of objects :", num_objects)

        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
              self.alpha = Variable(alpha)

        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        # print("Probability Shape: ",P.shape)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print("Class Mask Shape: ",class_mask.shape)
        # print(class_mask[0])

        # for idx, x in  enumerate(targets):
        #     if x==20:
        #         print("background class prob at: ",idx, "is : ", P[idx])


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        alpha=alpha.view(N,1)
        #print('-----alpha------')
        # print(alpha.shape)
        #print(alpha)
        probs = (P*class_mask).sum(1).view(-1,1)
        #print('-----probs------')
        # print(probs.shape)
        #print(probs)
        log_p = probs.log()
        #print('-----log_p------')
        # print(log_p.shape)
        #print(log_p)
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        # print('-----bacth_loss------')
        # print(batch_loss.shape)
        # print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        print("Focal Loss : ", loss)
        return loss


    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0] # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1 # B x (C + 1)
        target = target[:, :C] # B x C

        weight = 1
        if self.use_fed_loss and (self.freq_weight is not None): # fedloss
            appeared = get_fed_loss_inds(
                gt_classes, 
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1 # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none') # B x C
        loss =  torch.sum(cls_loss * weight) / B  
        return loss


    def softmax_cross_entropy_loss(self, pred_class_logits, gt_classes, alpha):

        """
        change _no_instance handling
        """

        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        if self.use_fed_loss and (self.freq_weight is not None):
            C = pred_class_logits.shape[1] - 1
            appeared = get_fed_loss_inds(
                gt_classes, 
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1).float()
            appeared_mask[appeared] = 1. # C + 1
            appeared_mask[C] = 1.
            loss = F.cross_entropy(
                pred_class_logits, gt_classes, 
                weight=appeared_mask, reduction="mean") 
               
        elif alpha is not None:
            loss = F.cross_entropy(
                pred_class_logits, gt_classes, weight=alpha ,reduction="mean")   
            print("Loss : ", loss)
        else:
            loss = F.cross_entropy(
                pred_class_logits, gt_classes, reduction="mean") 

        return loss
   
    def normalize(slef, x, k, p):
        exponent = -k * torch.pow(x, p)
        output = 1 - torch.exp(exponent)
        return output

    def softmax_cross_entropy_difficulty_loss(self, pred_class_logits, gt_classes, object_per_image, name, num_iter):
        """
        change _no_instance handling
        """
        p=0.6
        k=0.05
        offset=0.4
        parts=pred_class_logits.shape[0]//256
        # print("The parts: ", parts)

        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        if self.use_fed_loss and (self.freq_weight is not None):
            C = pred_class_logits.shape[1] - 1
            appeared = get_fed_loss_inds(
                gt_classes, 
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1).float()
            appeared_mask[appeared] = 1. # C + 1
            appeared_mask[C] = 1.
            loss = F.cross_entropy(
                pred_class_logits, gt_classes, 
                weight=appeared_mask, reduction="mean")
        else:

            if num_iter>=10:

                diff_features= GeneralizedRCNN.difficulty_features
                diff_scores_per_images= torch.sum(diff_features, 1)
                num_obj_tensor = torch.tensor(object_per_image)
                device = torch.device("cuda")
                num_obj_tensor = num_obj_tensor.to(device)

                # print(f"Shape: {diff_features.shape}")
                # print("The difficulty score for Images:  ", name, "is : ",diff_scores_per_images)
                # print("The number of objects for Images:  ", name, "is : ",num_obj_tensor)

                diff_score= (diff_scores_per_images * num_obj_tensor)/100
                diff_score= self.normalize(diff_score, k, p)+offset

                # print("The difficulty score for Images:  ", name, "is : ",diff_score)

                logits_list=[]
                gt_list=[]
                for i in range(0, parts):
                    logits_list.append(pred_class_logits[i*256:(i+1)*256])
                    gt_list.append(gt_classes[i*256:(i+1)*256])

                total_loss=[]

                cnt=0
                for pred_logits, gt in zip(logits_list, gt_list):
                    loss= F.cross_entropy(pred_logits, gt, reduction="mean")
                    # print("The type of loss:  ",type(loss), "value   :",loss)
                    loss=diff_score[cnt]*loss
                    # print("The type of loss:  ",type(loss), "value   :",loss)
                    total_loss.append(loss)
                    cnt+=1

                loss=sum(total_loss)
                # print("The total loss is: ",loss)

                # print("---------------------------------------------------")
                # print("Check features grad functions: ", loss.grad_fn)

            else:
                loss = F.cross_entropy(
                     pred_class_logits, gt_classes, reduction="mean")
                
        return loss

    def inference(self, predictions, proposals):
        """
        enable use proposal boxes
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        if self.cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE:
            proposal_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * ps[:, None]) ** 0.5 \
                for s, ps in zip(scores, proposal_scores)]
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )


    def predict_probs(self, predictions, proposals):
        """
        support sigmoid
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
