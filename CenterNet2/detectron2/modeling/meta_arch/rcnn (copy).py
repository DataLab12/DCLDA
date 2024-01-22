# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from detectron2.modeling.backbone.fpn import FPN 
from detectron2.layers.info_nce import InfoNCE


__all__ = ["GeneralizedRCNN", "ProposalNetwork"]

global local_global_features


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """


    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        box_in_features: List[str],
        num_neg_features: int = 16,
        img_per_dataset: int = 1,
        start_contrastive: int = 0
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()

        self._cpu_device = torch.device("cpu")
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.input_format = input_format
        self.vis_period = vis_period
        self.num_neg_features= num_neg_features
        self.box_in_features= box_in_features
        self.img_per_dataset= img_per_dataset
        #self.bottleneck= Bottleneck(256, 4)
        self.start_contrastive = start_contrastive

        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        #print("Backbone output shape: ",backbone.output_shape())
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "box_in_features": cfg.MODEL.ROI_HEADS.IN_FEATURES,
            "num_neg_features": cfg.MODEL.BACKBONE.NEGATIVE_FEATURES,
            "img_per_dataset": cfg.SOLVER.IMS_PER_BATCH,
            "start_contrastive": cfg.SOLVER.CONTRASTIVE_START_ITER,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def separte_samples(self, sample):
        dict_list=[]
    
        var1=sample.copy()
        var2=sample.copy()
        var1['file_name']=sample['file_names'][0]
        var1['image']=sample['images'][0]
        var1.pop('file_names', None)
        var1.pop('images', None)

        var2['file_name']=sample['file_names'][1]
        var2['image']=sample['images'][1]
        var2.pop('file_names', None)
        var2.pop('images', None)

        del sample

        dict_list.append(var1)
        dict_list.append(var2)
        return dict_list

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], dataloaders, iter_num):
    # def forward(self, dataloaders, iter_num):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # print(batched_inputs[0]['image'])
        name=[]

        image_path= [x['file_names'] for x in batched_inputs]

        # print("****The list of query paths***: ",image_path)

        for paths in image_path:
            for path in paths:
                splits=path.split("/")
                name.append(splits[-2]+'/'+splits[-1])
        # print("****The list of query names***: ",name)

        if not self.training:
            return self.inference(batched_inputs, name, iter_num)

        samples=[]
        for inputs in batched_inputs:
            samples += self.separte_samples(inputs)
        
        batched_inputs= samples
        del samples
    
        # print("The length of batch input: ", batched_inputs)
        only_src_len= int(len(batched_inputs)/2)
        rand_idx= torch.randint(0, only_src_len, (int(only_src_len/2), ))
        # print(f"Random Index: {rand_idx}")
        images = self.preprocess_image(batched_inputs)

        # Filtered inputs for detection training
        filtered_batch = [batched_inputs[idx] for idx in rand_idx.tolist()]
        filtered_names= [name[idx] for idx in rand_idx.tolist()]

        images_only_src = self.preprocess_image(filtered_batch)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in filtered_batch]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        # for x in features.keys():
        #     print(f"Shape of {x} is: {features[x].shape}")

        # print("Local and Global Features: ")
        # print(f"Shape Local{local_global_features['local_features'].grad_fn} and Shape Global: {local_global_features['global_features'].grad_fn}")
        
        # print("######## Activation Features########## :", FPN.activation_features)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images_only_src, rand_idx, filtered_names, features, gt_instances)

            # print(f"Type of proposals:{type(proposals)} and Length of Proposals: {len(proposals)}")
            # print(f"Debug proposals: {proposals[0]}")
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}
        _, detector_losses, ins_loss = self.roi_heads(images_only_src, features, proposals, rand_idx, filtered_names, iter_num, gt_instances)

        ins_loss = 0.1 * ins_loss

        # print(f"*****************Total ins loss:{ins_loss} and grad: {ins_loss.grad_fn}")
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}

        if iter_num >= self.start_contrastive:
            for idx, k in enumerate(features.keys()): 

                # print("Outcome from agn_hm in Centernet: ",cls_agnostic_hmap[idx].shape)
                # print("This is the Backbone Return :",k,"shape",features[k].shape)
                
                #temp = features[k] * cls_agnostic_hmap[idx]

                # print("This is the weighted feature :",k,"shape: ",temp.shape, "Grad_fn: ",temp.grad_fn)

                
                if k == 'p8':
                    local_query_f= features[k]
                    # print(f"Local feature shape: {torch.min(local_query_f[0])}")
                    # print(f"Heatmap shape: {agn_hm_pred_per_level[1].shape}")

                if k =='p9':
                    global_query_f= features[k]
                    #print(f"Global feature shape: {global_query_f.shape}")
                    # print(f"Heatmap shape: {agn_hm_pred_per_level[3].shape}")

            s_local= []
            tp_local= []
            t_local= []
            sp_local= []
            s_global= []
            tp_global= []
            t_global= []
            sp_global= []

            batch_length = 0
            partition = 0

            with torch.no_grad():

                for idx, k in enumerate(features.keys()):
                    batch_length = features[k].shape[0]
                    partition = int(batch_length / 2)

                    #temp = features[k] * cls_agnostic_hmap[idx]

                    #output=self.bottleneck(features[k])

                    if k == 'p8':
                        neg_local_query_f= features[k]
                        
                        # print(f"Check        losses.update({"instance_contrastive_loss":ins_loss}) device: local: {neg_local_query_f.is_cuda}"

                        #print(f"The value of the partition: {partition}")

                        for i in range(0 , partition):
                            if i%2 == 0:
                                s_local.append(neg_local_query_f[i])
                            else:
                                tp_local.append(neg_local_query_f[i])
                        
                        for i in range(partition , batch_length):
                            if i%2 == 0:
                                t_local.append(neg_local_query_f[i])
                            else:
                                sp_local.append(neg_local_query_f[i])
                        

                    if k =='p9':
                        neg_global_query_f= features[k]
                        # print(f"Check device: global: {neg_global_query_f.is_cuda}")  
                        for i in range(0 , partition):
                            if i%2 == 0:
                                s_global.append(neg_global_query_f[i])
                            else:
                                tp_global.append(neg_global_query_f[i])
                        
                        for i in range(partition , batch_length):
                            if i%2 == 0:
                                t_global.append(neg_global_query_f[i])
                            else:
                                sp_global.append(neg_global_query_f[i])

                
            #All local features as negative examples in the contrastive loss function
            s_local = torch.cat(s_local, 0)
            s_local = s_local.reshape(self.num_neg_features,1024)
            tp_local = torch.cat(tp_local, 0)
            tp_local = tp_local.reshape(self.num_neg_features,1024)
            t_local = torch.cat(t_local, 0)
            t_local = t_local.reshape(self.num_neg_features,1024)
            sp_local = torch.cat(sp_local, 0)
            sp_local = sp_local.reshape(self.num_neg_features,1024)


            # All global features as negative examples in the contrastive loss function

            s_global = torch.cat(s_global, 0)
            s_global = s_global.reshape(self.num_neg_features,1024)
            tp_global = torch.cat(tp_global, 0)
            tp_global = tp_global.reshape(self.num_neg_features,1024)
            t_global = torch.cat(t_global, 0)
            t_global = t_global.reshape(self.num_neg_features,1024)
            sp_global = torch.cat(sp_global, 0)
            sp_global = sp_global.reshape(self.num_neg_features,1024)

        
            loss = InfoNCE(negative_mode='unpaired')

            cnt =0
            local_loss_S_TP = 0
            global_loss_S_TP = 0

            print("Starting local and global S TP............................")

            for i in range(0 , partition, 2):

                idxs= [x for x in range(0, self.img_per_dataset) if x != cnt]
                #print("THe value of i: ",i)
                # print("Display index: ",idxs)
                offset_idxs = [x*2+1 for x in idxs]
                neg_names = [name[x] for x in offset_idxs]


                print(f"Query Image: {name[i]}, Pos Image: {name[i+1]} and Neg Images: {neg_names}")
                local_loss_S_TP = local_loss_S_TP + loss(local_query_f[i:i+1].reshape(1,-1),local_query_f[i+1:i+2].reshape(1,-1),tp_local[idxs]) 
                + loss(local_query_f[i+1:i+2].reshape(1,-1), local_query_f[i:i+1].reshape(1,-1), s_local[idxs])

                global_loss_S_TP = global_loss_S_TP + loss(global_query_f[i:i+1].reshape(1,-1),global_query_f[i+1:i+2].reshape(1,-1),tp_global[idxs]) 
                + loss(global_query_f[i+1:i+2].reshape(1,-1), global_query_f[i:i+1].reshape(1,-1), s_global[idxs])
                
                cnt+=1
            
            cnt =0                # print(f"Query Image: {name[i]}, Pos Image: {name[i+1]} and Neg Images: {neg_names}")
            local_loss_T_SP = 0
            global_loss_T_SP = 0

            print("Starting local and global T SP............................")

            for i in range(partition , batch_length, 2):

                idxs= [x for x in range(0, self.img_per_dataset) if x != cnt]
                #print("THe value of i: ",i)
                #print("Display index: ",idxs)

                offset_idxs = [x*2+17 for x in idxs]
                neg_names = [name[x] for x in offset_idxs]

                print(f"Query Image: {name[i]}, Pos Image: {name[i+1]} and Neg Images: {neg_names}")

                local_loss_T_SP = local_loss_T_SP + loss(local_query_f[i:i+1].reshape(1,-1),local_query_f[i+1:i+2].reshape(1,-1),sp_local[idxs]) 
                + loss(local_query_f[i+1:i+2].reshape(1,-1), local_query_f[i:i+1].reshape(1,-1), t_local[idxs])

                global_loss_T_SP =  global_loss_T_SP + loss(global_query_f[i:i+1].reshape(1,-1),global_query_f[i+1:i+2].reshape(1,-1),sp_global[idxs]) 
                + loss(global_query_f[i+1:i+2].reshape(1,-1), global_query_f[i:i+1].reshape(1,-1), t_global[idxs])
                
                cnt+=1
                
            # print(f"The global_loss_T_SP is:{global_loss_T_SP} and Grad Fn: {global_loss_T_SP.grad_fn}")
            # print(f"The global_loss_T_SP is:{global_loss_T_SP} and Grad Fn: {global_loss_T_SP.grad_fn.next_functions}")

            total_local_loss = local_loss_S_TP + local_loss_T_SP

            total_local_loss = 0.01 * total_local_loss
            
            # print(f"The local contrastive loss is:{total_local_loss} and Grad Fn: {total_local_loss.grad_fn}")


            total_global_loss = global_loss_S_TP + global_loss_T_SP

            total_global_loss =  0.1 * total_global_loss

            losses.update({"local_contrastive_loss":total_local_loss})
            losses.update({"global_contrastive_loss":total_global_loss})

            # print(f"The global contrastive loss is:{total_global_loss} and Grad Fn: {total_global_loss.grad_fn}")


        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update({"instance_contrastive_loss":ins_loss})

        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        name,
        iter_num,
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, name, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, name, iter_num)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

