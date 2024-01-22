from detectron2.structures import BoxMode
import cv2
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import time
import datetime
import json
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from fvcore.common.timer import Timer
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch

from detectron2.evaluation import (
    COCOEvaluator,
    LVISEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron2.modeling.test_time_augmentation import GeneralizedRCNNWithTTA
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.build import build_detection_train_loader

from centernet.config import add_centernet_config
from centernet.data.custom_build_augmentation import build_custom_augmentation

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        mapper = None if cfg.INPUT.TEST_INPUT_TYPE == 'default' else \
            DatasetMapper(
                cfg, False, augmentations=build_custom_augmentation(cfg, False))
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        output_folder = os.path.join(
            cfg.OUTPUT_DIR, "inference_{}".format(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "lvis":
            evaluator = LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == 'coco':
            evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        else:
            assert 0, evaluator_type
            
        results[dataset_name] = inference_on_dataset(
            model, data_loader, evaluator)
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(
                dataset_name))
            print_csv_format(results[dataset_name])
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(
            cfg.MODEL.WEIGHTS, resume=resume,
            ).get("iteration", -1) + 1
    )
    if cfg.SOLVER.RESET_ITER:
        logger.info('Reset loaded iteration. Start training from iteration 0.')
        start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER if cfg.SOLVER.TRAIN_ITER < 0 else cfg.SOLVER.TRAIN_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )


    mapper = DatasetMapper(cfg, True) if cfg.INPUT.CUSTOM_AUG == '' else \
        DatasetMapper(cfg, True, augmentations=build_custom_augmentation(cfg, True))
    if cfg.DATALOADER.SAMPLER_TRAIN in ['TrainingSampler', 'RepeatFactorTrainingSampler']:
        data_loaders=[]
        dataset_list= cfg.DATASETS.TRAIN_LIST
        for dataset in dataset_list:
            print(dataset)
            cfg.DATASETS.TRAIN = dataset
            data_loaders.append(build_detection_train_loader(cfg, mapper=mapper))
    else:
        from centernet.data.custom_dataset_dataloader import  build_custom_train_loader
        data_loaders = build_custom_train_loader(cfg, mapper=mapper)


    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        step_timer = Timer()
        data_timer = Timer()
        start_time = time.perf_counter()
        for data1, data2, iteration in zip(data_loaders[0], data_loaders[1], range(start_iter, max_iter)):
            # print(f"##########ITERATION: {iteration}  ###########")
            data_time = data_timer.seconds()
            storage.put_scalars(data_time=data_time)
            step_timer.reset()
            iteration = iteration + 1
            storage.step()
            data= data1 +data2

            # print(data)
            # print("Device of the data: ", data.get_device())

            loss_dict = model(data, iteration)

            losses = sum(
                loss for k, loss in loss_dict.items())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() \
                for k, v in comm.reduce_dict(loss_dict).items()}
            # print(f"Toal Loss dict:{loss_dict_reduced}")
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            # print(f"Toal Loss reduced:{losses_reduced}")

            if comm.is_main_process():
                storage.put_scalars(
                    total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            storage.put_scalar(
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

            step_time = step_timer.seconds()
            storage.put_scalars(time=step_time)
            data_timer.reset()
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                comm.synchronize()

            if iteration - start_iter > 5 and \
                (iteration % 50 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

        total_time = time.perf_counter() - start_time
        logger.info(
            "Total training time: {}".format(
                str(datetime.timedelta(seconds=int(total_time)))))

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"

    add_centernet_config(cfg)
    cfg.merge_from_list(args.opts)
    cfg.merge_from_file("/home/ubq3/Experiments/aerialAdaptation/CenterNet2/projects/CenterNet2/configs/CenterNet2-F_R50_1x.yaml")
    cfg.DATASETS.TRAIN_DIR_LIST = [("train_dior_S","train_dior_tp"),("train_dota_T","train_dota_sp")]
    cfg.DATASETS.TRAIN_ROOT_LIST = ["/home/ubq3/DomainAdaptationDataset/DIOR", "/home/ubq3/DomainAdaptationDataset/DOTA2.0"]
    cfg.DATASETS.TRAIN_LIST = ["dior_S_TP_train","dota_S_TP_train"]
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 10
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.RESET_ITER= True
    #cfg.DEBUG=True
    cfg.SOLVER.IMS_PER_BATCH= 8
    cfg.MODEL.BACKBONE.NEGATIVE_FEATURES = cfg.SOLVER.IMS_PER_BATCH
    # cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH / len(cfg.DATASETS.TRAIN_LIST)
    cfg.SOLVER.BASE_LR = 0.01 * 1 / 10  # pick a good LR
    cfg.SOLVER.MAX_ITER = 86250  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.CONTRASTIVE_START_ITER = 0 #cfg.SOLVER.MAX_ITER + 10 #int(cfg.SOLVER.MAX_ITER / 2)
    cfg.SOLVER.STEPS = [25000, 70000]        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  #For feature extraction set BATCH_SIZE_PER_IMAGE= 128, Otherwise set value = 256 or 512
                                                    # faster, and good enough for this toy dataset (default: 512)

    cfg.MODEL.CENTERNET.POST_NMS_TOPK_TARGETADAP= 64
    cfg.MODEL.TARGET_INS_WEIGHT = 0.01

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  #(see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS =  [0.6]  #For feature extraction set thershold value= 0.5, Otherwise set value = 0.6 or 0.7
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.40 #For feature extraction set POSITIVE_FRACTION= 0.1, Otherwise set value = 0.25 or 0.30
    cfg.OUTPUT_DIR=cfg.OUTPUT_DIR+"DomainAdaptation/debug" #ResNet101_MLP_layer_constrastive_learning
    cfg.H5_PATH= cfg.H5_PATH+"dior_val_features.h5"
    cfg.SAVE_H5= False
    cfg.H5_SAVE_ITERATION= 0

    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.


    # if '/auto' in cfg.OUTPUT_DIR:
    #     file_name = os.path.basename(args.config_file)[:-5]
    #     cfg.OUTPUT_DIR = cfg.OUTPUT_DIR.replace('/auto', '/{}'.format(file_name))
    #     logger.info('OUTPUT_DIR: {}'.format(cfg.OUTPUT_DIR))

    
    #cfg.freeze()
    default_setup(cfg, args)
    return cfg

def get_dicts(img_dirs, rootpath, skip = False):
    json_file = os.path.join(rootpath, "train_anno.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    cnt = 0
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename1 = os.path.join(rootpath, img_dirs[0], v["filename"])
        filename2 = os.path.join(rootpath, img_dirs[1], v["filename"])
        height, width = cv2.imread(filename1).shape[:2]
        
        record["file_names"] = [filename1, filename2]
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        #print(filename)
        annos = v["shape_attributes"]
        objs = []

        if annos != None and skip != True:
            for i,bbox in enumerate(annos):
                #print(anno_list[0][str(0)][0][0])

                xmin= bbox[str(i)][0][0]
                ymax= bbox[str(i)][0][1]
                width= bbox[str(i)][1][0]
                height= bbox[str(i)][1][1]

                obj = {
                    "bbox": [xmin, ymax, int(width), int(height)],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    #"segmentation": [poly],
                    "category_id": int(bbox['category_id']),
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
        cnt+=1
        if cnt == 200:
            return dataset_dicts
        
    return dataset_dicts

def create_h5_path(path):
    path= path.rsplit('/', 1)[0]

    if not os.path.exists(path):
        os.makedirs(path)

def registrar_dataset_dior(cfg):

        DatasetCatalog.register(cfg.DATASETS.TRAIN_LIST[0], lambda dataset=cfg.DATASETS.TRAIN_LIST[0]: get_dicts(cfg.DATASETS.TRAIN_DIR_LIST[0], cfg.DATASETS.TRAIN_ROOT_LIST[0], skip= False))
        
        MetadataCatalog.get(cfg.DATASETS.TRAIN_LIST[0]).set(thing_classes=['bridge', 'vehicle', 'harbor', 'storagetank', 
        'baseballfield','groundtrackfield', 'basketballcourt', 'tenniscourt', 'stadium', 'airport'])

def registrar_dataset_dota(cfg):

        DatasetCatalog.register(cfg.DATASETS.TRAIN_LIST[1], lambda dataset=cfg.DATASETS.TRAIN_LIST[1]: get_dicts(cfg.DATASETS.TRAIN_DIR_LIST[1], cfg.DATASETS.TRAIN_ROOT_LIST[1], skip= True))
        
        MetadataCatalog.get(cfg.DATASETS.TRAIN_LIST[0]).set(thing_classes=['bridge', 'vehicle', 'harbor', 'storagetank', 
        'baseballfield','groundtrackfield', 'basketballcourt', 'tenniscourt', 'stadium', 'airport'])

def main(args):

    

    cfg = setup(args)
    #print(cfg)
    registrar_dataset_dior(cfg)

    registrar_dataset_dota(cfg)

    model = build_model(cfg)

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        if cfg.TEST.AUG.ENABLED:
            logger.info("Running inference with test-time augmentation ...")
            model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)

        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            find_unused_parameters=True
        )

    if cfg.SAVE_H5:
        create_h5_path(cfg.H5_PATH)

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument('--manual_device', default='')
    args = args.parse_args()
    if args.manual_device != '':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.manual_device
    args.dist_url = 'tcp://127.0.0.1:{}'.format(
        torch.randint(11111, 60000, (1,))[0].item())
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
        )
