from easydict import EasyDict as edict
import os

__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.RDT_Reader                        = edict()

# Set the class name
__C.RDT_Reader.CLASSES                = "D:/source/repos/object_detection_mobile_v2/dataset/classes/rdt.names"
__C.RDT_Reader.ORIGINAL_WEIGHT        = ""
__C.RDT_Reader.DEMO_WEIGHT            = ""

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.BATCH_SIZE            = 2
__C.TRAIN.LABEL_PATH            = "D:/source/repos/object_detection_mobile_v2/dataset/labels_seg_tr"
__C.TRAIN.IMAGE_PATH            = "D:/source/repos/object_detection_mobile_v2/dataset/images_seg_tr"
__C.TRAIN.INPUT_SIZE            = (256,256)
__C.TRAIN.MOVING_AVE_DECAY      = 0.9995
__C.TRAIN.DATA_AUG              = True
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-10
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 5
__C.TRAIN.SECOND_STAGE_EPOCHS   = 20
__C.TRAIN.INITIAL_WEIGHT        = "D:/source/repos/object_detection_mobile_v2/model_save.hdf5"
__C.TRAIN.UPSAMPLE              = 4
__C.TRAIN.QUANT_DELAY           = 2
__C.TRAIN.OUTDATA           = "D:/source/repos/object_detection_mobile_v2/output_check/"
__C.TRAIN.PREDICTION_SCALE   = [0.5]
__C.TRAIN.NUMBER_CLASSES   = 3
__C.TRAIN.ANCHOR_ASPECTRATIO   = [[[20,10],[10,20],[30,30],[25,20],[20,25]]]
__C.TRAIN.IOU_THRESH   = 0.5
__C.TRAIN.NUMBER_BLOCKS            = 15






# TEST options
__C.TEST                        = edict()
__C.TEST.LABEL_PATH            = "D:/source/repos/object_detection_mobile_v2/dataset/labels_seg_tr"
__C.TEST.IMAGE_PATH            = "D:/source/repos/object_detection_mobile_v2/dataset/images_seg_te"
__C.TEST.BATCH_SIZE             = 1
__C.TEST.INPUT_SIZE             = (256,256)
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "D:/source/repos/object_detection_mobile_v2/dataset/detection"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.WEIGHT_FILE            = "D:/source/repos/object_detection_mobile_v2/model_save.hdf5"
__C.TEST.QUANTIZED_WEIGHT_FILE  = "D:/source/repos/object_detection_mobile_v2/eval_model/model_save_server.hdf5"
__C.TEST.SHOW_LABEL             = True
__C.TEST.UPSAMPLE               = 1
__C.TEST.EVAL_MODEL_PATH        = "D:/source/repos/object_detection_mobile_v2/eval_model"




