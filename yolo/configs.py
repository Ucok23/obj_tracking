# YOLO options
YOLO_TYPE                   = "yolov4"  # yolov4 or yolo
YOLO_FRAMEWORK              = "tf"  # "tf" or "trt"
YOLO_V3_WEIGHTS             = "model_data/yolo.weights"
YOLO_V4_WEIGHTS             = "model_data/yolov4.weights"
YOLO_V3_TINY_WEIGHTS        = "model_data/yolo-tiny.weights"
YOLO_V4_TINY_WEIGHTS        = "model_data/yolov4-tiny.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8"  # INT8, FP16, FP32
YOLO_CUSTOM_WEIGHTS         = False  # "checkpoints/yolov3_custom" # used in evaluate_mAP.py and custom model detection, if not using leave False
                            # YOLO_CUSTOM_WEIGHTS also used with TensorRT and custom model detection
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
YOLO_MAX_BBOX_PER_SCALE     = 100
YOLO_INPUT_SIZE             = 416
if YOLO_TYPE                == "yolov4":
    YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
                               [[36,  75], [76,   55], [72,  146]],
                               [[142,110], [192, 243], [459, 401]]]
if YOLO_TYPE                == "yolo":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
# Train options
TRAIN_YOLO_TINY             = False

if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32]    
    # YOLO_ANCHORS            = [[[23, 27],  [37, 58],   [81,  82]], # this line can be uncommented for default coco weights
    YOLO_ANCHORS            = [[[10, 14],  [23, 27],   [37, 58]],
                               [[81,  82], [135, 169], [344, 319]]]
