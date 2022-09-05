import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from mysql.connector import connect

from db import DB_USER, DB_PASSWORD, setup_table, insert_result
from deep_sort import generate_detections as gdet
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from yolo.configs import *
from yolo.helper import read_class_names
from yolo.utils import Load_Yolo_model, image_preprocess, postprocess_boxes, \
    nms, draw_bbox

logging.basicConfig(filename='tracking.log',
                    filemode='a',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d %b %Y %H:%M:%S',
                    level=logging.INFO)
video_path1 = "pedestrians.mp4"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

cx = connect(user=DB_USER, password=DB_PASSWORD)
cursor = cx.cursor()

setup_table(cursor, cx)


def create_deep_sort():
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    # initialize deep sort object
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    return encoder, tracker


def setup_stream(video_path):
    if video_path:
        vid = cv2.VideoCapture(video_path)  # detect on video
    else:
        vid = cv2.VideoCapture(0)  # detect from webcam
    return vid


def setup_images(input_size, original_frame):
    image_data = image_preprocess(np.copy(original_frame), [input_size, input_size])
    # image_data = tf.expand_dims(image_data, 0)
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    return image_data


def setup_classes(classes):
    num_class = read_class_names(classes)
    key_list = list(num_class.keys())
    val_list = list(num_class.values())
    return key_list, num_class, val_list


def extract_bboxes(bboxes, num_class, track_only):
    # extract bboxes to boxes (x, y, width, height), scores and names
    boxes, scores, names = [], [], []
    for bbox in bboxes:
        if len(track_only) != 0 and num_class[int(bbox[5])] in track_only or len(track_only) == 0:
            boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int) - bbox[0].astype(int),
                          bbox[3].astype(int) - bbox[1].astype(int)])
            scores.append(bbox[4])
            names.append(num_class[int(bbox[5])])
    # Obtain all the detections for the given frame.
    boxes = np.array(boxes)
    names = np.array(names)
    scores = np.array(scores)
    return boxes, names, scores


def object_tracking(yolo_model, video_path, output_path, input_size=416, show=False, classes=YOLO_COCO_CLASSES,
                    score_threshold=0.3, iou_threshold=0.45, track_only=[]):

    encoder, tracker = create_deep_sort()

    vid = setup_stream(video_path)

    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))  # output_path must be .mp4

    key_list, num_class, val_list = setup_classes(classes)

    # to track which id is inserted and logged
    inserted_id = []

    while True:
        _, frame = vid.read()

        try:
            original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        except:
            break

        image_data = setup_images(input_size, original_frame)

        # predict
        pred_bbox = yolo_model.predict(image_data)

        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, original_frame, input_size, score_threshold)
        bboxes = nms(bboxes, iou_threshold, method='nms')

        boxes, names, scores = extract_bboxes(bboxes, num_class, track_only)

        features = np.array(encoder(original_frame, boxes))
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(boxes, scores, names, features)]

        # Pass detections to the deepsort object and obtain the track information.
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue
            bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
            class_name = track.get_class()  # Get the class name of particular object
            tracking_id = track.track_id  # Get the ID for the particular track
            if tracking_id not in inserted_id:
                logging.info(f'class: {class_name}, id: {tracking_id}')

                data = {'id': tracking_id}
                insert_result(cursor, cx, class_name, data)
            inserted_id.append(tracking_id)

            index = key_list[val_list.index(class_name)]  # Get predicted object index by object name
            tracked_bboxes.append(bbox.tolist() + [tracking_id, index])

        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=classes, tracking=True)

        if output_path != '' : out.write(image)
        if show:
            cv2.imshow('output', image)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    yolo = Load_Yolo_model()
    object_tracking(yolo, video_path1, "detection.mp4", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1,
                    track_only=CLASSES_TO_DETECT)
