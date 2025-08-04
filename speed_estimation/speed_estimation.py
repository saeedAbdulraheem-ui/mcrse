"""Speed Estimation Pipeline.

This module defines the pipeline that takes a video or stream as input and estimates the speed of
the vehicles in the video footage.
Therefore, the different modules implemented in `speed_estimation/modules` are combined to derive
the vehicles' speeds.

The  main steps are:
1. Initialize the logging that will later on capture the speed estimates

2. Initialize the object detection that detects the vehicles. Per default a YoloV4 model.

3. Analyze if the video perspective changed. If yes pause the speed estimation and recalibrate
(future work)

4. Detect the vehicles and assign unique ids to each recognized bounding box. The ids are used for
tracking the vehicles and distinguish them.

5. As long as the pipeline is not calibrated, do a scaling factor estimation.

6. As soon as the calibration is done, do the speed estimation based on the scaling factor and the
detected bounding boxes for the vehicles.
"""

import argparse
import configparser
import json
import logging
import math
import os
import time
import uuid
from collections import defaultdict
from datetime import datetime
from importlib import reload
from typing import Dict, List
import numpy as np
import cv2
from tqdm import tqdm
import hydra

from speed_estimation.get_fps import get_fps
from speed_estimation.modules.depth_map.depth_map_utils_absolute import (
    DepthModelAbsolute,
)
from speed_estimation.modules.object_detection.yolov4.object_detection import (
    ObjectDetection as ObjectDetectionYoloV4,
)
from speed_estimation.modules.scaling_factor.scaling_factor_extraction import (
    GeometricModel,
    CameraPoint,
    get_ground_truth_events,
    offline_scaling_factor_estimation_from_least_squares,
)
from speed_estimation.modules.shake_detection.shake_detection import ShakeDetection
from speed_estimation.paths import SESSION_PATH, VIDEO_NAME
from speed_estimation.utils.speed_estimation import (
    Direction,
    TrackingBox,
    MovingObject,
    calculate_car_direction,
)
from speed_estimation.modules.evaluation.evaluate import plot_absolute_error
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from ocsort import ocsort

config = configparser.ConfigParser()
config.read("speed_estimation/config.ini")

MAX_TRACKING_MATCH_THRESHOLD = config.getfloat("tracker", "max_match_distance")
# PED_CLASS_ID = config.getint("tracker", "ped_class_id")
# CYCLE_CLASS_ID = config.getint("tracker", "cycle_class_id")
CAR_CLASS_ID = config.getint("tracker", "car_class_id")
# MOTORBIKE_CLASS_ID = config.getint("tracker", "motorbike_class_id")
NUM_TRACKED_CARS = config.getint("calibration", "num_tracked_cars")
NUM_GT_EVENTS = config.getint("calibration", "num_gt_events")
AVG_FRAME_COUNT = config.getfloat("analyzer", "avg_frame_count")
SPEED_LIMIT = config.getint("analyzer", "speed_limit")
SLIDING_WINDOW_SEC = config.getint("main", "sliding_window_sec")
FPS = config.getint("main", "fps")
# CUSTOM_OBJECT_DETECTION = config.getboolean("main", "custom_object_detection")
OBJECT_DETECTION_MIN_CONFIDENCE_SCORE = config.getfloat(
    "tracker", "object_detection_min_confidence_score"
)
DEBUG_MODE = config.getboolean("main", "debug_mode")

frame_start_time = 0
# frame_end_time = 0
# frame_times = []


def run(
    path_to_video: str,
    data_dir: str,
    fps: float = 0.0,
    max_frames: int = 0,
    custom_object_detection: bool = False,
    enable_visual: bool = True,
    cfg: DictConfig = None,
) -> str:
    """Run the full speed estimation pipeline.

    This method runs the full speed estimation pipeline, including the automatic calibration using
    depth maps, object detection, and speed estimation.

    @param path_to_video:
        The path to the video that should be analyzed. The default path is defined in
        `speed_estimation/paths.py`.

    @param data_dir:
        The path to the dataset directory. The default path is defined in
        `speed_estimation/paths.py`.

    @param fps:
        The frames per second of the video that should be analyzed. If nothing is defined, the
        pipeline will derive the fps automatically.

    @param max_frames:
        The maximum frames that should be analyzed. The pipeline will stop as soon as the given
        number is reached.

    @param custom_object_detection:
        If a custom/other object detection should be used, set this parameter to true. If the
        parameter is set to true, the pipeline expects the detection model in
        `speed_estimation/modules/custom_object_detection`. The default detection is a YoloV4 model.

    @param enable_visual:
        Enable a visual output of the detected and tracked cars. If the flag is disabled the frame
        speed_estimation/frames_detected/frame_after_detection.jpg will be updated.

    @return:
        The string to the log file containing the speed estimates.
    """
    reload(logging)

    run_id = uuid.uuid4().hex[:10]
    print(f"Run No.: {run_id}")

    # Initialize logging
    now_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"logs/{now_str}_run_{run_id}.log"
    os.makedirs(os.path.dirname(log_name), exist_ok=True)

    logging.basicConfig(
        filename=f"logs/{now_str}_run_{run_id}.log", level=logging.DEBUG
    )
    logging.info("Run No.: %s, Video: %s", str(run_id), str(data_dir))

    start = time.time()

    # Initialize Object Detection
    if custom_object_detection:
        # Insert your custom object detection here
        object_detection = ObjectDetectionYoloV4()
    else:
        object_detection = ObjectDetectionYoloV4()

    input_video = cv2.VideoCapture(path_to_video)
    if not input_video.isOpened():
        logging.error("Could not open video file: %s", path_to_video)
        raise FileNotFoundError(f"Could not open video file: {path_to_video}")

    fps = get_fps(path_to_video) if fps == 0 else fps

    sliding_window = SLIDING_WINDOW_SEC * fps

    # Initialize running variables
    frame_count = 0
    track_id = 0
    tracking_objects: Dict[int, TrackingBox] = {}
    tracked_object: Dict[int, MovingObject] = {}
    tracker_ocsort = ocsort.OCSort(
        det_thresh=MAX_TRACKING_MATCH_THRESHOLD, max_age=15, min_hits=2
    )
    # tracked_boxes: Dict[int, List[TrackingBox]] = defaultdict(list)
    # tracked_boxes: Dict[int, List[TrackingBox]] = defaultdict(list)
    depth_model = DepthModelAbsolute(data_dir, path_to_video)
    geo_model = GeometricModel(depth_model)
    # is_calibrated = False
    text_color = (255, 255, 255)

    # for shake_detection
    shake_detection = ShakeDetection()

    while True:
        ############################
        # load frame, shake detection and object detection
        ############################
        ret, frame = input_video.read()
        frame_start_time = time.time()

        if frame_count == 0:
            # set normalization axes once at beginning
            center_x = int(frame.shape[1] / 2)
            center_y = int(frame.shape[0] / 2)
            geo_model.set_normalization_axes(center_x, center_y)

        # TODO(SAID): run for only X frames, testing
        if frame_count == 200:
            print("Stopping after 200 frames for testing purposes.")
            break

        if not ret:
            print("No more frames to read.")
            break

        # for shake_detection
        if shake_detection.is_hard_move(frame):
            logging.info(
                "Run No.: %s, Video: %s, Hard Move Detected Frame: %d",
                str(run_id),
                str(data_dir),
                frame_count,
            )

        ############################
        # Detect cars + ped and cyclists on frame
        ############################
        car_detection_start_time = time.time()
        # print(f"Detecting objects in frame {frame_count}...")
        (class_ids, scores, boxes) = object_detection.detect(frame)

        boxes = [
            [boxes[i], class_ids[i]]
            for i, class_id in enumerate(class_ids)
            if (
                class_id
                == CAR_CLASS_ID
                # or class_id == PED_CLASS_ID
                # or class_id == CYCLE_CLASS_ID
                # or class_id == MOTORBIKE_CLASS_ID
            )
            and scores[i] >= OBJECT_DETECTION_MIN_CONFIDENCE_SCORE
        ]

        # collect tracking boxes
        dets = []
        for box, score in zip(boxes, scores):
            (x_coord, y_coord, width, height) = box[0].astype(int)
            center_x = int((x_coord + x_coord + width) / 2)
            center_y = int((y_coord + y_coord + height) / 2)
            class_id = int(box[1])
            dets.append([x_coord, y_coord, x_coord + width, y_coord + height, score])

        car_detection_end_time = time.time()
        ############################
        # assign tracking box IDs
        ############################
        tracking_start_time = time.time()
        # Initialize ocsort tracker if not already done

        # Update tracker with current detections
        if len(dets) == 0:
            dets = np.empty((0, 5))
        tracks = tracker_ocsort.update(
            np.array(dets),
            img_info=frame.shape[:2],
            img_size=(frame.shape[0], frame.shape[1]),
        )

        # clear and update tracking_objects with ocsort output
        tracking_objects.clear()
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5].astype(int)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            width = x2 - x1
            height = y2 - y1
            # find class_id for this track (fallback to CAR_CLASS_ID)
            class_id = CAR_CLASS_ID
            for box in boxes:
                bx, by, bw, bh = box[0].astype(int)
            if (
                abs(bx - x1) < 3
                and abs(by - y1) < 3
                and abs(bw - width) < 3
                and abs(bh - height) < 3
            ):
                class_id = int(box[1])
                break
            tracking_objects[track_id] = TrackingBox(
                center_x, center_y, x1, y1, width, height, frame_count, class_id
            )
            if DEBUG_MODE:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x1 + width, y1 + height),
                    (255, 0, 0),
                    2,
                )
        ############################
        # scaling factor estimation
        ############################
        # if not is_calibrated:
        #     if len(tracked_boxes) >= NUM_TRACKED_CARS:
        #         # more than x cars were tracked
        #         ground_truth_events = get_ground_truth_events(tracked_boxes)
        #         if len(ground_truth_events) >= NUM_GT_EVENTS:
        #             # could extract more than x ground truth events
        #             geo_model.scale_factor = 2 * (
        #                 offline_scaling_factor_estimation_from_least_squares(
        #                     frame, geo_model, ground_truth_events
        #                 )
        #             )
        #             logging.info(
        #                 "Is calibrated: scale_factor: %d", geo_model.scale_factor
        #             )
        #             print(
        #                 f"Is calibrated: scale_factor: {geo_model.scale_factor}",
        #                 flush=True,
        #             )
        #             is_calibrated = True
        #             object_detection = ObjectDetectionYoloV4()
        #             # todo:make this more robust, resets video when calibration is done
        #             input_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #             frame_count = 0
        #             continue

        #     for object_id, tracking_box in tracking_objects.items():
        #         tracked_boxes[object_id].append(tracking_box)
        # else:
        ############################
        # track objects and update tracked_object
        ############################
        for object_id, tracking_box in tracking_objects.items():
            if object_id in tracked_object:
                tracked_object[object_id].tracked_boxes.append(tracking_box)
                tracked_object[object_id].frames_seen += 1
                tracked_object[object_id].frame_end += 1
            else:
                tracked_object[object_id] = MovingObject(
                    [tracking_box],
                    1,
                    frame_count,
                    frame_count,
                    Direction.UNDEFINED,
                    0.0,
                )
        tracking_end_time = time.time()
        ############################
        # speed estimation
        ############################
        speed_estimation_start_time = time.time()
        if frame_count >= fps:  # and frame_count % sliding_window == 0:
            # every x seconds
            car_count_towards = 0
            car_count_away = 0
            total_speed_towards = 0
            total_speed_away = 0
            total_speed_meta_appr_towards = 0.0
            total_speed_meta_appr_away = 0.0
            ids_to_drop = []

            for car_id, car in tracked_object.items():
                if car.frame_end >= frame_count - sliding_window:
                    if 9 < car.frames_seen < 750:
                        car.direction = calculate_car_direction(car)
                        car_first_box = car.tracked_boxes[0]
                        car_last_box = car.tracked_boxes[-1]
                        meters_moved = geo_model.get_scaled_distance_from_camera_points(
                            frame,
                            CameraPoint(
                                car_first_box.frame_count,
                                car_first_box.center_x,
                                car_first_box.center_y,
                            ),
                            CameraPoint(
                                car_last_box.frame_count,
                                car_last_box.center_x,
                                car_last_box.center_y,
                            ),
                        )
                        if meters_moved <= 6:
                            continue

                        if car.direction == Direction.TOWARDS:
                            car_count_towards += 1
                            total_speed_towards += (meters_moved) / (
                                car.frames_seen / fps
                            )
                            if total_speed_towards > SPEED_LIMIT:
                                total_speed_towards = SPEED_LIMIT
                            total_speed_meta_appr_towards += (
                                AVG_FRAME_COUNT / int(car.frames_seen)
                            ) * SPEED_LIMIT
                        else:
                            car_count_away += 1
                            total_speed_away += (meters_moved) / (car.frames_seen / fps)
                            if total_speed_away > SPEED_LIMIT:
                                total_speed_away = SPEED_LIMIT
                            total_speed_meta_appr_away += (
                                AVG_FRAME_COUNT / int(car.frames_seen)
                            ) * SPEED_LIMIT
                        # Write car ID and estimated speed on the car
                        speed_kmh = round(
                            (meters_moved / (car.frames_seen / fps)) * 3.6, 2
                        )
                        # print(f"obj ID {car_id}, speed {speed_kmh} km/h")
                        car.speed = speed_kmh
                        # TODO(SAID): remove debug
                        if DEBUG_MODE:
                            cv2.putText(
                                frame,
                                f"ID: {car_id} V: {speed_kmh}",
                                (
                                    car.tracked_boxes[-1].center_x,
                                    car.tracked_boxes[-1].center_y,
                                ),
                                0,
                                1,
                                (255, 0, 255),  # red color in BGR
                                1,
                            )
                else:
                    # car is too old, drop from tracked_object
                    ids_to_drop.append(car_id)

            for car_id in ids_to_drop:
                del tracked_object[car_id]

            if car_count_towards > 0:
                avg_speed = round((total_speed_towards / car_count_towards) * 3.6, 2)
                logging.info(
                    json.dumps(dict(frameId=frame_count, avgSpeedTowards=avg_speed))
                )

            if car_count_away > 0:
                avg_speed = round((total_speed_away / car_count_away) * 3.6, 2)
                cv2.putText(
                    frame,
                    f"Avg Speed: {avg_speed} km/h",
                    (7, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    text_color,
                    2,
                )
                # print(f"Average speed away: {avg_speed} km/h")
                # print(
                #     f"Average META speed away: "
                #     f"{(total_speed_meta_appr_away / car_count_away)} km/h"
                # )
                logging.info(
                    json.dumps(dict(frameId=frame_count, avgSpeedAway=avg_speed))
                )
        speed_estimation_end_time = time.time()
        ############################
        # output text on video stream
        ############################
        # TODO(SAID): remove debug
        if frame_count % 10 == 0:
            print(f"submodule times for frame {frame_count}:")
            print(
                f"  - car detection: {car_detection_end_time - car_detection_start_time:.2f}s"
            )
            print(f"  - tracking: {tracking_end_time - tracking_start_time:.2f}s")
            print(
                f"  - speed estimation: {speed_estimation_end_time - speed_estimation_start_time:.2f}s"
            )
            print(f"  - total: {time.time() - frame_start_time:.2f}s")

        timestamp = frame_count / fps
        if DEBUG_MODE:
            cv2.putText(
                frame,
                f"Timestamp: {timestamp :.2f} s",
                (7, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps}",
                (7, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                2,
            )

            cv2.imwrite(
                f"speed_estimation/frames_detected/frame_after_detection_{frame_count}.jpg",
                frame,
            )

        if frame_count % 50 == 0:
            print(
                f"Frame no. {frame_count} time since start: {(time.time() - start):.2f}s"
            )
        frame_count += 1
        # if max_frames != 0 and frame_count >= max_frames:
        #     if not is_calibrated:
        #         log_name = ""
        #     print("Max frames reached, stopping speed estimation.")
        #     break

    input_video.release()
    cv2.destroyAllWindows()
    logging.shutdown()
    return log_name


@hydra.main(
    config_path="modules/depth_map/FlashDepth/configs/flashdepth",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--session_path_local",
        nargs="?",
        help="Path to session (e.g., the directory where the video is stored)",
        default=SESSION_PATH,
    )
    parser.add_argument(
        "-p",
        "--path_to_video",
        nargs="?",
        help="Path to video",
        default=os.path.join(SESSION_PATH, VIDEO_NAME),
    )
    parser.add_argument(
        "-v",
        "--enable_visual",
        nargs="?",
        help="Enable visual output.",
        default=False,
    )
    args = parser.parse_args()
    """Run the speed estimation pipeline."""
    max_frames = FPS * 60 * 20  # fps * sec * min
    hydra_cfg = HydraConfig.get()
    cfg.config_dir = [
        path["path"]
        for path in hydra_cfg.runtime.config_sources
        if path["schema"] == "file"
    ][0]
    print(args.session_path_local)
    print(args.path_to_video)

    log_name = run(
        args.path_to_video,
        args.session_path_local,
        FPS,
        max_frames=max_frames,
        custom_object_detection=False,
        enable_visual=args.enable_visual,
        cfg=cfg,
    )

    if log_name is None:
        print("Calibration did not finish, skip evaluation.")
    else:
        # Evaluation
        plot_absolute_error([log_name], "logs/", "speed_estimation/gt_logs")
        print("Put your evaluation here.")


if __name__ == "__main__":
    # Run pipeline
    main()
