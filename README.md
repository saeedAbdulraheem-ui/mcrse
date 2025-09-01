# MC-RMSE Monocular Camera Real-time Metric Speed Estimation

### This repository provides an easy way of estimating the speed of traffic using video footage

## acknowledgement: thanks to the authors of FARSEC for providing the base framework: https://arxiv.org/html/2309.14468

## Structure
The project is split into multiple modules, each handling a part of the total pipeline, the flowchart of the full pipeline is shown below.

<img width="1153" height="214" alt="image" src="https://github.com/user-attachments/assets/27111f77-aaa0-49dd-a0cb-880dec86a034" />


The different modules of this project can be found inside the folder *speed_estimation/modules*
The relevant modules are:

| Module Name                     | Folder                   | Description                                                                                                                                                           |
|---------------------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Depth map                       | modules/depth_map        | Generates a depth map for a provided frame, either utilizes UniDepthV2 https://github.com/lpiccinelli-eth/UniDepth for metric depth estimation                        |
| (requires no calibration). Or | uses FlashDepth for temporally-coherent depth estimation and would provide better results if calibration parameters to be set in modules/scaling_factor_estimation. For Flashdepth you must setup  |
| submodule, to do so follow the instructions found in https://github.com/Eyeline-Labs/FlashDepth/tree/main                                                                                                                          |
| Evaluation                      | modules/evaluation       | Compares videos with the provided ground truth on the kitti-raw Dataset.                                                                                              |
| Car Tracking                    | modules/object_detection | Detecting cars in a video frame by with a YOLOv4 model. Newer models may be used as well with minor modifications in `modules/speed_estimation.py`                    |
| Calibration                     | modules/scaling_factor   | Automatically calibrates the pipeline at start and derives a scaling factor.                                                                                          |
| Shake Detection                 | modules/shake_detection  | Detects if the camera perspective changed. If so a recalibration is required.                                                                                         |
| Stream-Conversion & Downsampler | modules/streaming        | Reads a stream, caps it to 30 FPS and provides the frames.                                                                                                            |

## Setup

to run the code, a docker image setup is preffered, however, the instructions can also be run to setup the environment locally

 
### Docker Setup

#### With CUDA
**Note: We used this setup on an Nvidia GeForce RTX 4050 with Cuda 11.4. It can happen that this setup needs some modifications to fit your individual setup.**
-1. for Flashdepth usage, setup the Flashdepth model first by following the link: https://github.com/Eyeline-Labs/FlashDepth/tree/main  
0. (Have `docker` installed)
2. Go to `docker/cuda` directory in a terminal.
3. Run `docker build .` Assign a tag, if you like.
4. Run the docker container with the following command: ./run_docker.sh, however, modify the paths in the script to match the local ones.
5. to run the speed estimation pipeline, use the command ./run_estimation.sh, the same as the above, modify the paths first

**Note: This repository has a default configuration (`speed_estimation/config.ini`) that can be adjusted if necessary (see Section [Configuration](#configuration)).**


## Configuration
This project comes with a default configuration, which can be adjusted. To do so, have a closer look into `speed_estimation/config.ini`

| Name                               | Description                                                                                                                                                                 | Values |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|
| fps                                | Default FPS to use, if they can't be detected from the provided video.                                                                                                      | integer |
| custom_object_detection            | Wether to use your custom trained model or pretrained yolov4 (default).                                                                                                     | boolean |
| sliding_window_sec                 | Seconds to use for the sliding window, in which the speed es estimated.                                                                                                     | integer |
| num_tracked_cars                   | Number of cars the pipeline should use to calibrate itself.                                                                                                                 | integer |
| num_gt_events                      | Number of ground truth events the pipeline should use to calibrate itself.                                                                                                  | integer |
| ped_class_id                       | The class the detection model uses to identify a person.                                                                                                                    | integer |
| car_class_id                       | The class the detection model uses to identify a car.                                                                                                                       | integer |
| cycle_class_id                     | The class the detection model uses to identify a cycler.                                                                                                                    | integer |
| motorbike_class_id                 | The class the detection model uses to identify a motorbike.                                                                                                                 | integer |
| max_match_distance                 | Maximum distance for that bounding boxes are accepted (from the closest bounding box).                                                                                      | integer |
| object_detection_min_confidence_score | The minimum allowed score with which the model should recognize a vehicle.                                                                                                | float  |
| speed_limit                        | Speed limit on the road segment shown in the video (in km/h).                                                                                                               | integer |  
| avg_frame_count                    | Output of meta statistics approach gets written here. Average frames a standard car was taking to drive through the CCTV segment (average tracked over a longer time frame). | float  |


Additionally, the `speed_estimation/paths.py` can be adjusted.

| Name                         | Description                                                    | Values |
|------------------------------|----------------------------------------------------------------|--------|
| PATH_TO_HAAR_FILE            | Path to the HAAR file required for the object detection model. | string |
| YOLOV4_WEIGHTS               | Path to the model weights.                                     | string |
| YOLOV4_CLASSES               | Path to the different classes the model can detect.            | string |
| YOLOV4_CONFIG                | Path to config file of the model.                              | string |
| SESSION_PATH                 | Directory where the video that should be analyzed is stored.   | string |
| VIDEO_NAME                   | The name of the video that should be analyzed.                 | string |
| SPEED_ESTIMATION_CONFIG_FILE | Location of the `config.ini` file described above.             | string |

## Dataset

we use the kitti raw dataset for evaluation, found in https://www.cvlibs.net/datasets/kitti/raw_data.php, for setting up the ground truth, and generating a stitched video to be used use the repo https://github.com/saeedAbdulraheem-ui/mcrse_preprocess_kitti

**The pipline does also work with other videos and datasets, what means that you do not necessarily use the Brno CompSpeed dataset, but your own ones.**
Store the video(s) in `datasets`. If you store them somewhere else adjust the `SESSION_PATH` and `VIDEO_NAME` in `speed_estimation/paths.py` accordingly
