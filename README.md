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

1. Store the video(s) in `datasets`. If you store them somewhere else adjust the `SESSION_PATH` and `VIDEO_NAME` in `speed_estimation/paths.py` accordingly

## Run

The path to the video should be given to `speed_estimation/speed_estimation.py` as argument.
If you do not give the path as argument adjust the `speed_estimation/paths.py` accordingly.
To get a visual output of the detections and tracking in the frame, set `enable_visual`.

1. `cd speed_estimation`
2. ```python speed_estimation.py --session_path_local /path/to/session --path_to_video /path/to/video.mp4``` 
or `python speed_estimation.py` (this will use the default paths configured).
The visual output will be enabled when running the following command `python speed_estimation.py --session_path_local /path/to/session --path_to_video /path/to/video.mp4 --enable_visual true`

During speed analysis the pipline will update the picture `speed_estimation/frames_detected/frame_after_detection`, which gives you visual impression of what cars are detected and tracked even if you run the pipeline on a headless system.

## Evaluation

To evaluate the speed estimates, the repository holds the module `speed_estimation/modules/evaluation`.
This module is called as soon as the video footage is analyzed. Please note that the evaluation module was build on top of the BrnoCompSpeed dataset.
If you are not using this dataset, the evaluation module will not be applicable to you in a plug and play manner.
Feel free to extend the module to fit your requirements.

## How to cite

Please consider citing our paper if you use our code in your project.

Liebe, L., Sauerwald, F., Sawicki, S., Schneider, M., Schuhmann, L., Buz, T., Boes, P., Ahmadov, A., de Melo, G. (2023). **[A Self-Calibrating End-to-End Pipeline for Real-Time Speed Estimation for
Traffic Monitoring](https://arxiv.org/abs/2309.14468)**. arXiv preprint arXiv:2309.14468

```
@misc{liebe2023farsec,
      title={FARSEC: A Reproducible Framework for Automatic Real-Time Vehicle Speed Estimation Using Traffic Cameras}, 
      author={Lucas Liebe and Franz Sauerwald and Sylwester Sawicki and Matthias Schneider and Leo Schuhmann and Tolga Buz and Paul Boes and Ahmad Ahmadov and Gerard de Melo},
      year={2023},
      eprint={2309.14468},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contributing

FARSEC is openly developed in the wild and contributions (both internal and external) are highly appreciated.
See [CONTRIBUTING.md](./CONTRIBUTING.md) on how to get started.

If you have feedback or want to propose a new feature, please [open an issue](https://github.com/porscheofficial/speed-estimation-traffic-monitoring/issues).
Thank you! 😊

## Acknowledgements

This project is a joint initiative of [Porsche AG](https://www.porsche.com), [Porsche Digital](https://www.porsche.digital/) and the [Hasso Plattner Institute](https://hpi.de) (Seminar: [AI in Practice](https://hpi.de/entrepreneurship/ai-in-practice.html)). ✨


## License

Copyright © 2023 Dr. Ing. h.c. F. Porsche AG

Dr. Ing. h.c. F. Porsche AG publishes this open source software and accompanied documentation (if any) subject to the terms of the [MIT license](./LICENSE.md). All rights not explicitly granted to you under the MIT license remain the sole and exclusive property of Dr. Ing. h.c. F. Porsche AG.

Apart from the software and documentation described above, the texts, images, graphics, animations, video and audio files as well as all other contents on this website are subject to the legal provisions of copyright law and, where applicable, other intellectual property rights. The aforementioned proprietary content of this website may not be duplicated, distributed, reproduced, made publicly accessible or otherwise used without the prior consent of the right holder.
