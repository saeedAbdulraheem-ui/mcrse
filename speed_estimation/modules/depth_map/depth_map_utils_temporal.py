import os
from typing import Tuple, List

import cv2
import imutils
from speed_estimation.modules.depth_map.FlashDepth.flashdepth import hydra
import torch
from numpy.typing import NDArray
from .FlashDepth.utils.init_setup import setup_model, dist_init
import numpy as np

from .FlashDepth.infer import process_frame
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


class DepthModelTemporal:
    """This class holds the depth map generation."""

    def __init__(self, data_dir: str, path_to_video: str, cfg: DictConfig) -> None:
        """Create an instance of DepthModel.

        @param data_dir:
            The directory where the generated depth maps should be stored and loaded from.

        @param path_to_video:
            The path to the video that should be analyzed.
        """
        self.path_to_video = path_to_video
        self.data_dir = data_dir
        self.memo: dict[int, NDArray] = {}
        process_dict = dist_init()
        hydra_cfg = HydraConfig.get()
        cfg.config_dir = [path["path"] for path in hydra_cfg.runtime.config_sources if path["schema"] == "file"][0] 
        self.config = cfg
        with torch.no_grad():
            model, _, _, train_step = setup_model(self.config, process_dict)
            model.eval()
            self.model = torch.compile(model)

    
    def predict_depth(self, frame_id: int, frame) -> NDArray:
        """Predict the depth map for the defined frame.

        Predict the depth map that estimates the depth in the frame. The estimated depth relative,
        what means that it will not be in meters or miles, but in the range from 0 to 1 for each
        pixel in the frame. The closer to 1 the further away the pixel.

        @param frame_id:
            The count of the frame.

        @return:
            Returns a NDArray in the dimension of the original frame. The array hold the relative
            distances for each pixel.
        """
        if frame_id in self.memo:
            # Depth map already predicted
            return self.memo[frame_id]

        # if len(self.memo) > 10:
        #     # Take the mean over all depth maps
        #     depth_maps: List[NDArray] = [
        #         np.array(self.memo[frame]) for frame in self.memo
        #     ]
        #     # Ensure all depth maps have the same shape before stacking
        #     min_shape = np.min([dm.shape for dm in depth_maps], axis=0)
        #     depth_maps = [dm[:min_shape[0], :min_shape[1]] for dm in depth_maps]
        #     return np.mean(np.stack(depth_maps, axis=0), axis=0)

        self.memo[frame_id] = self.load_depth_map(
            self.data_dir, self.path_to_video, max_depth=1, frame_idx=frame_id, frame=frame
        )

        # predict depth here
        return self.memo[frame_id]


    def load_depth_map(self, current_folder: str, path_to_video: str, max_depth: int = 1, frame_idx: int = 0, frame = None) -> NDArray:
        """Load the depth map.

        This function loads the depth map for one specific frame. The output size of the depth map is
        the same as the one of the original frame

        @param current_folder:
            The folder the depth map generation should work on. Usually the folder where the input
            video is stored.

        @param path_to_video:
            The path to the video that should be analyzed.

        @param max_depth:
            Configures the maximum depth allowed for the depth map estimation. Since the depth map is
            relative the default value is 1.

        @param frame_idx:
            The index of the frame the depth map estimation should take as input.

        @return:
            Returns the depth map of the specified frame.
        """
        # print("Depth map generation.")
        # scaled_image_name, original_shape = extract_frame(
        #     path_to_video, current_folder, "frame_%d_scaled.jpg", frame_idx
        # )
        # print(f"Extracted scaled frame to {scaled_image_name}")
        
        # img_path = os.path.join(current_folder, scaled_image_name)
        # frame = cv2.imread(img_path)

        return process_frame(
            frame=frame,
            frame_id=frame_idx,
            model=self.model,
            dims = (frame.shape[1], frame.shape[0]),
            eval_args=self.config,
            )

def get_padding_right(shape: Tuple[int, int, int], height: int = 352) -> int:
    """Get the right padding.

    This function returns the required right padding in pixels that need to be added to the frame.

    @param shape:
        The shape of the original frame (width, height, color channels (3 -> RGB)).

    @param height:
        The fixed height the depth map model expects as input. This value should not be changed as
        long as the depth map model isn't replaced.
    @return:
        The padding in pixels that needs to be added on the right side of the frame.
    """
    h, w = shape[:2]

    r = height / float(h)
    dim = (int(w * r), height)

    width = min(dim[0], 1216)

    return 1216 - width


def resize_input(frame):
    """Resize the original frame to fit the depth map prediction input shape.

    The depth map prediction only allows images with a width of 1216 pixels and a height of 352
    pixels.
    Therefore, the size of the original image must be adjusted, which means adding right padding.
    Since the padding is black, it does not interfere with the depth map prediction.

    @param frame:
        The frame that has to be resized.

    @return:
        The resized frame with dimensions 1216x352 (width x height).
    """
    padding_right = get_padding_right(frame.shape)

    frame = imutils.resize(frame, height=352)

    if frame.shape[1] > 1216:
        frame = frame[:, 0:1215]

    frame = cv2.copyMakeBorder(
        frame,
        left=0,
        right=padding_right,
        top=0,
        bottom=0,
        borderType=cv2.BORDER_CONSTANT,
    )

    return frame


def resize_output(prediction: NDArray, shape: Tuple[int, int, int]) -> NDArray:
    """Remove the right padding.

    This function removes the right padding that was needed to generate the depth map.

    @param prediction:
        The predicted depth map in the shape (1216, 352).

    @param shape:
        The desired (original) size  of the frame (width, height, color channels).

    @return:
        The resized frame.
    """
    padding_right = get_padding_right(shape)

    prediction = prediction[:, 0:-padding_right]

    return cv2.resize(
        prediction, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_CUBIC
    )


def extract_frame(
    video_path: str, output_folder: str, output_file: str, frame_idx: int = 0
) -> Tuple[str, Tuple[int, int, int]]:
    """Extract a specific frame from the video.

    This function extracts a frame and its size from the video by using the frame count.

    @param video_path:
        The path to video.

    @param output_folder:
        The folder where the resized images should be stored.

    @param output_file:
        The name of the output file.

    @param frame_idx:
        The index of the frame that should be considered.

    @return:
        The frame name and the size is returned.
    """
    input_video = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = input_video.read()
        original_shape = frame.shape

        if frame_idx == frame_count:
            frame = resize_input(frame)
            path = os.path.join(output_folder, output_file % frame_idx)
            # cv2.imwrite(path, frame)
            return output_file % frame_idx, original_shape

        frame_count += 1

