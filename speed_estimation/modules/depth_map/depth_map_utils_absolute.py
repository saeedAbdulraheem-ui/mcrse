import os
from typing import Tuple, List

import cv2
from numpy.typing import NDArray
import torch
from PIL import Image
import numpy as np
# from Unidepth.unidepth.models import UniDepthV2


class DepthModelAbsolute:
    """This class holds the depth map generation."""
    def __init__(self, data_dir: str, path_to_video: str) -> None:
        """Create an instance of DepthModel.

        @param data_dir:
            The directory where the generated depth maps should be stored and loaded from.

        @param path_to_video:
            The path to the video that should be analyzed.
        """
        # Import here to avoid circular import issues
        self.path_to_video = path_to_video
        self.data_dir = data_dir
        self.memo: dict[int, NDArray] = {}
        self.pred_memo: dict[int, dict] = {}
        from speed_estimation.modules.depth_map.Unidepth.unidepth.models import UniDepthV2
        model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14") # load a lighter model maybe
        model.eval()
        self.model = model.to("cuda")


    def predict_depth(self, frame_id: int):
        """Predict the absolute depth map for the defined frame.

        Predict the depth map that estimates the absolute depth (in meters) in the frame.
        Each pixel value represents the distance in meters from the camera.

        @param frame_id:
            The count of the frame.

        @return:
            Returns a NDArray in the dimension of the original frame. The array holds the absolute
            distances (in meters) for each pixel. also returns a dict with predictions
        """
        if frame_id in self.memo and frame_id in self.pred_memo:
            # Depth map already predicted
            return self.memo[frame_id], self.pred_memo[frame_id]

        if len(self.memo) > 10 and len(self.pred_memo) > 1:
            # Take the mean over all depth maps
            depth_maps: List[NDArray] = [self.memo[frame] for frame in self.memo]
            latest_key = max(self.pred_memo.keys())
            return sum(depth_maps) / len(depth_maps), self.pred_memo[latest_key]

        self.memo[frame_id], self.pred_memo[frame_id] = self.load_depth(
            self.data_dir, self.path_to_video, frame_idx=frame_id
        )

        # predict depth here
        return self.memo[frame_id], self.pred_memo[frame_id]

    def extract_frame(self,
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
                # deprecated
                # frame = resize_input(frame)
                path = os.path.join(output_folder, output_file % frame_idx)
                cv2.imwrite(path, frame)
                return output_file % frame_idx, original_shape

            frame_count += 1


    def load_depth(self, current_folder: str, path_to_video: str, frame_idx: int = 0
    ) -> NDArray:
        """Load the absolute depth map.

        This function loads the depth map for one specific frame. The output size of the depth map is
        the same as the one of the original frame. Each pixel value represents the absolute distance
        in meters.

        @param current_folder:
            The folder the depth map generation should work on. Usually the folder where the input
            video is stored.

        @param path_to_video:
            The path to the video that should be analyzed.

        @param frame_idx:
            The index of the frame the depth map estimation should take as input.

        @return:
            Returns the absolute depth map (in meters) of the specified frame.
        """
        print("Depth map generation.")
        scaled_image_name, original_shape = self.extract_frame(
            path_to_video, current_folder, "frame_%d_scaled.jpg", frame_idx
        )
        print(f"Extracted scaled frame to {scaled_image_name}")
        # generate_depth now returns absolute depth in meters
        return self.generate_depth(current_folder, scaled_image_name)

    def generate_depth(self, data_folder: str, file_name: str, device: str = "cuda"):
        """
        Generate a depth map using UniDepthV2.

        Args:
            data_folder (str): Folder containing the image.
            file_name (str): Image file name.
            device (str): Device to run inference on ("cuda" or "cpu").

        Returns:
            np.ndarray: Depth map in meters, dict containing predictions: depth, intrinsics, points.
        """

        # Load and preprocess image
        img_path = os.path.join(data_folder, file_name)
        # img = cv2.imread(img_path)
        # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_resized = resize_input(img_rgb)  # Ensure correct input size

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(Image.open(img_path))).permute(2, 0, 1)

        # Inference
        with torch.no_grad():
            output = self.model.infer(img_tensor)
            if isinstance(output, dict) and "depth" in output:
                depth_map = output["depth"].squeeze().cpu().numpy()
            else:
                depth_map = output.squeeze().cpu().numpy()
        return depth_map, output
    