from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
from speed_estimation.modules.depth_map.depth_map_utils import DepthModel
from numpy.typing import NDArray
from scipy.linalg import norm
from scipy.spatial import distance
from speed_estimation.utils.speed_estimation import (
    Line,
    Point,
    TrackingBox,
    get_intersection,
)

# Mapping from YOLO class IDs to average object lengths in meters
# Mapping from YOLOv4 class IDs to average horizontal object lengths in meters
# (indices based on YOLOv4: 0=person, 1=bicycle, 2=car, 3=motorbike)
YOLO_CLASS_ID_TO_AVG_LENGTH = {
    0: 0.5,  # person (average horizontal width)
    # 1: 1.7,  # bicycle (horizontal length)
    2: 5.5,  # car (horizontal length)
    # 3: 2.1,  # motorbike (horizontal length)
}


@dataclass
class CameraPoint:
    """A Camera Point in the frame.

    A camera point is always two-dimensional, in a given frame and uses pixels as unit.

    @param frame
        The count of the frame the point belongs to.
    @param x_coord
        The x coordinate of the point in pixels.
    @param y_coord
        The y coordinate of the point in pixels.
    """

    frame: int
    x_coord: int
    y_coord: int

    def coords(self) -> NDArray:
        """Get coordinates of the CameraPoint.

        @return:
            Returns the x and y coordinate of the point.
        """
        return np.array([self.x_coord, self.y_coord])


@dataclass
class WorldPoint:
    """A three-dimensional world point.

    A world point is always three-dimensional and the projection of a CameraPoint into the real
    world.

    @param frame
        The count of the frame the point belongs to.
    @param x_coord
        The x coordinate of the point in pixels.
    @param y_coord
        The y coordinate of the point in pixels.
    @param z_coord
        The z coordinate of the point in pixels.
    """

    frame: int
    x_coord: float
    y_coord: float
    z_coord: float

    def coords(self) -> NDArray:
        """Get coordinates of WorldPoint.

        @return:
            Returns the x, y and z coordinate of the point.
        """
        return np.array([self.x_coord, self.y_coord, self.z_coord])


class GroundTruthEvent(NamedTuple):
    """A car tracked during the calibration phase.

    GroundTruth events are created from cars that have been tracked during the calibration phase.
    Cars and their tracking boxes are used as reference and calibration ground truth.

    @param coords1
        The first tuple holding the frame count, x coordinate and y coordinate
    @param coords2
        The second tuple holding the frame count, x coordinate and y coordinate
    @param distance
        The distance that lies between coord1 and coord2. Per default hard coded to 6m as this is
        a reasonable value for the ground truth length of a car.
    """

    coords1: Tuple
    coords2: Tuple
    distance: float


@dataclass
class GeometricModel:
    """Geometric model that is used to retrieve the distance between two points.

    This model hold the most important information to get the distance between two CameraPoints in
    meters.

    @param depth_model
        The depth model that has to be applied to predict the depth of the whole frame.
    @param focal_length
        Focal length of the camera recording the video/stream.
    @param scaling_x
        Scaling factor in x direction to translate pixels into meters.
    @param scaling _y
        Scaling factor in y direction to translate pixels into meters.
    @param center_x
        The center point in x direction.
    @param center_y
        The center point in y direction.
    """

    def __init__(self, depth_model) -> None:
        """Create an instance of GeometricModel.

        @param depth_model:
            The depth model that has to be applied to predict the depth of the whole frame.
        """
        self.depth_model = depth_model
        # Calibration parameters from KITTI (example: K_00, S_00)
        # S_00: 1.392000e+03 5.120000e+02
        # K_00: 9.842439e+02 0.000000e+00 6.900000e+02 0.000000e+00 9.808141e+02 2.331966e+02 0.000000e+00 0.000000e+00 1.000000e+00
        # Use K_00 for focal length and principal point
        self.focal_length: float = 984.2439  # fx from K_00
        self.scaling_x: float = 1.0 / 984.2439  # 1/fx
        self.scaling_y: float = 1.0 / 980.8141  # 1/fy from K_00
        self.center_x: float = 690.0  # cx from K_00
        self.center_y: float = 233.1966  # cy from K_00
        self.scale_factor: float = 1.0

    def set_normalization_axes(self, center_x: int, center_y: int) -> None:
        """Set the normalization axis.

        @param center_x:
            The x coordinate of the center point.
        @param center_y:
            The y coordinate of the center point.
        """
        self.center_x = center_x
        self.center_y = center_y

    def get_unscaled_distance_from_camera_points(
        self, frame, cp1: CameraPoint, cp2: CameraPoint
    ) -> float:
        """Get the unscaled distance between two two-dimensional CameraPoints.

        @param cp1:
            First CameraPoint.
        @param cp2:
            Second CameraPoint.
        @return:
            Returns the unscaled distance between those CameraPoints.
        """
        unscaled_wp1 = self.__get_unscaled_world_point(frame, cp1)
        unscaled_wp2 = self.__get_unscaled_world_point(frame, cp2)

        return self.__calculate_distance_between_world_points(
            unscaled_wp1, unscaled_wp2
        )

    def get_distance_from_camera_points(
        self, frame, cp1: CameraPoint, cp2: CameraPoint
    ) -> float:
        """Get the scaled distance between two two-dimensional CameraPoints in meters.

        @param cp1:
            First CameraPoint.
        @param cp2:
            Second CameraPoint.
        @return:
            Returns the scaled distance between those CameraPoints in meters.
        """
        return self.scale_factor * self.__get_unscaled_distance_from_camera_points(
            frame, cp1, cp2
        )

    def get_scaled_distance_from_camera_points(
        self, cp1: CameraPoint, cp2: CameraPoint
    ) -> float:
        """Get the scaled distance between two two-dimensional CameraPoints in meters.

        @param cp1:
            First CameraPoint.
        @param cp2:
            Second CameraPoint.
        @return:
            Returns the scaled distance between those CameraPoints in meters.
        """
        return self.__get_scaled_distance_from_camera_points(cp1, cp2)

    def __get_unscaled_world_point(self, frame, cp: CameraPoint) -> WorldPoint:
        normalised_u = self.scaling_x * (cp.x_coord - self.center_x)
        normalised_v = self.scaling_y * (cp.y_coord - self.center_y)

        # we relabel the axis here to deal with different reference conventions
        _, theta, phi = self.__cartesian_to_spherical(
            x=self.focal_length, y=normalised_u, z=normalised_v
        )

        result = self.depth_model.predict_depth(cp.frame, frame)
        if result is None:
            raise ValueError(f"Depth model returned None for frame {cp.frame}")
        depth_map = result[0] if isinstance(result, tuple) else result
        np_depth_map = np.array(depth_map)
        unscaled_depth: float = np_depth_map[cp.y_coord, cp.x_coord]

        # we also mirror theta around pi and phi around 0
        theta = np.pi - theta
        phi = -phi

        x, y, z = self.__spherical_to_cartesian(r=unscaled_depth, theta=theta, phi=phi)

        # relabeling again for the world reference system
        return WorldPoint(frame=cp.frame, x_coord=y, y_coord=z, z_coord=x)

    def __get_scaled_world_point(self, frame, cp: CameraPoint) -> WorldPoint:
        """
        Computes the scaled world coordinates of a given camera point using depth estimation and camera intrinsics.

        Args:
            cp (CameraPoint): The camera point containing frame and pixel coordinates.

        Returns:
            WorldPoint: The corresponding world point with coordinates (x, y, z) in the world reference system.

        Notes:
            - Updates camera intrinsics (focal length, principal point) from depth model predictions.
            - Normalizes pixel coordinates based on intrinsics.
            - Converts normalized coordinates from cartesian to spherical, applies axis relabeling and mirroring.
            - Uses predicted depth to scale the spherical coordinates and converts back to cartesian.
            - Returns the world point with relabeled axes to match the world reference system.
        """
        depth_map, preds = self.depth_model.predict_depth(frame, cp.frame)
        # update camera intrinsics from preds
        intrinsics = preds["intrinsics"]
        self.focal_length = float(intrinsics[0, 0, 0])
        self.scaling_x = 1.0 / float(intrinsics[0, 0, 0])
        self.scaling_y = 1.0 / float(intrinsics[0, 1, 1])

        normalised_u = self.scaling_x * (cp.x_coord - self.center_x)
        normalised_v = self.scaling_y * (cp.y_coord - self.center_y)

        # we relabel the axis here to deal with different reference conventions
        _, theta, phi = self.__cartesian_to_spherical(
            x=self.focal_length, y=normalised_u, z=normalised_v
        )

        # Use the scaled depth model (depth in meters)
        scaled_depth: float = depth_map[cp.y_coord, cp.x_coord]

        # we also mirror theta around pi and phi around 0
        theta = np.pi - theta
        phi = -phi

        x, y, z = self.__spherical_to_cartesian(r=scaled_depth, theta=theta, phi=phi)

        # relabeling again for the world reference system
        return WorldPoint(frame=cp.frame, x_coord=y, y_coord=z, z_coord=x)

    def get_scaled_world_point(self, frame, cp: CameraPoint) -> WorldPoint:
        """
        Computes the scaled world coordinates of a given camera point using depth estimation and camera intrinsics.

        Args:
            cp (CameraPoint): The camera point containing frame and pixel coordinates.

        Returns:
            WorldPoint: The corresponding world point with coordinates (x, y, z) in the world reference system.

        Notes:
            - Updates camera intrinsics (focal length, principal point) from depth model predictions.
            - Normalizes pixel coordinates based on intrinsics.
            - Converts normalized coordinates from cartesian to spherical, applies axis relabeling and mirroring.
            - Uses predicted depth to scale the spherical coordinates and converts back to cartesian.
            - Returns the world point with relabeled axes to match the world reference system.
        """
        depth_map, preds = self.depth_model.predict_depth(frame, cp.frame)

        # Write the depth map to an image file for debugging
        # Normalize depth map to 0-255 and apply a colormap for better visualization
        # TODO(SAID): Remove this debug code in production
        # import cv2
        # debug_depth_img = (255 * (depth_map - np.min(depth_map)) / (np.ptp(depth_map) + 1e-8)).astype(np.uint8)
        # debug_depth_img_color = cv2.applyColorMap(debug_depth_img, cv2.COLORMAP_JET)
        # cv2.imwrite(f"debug/debug_depth_frame_{cp.frame}.png", debug_depth_img_color)

        # update camera intrinsics from preds
        intrinsics = preds["intrinsics"]
        self.focal_length = float(intrinsics[0, 0, 0])
        self.scaling_x = 1.0  # / float(intrinsics[0, 0, 0])
        self.scaling_y = 1.0  # / float(intrinsics[0, 1, 1])

        normalised_u = self.scaling_x * (cp.x_coord - self.center_x)
        normalised_v = self.scaling_y * (cp.y_coord - self.center_y)

        # we relabel the axis here to deal with different reference conventions
        _, theta, phi = self.__cartesian_to_spherical(
            x=self.focal_length, y=normalised_u, z=normalised_v
        )

        # Use the scaled depth model (depth in meters)
        # Get the median depth in a 5x5 window around the point of interest
        y, x = cp.y_coord, cp.x_coord
        h, w = depth_map.shape
        y_min = max(0, y - 2)
        y_max = min(h, y + 3)
        x_min = max(0, x - 2)
        x_max = min(w, x + 3)
        window = depth_map[y_min:y_max, x_min:x_max]
        scaled_depth: float = float(np.median(window))

        # we also mirror theta around pi and phi around 0
        theta = np.pi - theta
        phi = -phi

        x, y, z = self.__spherical_to_cartesian(r=scaled_depth, theta=theta, phi=phi)

        # relabeling again for the world reference system
        return WorldPoint(frame=cp.frame, x_coord=y, y_coord=z, z_coord=x)

    def __get_world_point(self, cp: CameraPoint) -> WorldPoint:
        unscaled_world_point = self.__get_unscaled_world_point(cp)
        unscaled_world_point.x_coord *= self.scale_factor
        unscaled_world_point.y_coord *= self.scale_factor
        unscaled_world_point.z_coord *= self.scale_factor

        return unscaled_world_point

    def __get_camera_point(self, wp: WorldPoint) -> CameraPoint:
        x, y, z = wp.coords()
        # Note that we here relabel the coordinates to keep the two coordinate systems aligned!
        r, theta, phi = self.__cartesian_to_spherical(x=z, y=x, z=y)
        # we also mirror theta around pi and phi around 0
        theta = np.pi - theta
        phi = -phi
        z_inner, x_inner, y_inner = self.__spherical_to_cartesian(
            r=np.abs(self.focal_length / (np.sin(theta) * np.cos(phi))),
            theta=theta,
            phi=phi,
        )

        assert np.isclose(z_inner, self.focal_length)

        return CameraPoint(frame=wp.frame, x_coord=x_inner, y_coord=y_inner)

    @staticmethod
    def __spherical_to_cartesian(r, theta, phi):
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def __cartesian_to_spherical(x, y, z):
        r = norm([x, y, z])
        if r == 0:
            return 0, 0, 0
        theta = np.arccos(z / r)
        if norm([x, y]) == 0:
            return z, 0, 0
        phi = np.sign(y) * np.arccos(x / norm([x, y]))
        return r, theta, phi

    @staticmethod
    def __calculate_distance_between_world_points(
        wp1: WorldPoint, wp2: WorldPoint
    ) -> float:
        return norm(wp1.coords() - wp2.coords())

    def __get_unscaled_distance_from_camera_points(
        self, frame, cp1: CameraPoint, cp2: CameraPoint
    ):
        unscaled_wp1 = self.__get_unscaled_world_point(frame, cp1)
        unscaled_wp2 = self.__get_unscaled_world_point(frame, cp2)
        return self.__calculate_distance_between_world_points(
            unscaled_wp1, unscaled_wp2
        )

    def __get_scaled_distance_from_camera_points(
        self, cp1: CameraPoint, cp2: CameraPoint
    ):
        scaled_wp1 = self.__get_scaled_world_point(cp1)
        scaled_wp2 = self.__get_scaled_world_point(cp2)
        return self.__calculate_distance_between_world_points(scaled_wp1, scaled_wp2)


def offline_scaling_factor_estimation_from_least_squares(
    frame,
    geometric_model: GeometricModel,
    ground_truths: List,
) -> float:
    """Get the scaling factor that should be applied for the speed estimation.

    By applying the least square method to multiple unscaled length predictions this method
    extracts the scaling factor.

    @param geometric_model:
        The GeoMetric model that should be applied to find the scaling factor.
    @param ground_truths:
        The ground truth events that where detected in the video (cars). Each tuple in the list
        holds two points and the distance between those points in meters.
    @return:
        The scaling factor that should be applied for the speed estimation.
    """
    unscaled_predictions = []
    labels = []

    for coords1, coords2, distance in ground_truths:
        f1, u1, v1 = coords1
        f2, u2, v2 = coords2
        cp1 = CameraPoint(frame=f1, x_coord=u1, y_coord=v1)
        cp2 = CameraPoint(frame=f2, x_coord=u2, y_coord=v2)
        unscaled_predictions.append(
            geometric_model.get_unscaled_distance_from_camera_points(frame, cp1, cp2)
        )
        labels.append(distance)

    # return optimal scaling factor under least sum of squares estimator
    return np.dot(unscaled_predictions, labels) / np.dot(
        unscaled_predictions, unscaled_predictions
    )


def __online_scaling_factor_estimation_from_least_squares(stream_of_events):
    ###################
    # TODO: integrate
    ###################
    counter = 0

    depth_model = DepthModel(data_dir="")
    geometric_model = GeometricModel(depth_model=depth_model)

    mean_predictions_two_norm = 0
    mean_prediction_dot_distance = 0

    while stream_of_events.has_next():
        counter += 1

        # calibration phase uses a stream of ground truth events
        frame, coords1, coords2, true_distance = stream_of_events.next()

        # this would e.g. be the pixel coordinates for the corners of a bounding box
        _, u1, v1 = coords1
        _, u2, v2 = coords2

        # calculate the unscaled predicted distance
        cp1 = CameraPoint(frame=frame, x_coord=u1, y_coord=v1)
        cp2 = CameraPoint(frame=frame, x_coord=u2, y_coord=v2)
        prediction = geometric_model.get_unscaled_distance_from_camera_points(cp1, cp2)

        mean_predictions_two_norm = (1 - 1 / counter) * mean_predictions_two_norm + (
            prediction**2
        ) / counter
        mean_prediction_dot_distance = (
            1 - 1 / counter
        ) * mean_prediction_dot_distance + (prediction * true_distance) / counter

        geometric_model.scale_factor = (
            mean_prediction_dot_distance / mean_predictions_two_norm
        )

        # once calibration is finished, we can start using the geometric_model to perform actual predictions for
        # velocities, however, even then we can still continue updating the scale factor


def get_ground_truth_events(
    tracking_boxes: Dict[int, List[TrackingBox]],
) -> List[GroundTruthEvent]:
    """Get ground truth events to calculate the scaling factor.

    The method takes tracking boxes as input and derives two points with the corresponding distance
    in meters.

    @param tracking_boxes:
        The TrackingBoxes of the cars that should be analyzed.
    @return:
        Returns a list of GroundTruth events extracted from the TrackingBoxes.
    """
    # extract medium pixel distance traveled by object
    box_distances = []
    for object_id in tracking_boxes:
        start_box = tracking_boxes[object_id][0]
        end_box = tracking_boxes[object_id][-1]
        tracking_box_distance = distance.euclidean(
            [start_box.center_x, start_box.center_y],
            [end_box.center_x, end_box.center_y],
        )
        box_distances.append(tracking_box_distance)

    median_distance = np.percentile(np.array(box_distances), 50)

    # extract ground truth value for each tracking box
    ground_truth_events = []
    for object_id in tracking_boxes:
        center_points = np.array(
            [(box.center_x, box.center_y) for box in tracking_boxes[object_id]]
        )
        start_box = center_points[0]
        end_box = center_points[-1]
        tracking_box_distance = distance.euclidean(start_box, end_box)
        if (
            len(center_points) < 2
            or len(center_points) > 750
            or tracking_box_distance < median_distance
        ):
            continue
        center_points_line = Line(Point(*center_points[0]), Point(*center_points[-1]))

        # extract ground truth value for each tracking box
        for box in tracking_boxes[object_id]:
            object_class = box.class_id
            if object_class not in YOLO_CLASS_ID_TO_AVG_LENGTH:
                continue
            average_object_length = YOLO_CLASS_ID_TO_AVG_LENGTH[object_class]
            # check each of the four lines, spanned by the bounding box rectangle
            upper_line = Line(
                Point(box.x_coord, box.y_coord),
                Point(box.x_coord + box.width, box.y_coord),
            )
            right_line = Line(
                Point(box.x_coord + box.width, box.y_coord),
                Point(box.x_coord + box.width, box.y_coord + box.height),
            )
            lower_line = Line(
                Point(box.x_coord, box.y_coord + box.height),
                Point(box.x_coord + box.width, box.y_coord + box.height),
            )
            left_line = Line(
                Point(box.x_coord, box.y_coord),
                Point(box.x_coord, box.y_coord + box.height),
            )

            intersections = []
            for bounding_box_line in [upper_line, right_line, lower_line, left_line]:
                intersection = get_intersection(center_points_line, bounding_box_line)
                if intersection is not None:
                    intersections.append(intersection)

            if len(intersections) == 2:
                # append ground truth only if line fully cuts bounding box
                intersect1, intersect2 = intersections
                ground_truth_events.append(
                    GroundTruthEvent(
                        (
                            box.frame_count,
                            int(intersect1.x_coord),
                            int(intersect1.y_coord),
                        ),
                        (
                            box.frame_count,
                            int(intersect2.x_coord),
                            int(intersect2.y_coord),
                        ),
                        average_object_length,
                    )
                )

    return ground_truth_events
