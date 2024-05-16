#!/usr/bin/env python

import numpy as np
import PIL.Image
import time
import unittest

from rclpy.constants import S_TO_NS
from rclpy.parameter import Parameter

from geometry_msgs.msg import Point, Pose, Quaternion
from sensor_msgs.msg import Image, CompressedImage

from ament_index_python.packages import get_package_share_directory


from launch_testing.io_handler import ActiveIoHandler


from happypose_testing_utils import (
    HappyPoseTestCase,
    assert_bbox,
    assert_and_find_detection,
    assert_and_find_marker,
    assert_pose_equal,
    assert_url_exists,
)


class SingleViewBase(HappyPoseTestCase):
    @classmethod
    def setUpClass(cls, namespace: str = "", use_compressed: bool = False) -> None:
        if cls.__name__ == SingleViewBase.__name__:
            raise unittest.SkipTest("Skipping because of case class")

        super().setUpClass(
            [("cam_1", CompressedImage if use_compressed else Image)],
            namespace,
        )
        cls.compressed = use_compressed
        image_path = get_package_share_directory("happypose_ros") + "/test"
        cls.rgb = np.asarray(PIL.Image.open(image_path + "/000629.png"))
        cls.K = np.array(
            [1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]
        )

    def setUp(self) -> None:
        if self.__class__.__name__ == SingleViewBase.__name__:
            raise unittest.SkipTest("Skipping because of case class")
        super().setUp()

    def test_01_node_startup(self, proc_output: ActiveIoHandler) -> None:
        # Check if the node outputs correct initialization
        proc_output.assertWaitFor("Node initialized", timeout=20.0)

    def test_02_check_topics(self) -> None:
        # Check if node subscribes to correct topics
        self.node.assert_node_is_subscriber(
            "cam_1/image_color" + ("/compressed" if self.compressed else ""),
            timeout=3.0,
        )
        self.node.assert_node_is_subscriber("cam_1/camera_info", timeout=3.0)
        self.node.assert_node_is_publisher("happypose/detections", timeout=3.0)
        self.node.assert_node_is_publisher("happypose/markers", timeout=3.0)
        self.node.assert_node_is_publisher("happypose/vision_info", timeout=3.0)

    def test_03_trigger_pipeline(self, proc_output: ActiveIoHandler) -> None:
        # Clear buffer before expecting any messages
        self.node.clear_msg_buffer()

        # Publish images several times to ensure they are received
        start = time.time()
        timeout = 30.0
        ready = False
        # Wait for the first pipeline to be triggered
        while time.time() - start < timeout and not ready:
            self.node.publish_image("cam_1", self.rgb, self.K)
            ready = proc_output.waitFor("HappyPose initialized", timeout=0.5)
        assert ready, "Failed to trigger the pipeline!"

    def test_04_recive_messages(self) -> None:
        self.node.assert_message_received("happypose/detections", timeout=20.0)
        self.node.assert_message_received("happypose/markers", timeout=2.0)
        self.node.assert_message_received("happypose/vision_info", timeout=2.0)

    def test_05_check_vision_info(self) -> None:
        vision_info = self.node.get_received_message("happypose/vision_info")
        self.assertEqual(vision_info.method, "cosypose")
        self.assertTrue("ycbv" in vision_info.database_location)

    def test_06_check_detection(self) -> None:
        detections = self.node.get_received_message("happypose/detections")
        self.assertEqual(
            detections.header.frame_id, "cam_1", "Incorrect frame_id in the header!"
        )
        self.assertGreaterEqual(
            len(detections.detections), 3, "Incorrect number of detected objects!"
        )
        self.assertEqual(
            detections.header,
            detections.detections[0].header,
            "Main header differs from the detection header!",
        )
        self.assertEqual(
            detections.detections[0].header,
            detections.detections[1].header,
            "Detected object headers differ!",
        )

        ycbv_02 = assert_and_find_detection(detections, "ycbv-obj_000002")
        ycbv_05 = assert_and_find_detection(detections, "ycbv-obj_000005")
        ycbv_15 = assert_and_find_detection(detections, "ycbv-obj_000015")

        minimum_score = self.node.get_params(
            ["cosypose.inference.detector.detection_th"], 5.0
        )[0].value

        self.assertGreaterEqual(ycbv_02.results[0].hypothesis.score, minimum_score)
        self.assertLessEqual(ycbv_02.results[0].hypothesis.score, 1.0)

        self.assertGreaterEqual(ycbv_05.results[0].hypothesis.score, minimum_score)
        self.assertLessEqual(ycbv_05.results[0].hypothesis.score, 1.0)

        self.assertGreaterEqual(ycbv_15.results[0].hypothesis.score, minimum_score)
        self.assertLessEqual(ycbv_15.results[0].hypothesis.score, 1.0)

        # Based on ground truth, object poses for image 629
        ycbv_02_pose = Pose(
            position=Point(**dict(zip("xyz", [0.0552, -0.0913, 1.0283]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [0.2279, 0.1563, 0.0245, 0.9607]))
            ),
        )
        ycbv_05_pose = Pose(
            position=Point(**dict(zip("xyz", [0.0946, -0.0123, 0.8399]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [-0.4171, 0.7404, -0.4506, -0.273]))
            ),
        )
        ycbv_15_pose = Pose(
            position=Point(**dict(zip("xyz", [-0.1013, 0.0329, 0.9138]))),
            orientation=Quaternion(
                **dict(zip("xyzw", [0.2526, 0.4850, 0.7653, -0.3392]))
            ),
        )

        assert_pose_equal(ycbv_02.results[0].pose.pose, ycbv_02_pose)
        assert_pose_equal(ycbv_05.results[0].pose.pose, ycbv_05_pose)
        assert_pose_equal(ycbv_15.results[0].pose.pose, ycbv_15_pose)

        # Based on ground truth ``bbox_visib`` for image 629
        assert_bbox(ycbv_02.bbox, [303, 34, 121, 236])
        # Bounding box is drawn only for a half of the bottle
        assert_bbox(ycbv_05.bbox, [394, 107, 77, 248], percent_error=50.0)
        assert_bbox(ycbv_15.bbox, [64, 204, 267, 151])

    def test_07_check_markers(self) -> None:
        detections = self.node.get_received_message("happypose/detections")
        ycbv_02 = assert_and_find_detection(detections, "ycbv-obj_000002")
        ycbv_05 = assert_and_find_detection(detections, "ycbv-obj_000005")
        ycbv_15 = assert_and_find_detection(detections, "ycbv-obj_000015")

        markers = self.node.get_received_message("happypose/markers")
        self.assertEqual(
            len(markers.markers),
            len(detections.detections),
            "Number of markers differs from number of detections!",
        )
        self.assertEqual(
            markers.markers[0].header,
            detections.header,
            "Markers have different header than detections!",
        )

        # Check if marker poses are correct
        ycbv_02_marker = assert_and_find_marker(markers, "ycbv-obj_000002", "ycbv")
        ycbv_05_marker = assert_and_find_marker(markers, "ycbv-obj_000005", "ycbv")
        ycbv_15_marker = assert_and_find_marker(markers, "ycbv-obj_000015", "ycbv")

        assert_pose_equal(ycbv_02.results[0].pose.pose, ycbv_02_marker.pose)
        assert_pose_equal(ycbv_05.results[0].pose.pose, ycbv_05_marker.pose)
        assert_pose_equal(ycbv_15.results[0].pose.pose, ycbv_15_marker.pose)

        marker_lifetime = self.node.get_params(["visualization.markers.lifetime"], 5.0)[
            0
        ].value
        for marker in markers.markers:
            # Check if lifetime matches one from parameter
            lifetime_sec = marker.lifetime.sec + marker.lifetime.nanosec / S_TO_NS
            self.assertAlmostEqual(
                lifetime_sec,
                marker_lifetime,
                places=6,
                msg="Lifetime doesn't match one from the ROS parameter",
            )
            # Check scale of the mesh
            self.assertAlmostEqual(
                marker.scale.x, 0.001, msg="Marker has incorrect scale on axis x"
            )
            self.assertAlmostEqual(
                marker.scale.y, 0.001, msg="Marker has incorrect scale on axis y"
            )
            self.assertAlmostEqual(
                marker.scale.z, 0.001, msg="Marker has incorrect scale on axis z"
            )
            # Check color
            self.assertAlmostEqual(
                marker.color.r,
                1.0,
                msg="Marker has incorrect color intensity on channel r",
            )
            self.assertAlmostEqual(
                marker.color.g,
                1.0,
                msg="Marker has incorrect color intensity on channel g",
            )
            self.assertAlmostEqual(
                marker.color.b,
                1.0,
                msg="Marker has incorrect color intensity on channel b",
            )
            self.assertAlmostEqual(
                marker.color.a,
                1.0,
                msg="Marker has incorrect color intensity on channel a",
            )
            # Check if mesh exists
            assert_url_exists(marker.mesh_resource)
            self.assertTrue(
                marker.mesh_use_embedded_materials,
                "Mesh expected to use 'mesh_use_embedded_materials'!",
            )

    def test_08_dynamic_params_labels_to_keep(
        self, proc_output: ActiveIoHandler
    ) -> None:
        label_to_keep = "ycbv-obj_000002"
        self.node.set_params(
            [
                Parameter(
                    "cosypose.inference.labels_to_keep",
                    Parameter.Type.STRING_ARRAY,
                    [label_to_keep],
                )
            ]
        )

        # Clear old messages
        self.node.clear_msg_buffer()

        # Publish new to trigger parameter change
        self.node.publish_image("cam_1", self.rgb, self.K)
        proc_output.assertWaitFor("Parameter change occurred", timeout=0.5)
        self.node.assert_message_received("happypose/detections", timeout=20.0)

        self.node.assert_message_received("happypose/detections", timeout=20.0)
        detections = self.node.get_received_message("happypose/detections")

        self.assertEqual(
            len(detections.detections), 1, msg="Detections were not filtered!"
        )
        self.assertEqual(
            detections.detections[0].results[0].hypothesis.class_id,
            label_to_keep,
            msg="Filtered label is not the same as the expected one!",
        )

    def test_09_dynamic_params_markers(self) -> None:
        lifetime = 2137.0
        self.node.set_params(
            [
                Parameter(
                    "visualization.markers.dynamic_opacity",
                    Parameter.Type.BOOL,
                    True,
                ),
                Parameter(
                    "visualization.markers.lifetime",
                    Parameter.Type.DOUBLE,
                    lifetime,
                ),
            ]
        )

        # Clear old messages
        self.node.clear_msg_buffer()

        # Publish new to trigger parameter change
        self.node.publish_image("cam_1", self.rgb, self.K)
        self.node.assert_message_received("happypose/markers", timeout=20.0)
        self.node.assert_message_received("happypose/detections", timeout=20.0)

        detections = self.node.get_received_message("happypose/detections")
        markers = self.node.get_received_message("happypose/markers")

        self.assertAlmostEqual(
            markers.markers[0].color.a,
            detections.detections[0].results[0].hypothesis.score,
            msg="Dynamic opacity did not apply!",
        )

        marker_lifetime = markers.markers[0].lifetime
        lifetime_sec = marker_lifetime.sec + marker_lifetime.nanosec / S_TO_NS
        self.assertAlmostEqual(
            lifetime,
            lifetime_sec,
            places=6,
            msg="Marker lifetime did not change!",
        )

    def test_10_dynamic_params_labels_to_keep_reset(self: ActiveIoHandler) -> None:
        # Undoo filtering of the labels
        self.node.set_params(
            [
                Parameter(
                    "cosypose.inference.labels_to_keep",
                    Parameter.Type.STRING_ARRAY,
                    [""],
                )
            ]
        )

        # Clear old messages
        self.node.clear_msg_buffer()

        # Publish new to trigger parameter change
        self.node.publish_image("cam_1", self.rgb, self.K)
        self.node.assert_message_received("happypose/detections", timeout=20.0)
        detections = self.node.get_received_message("happypose/detections")

        self.assertGreaterEqual(
            len(detections.detections),
            3,
            msg="Detections were not brought back to being unfiltered!",
        )
