import time
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from abc import ABC, abstractmethod
from typing import final
import math
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from happypose.toolbox.inference.types import ObservationTensor, DetectionsType
from happypose.toolbox.inference.utils import (
    filter_detections,
    make_detections_from_object_data,
)
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset

from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper,
)
from happypose.pose_estimators.cosypose.cosypose.integrated.multiview_predictor import (
    MultiviewScenePredictor,
)
from happypose.pose_estimators.cosypose.cosypose.utils.tensor_collection import (
    PandasTensorCollection,
)
from happypose.pose_estimators.cosypose.cosypose.datasets.bop_object_datasets import (
    BOPObjectDataset,
)
from happypose.pose_estimators.cosypose.cosypose.lib3d.rigid_mesh_database import (
    MeshDataBase,
)

from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.datasets.scene_dataset import ObjectData


#! ==============================================================================
# #! ros2 logger
from rclpy.impl import rcutils_logger

# self.logger = rcutils_logger.RcutilsLogger(name="HHPose-pipeline")
# self.logger.info("Starting")
# !==============================================================================


class InferencePipeline(ABC):
    """TODO"""

    def __init__(self, params: dict) -> None:
        """TODO"""
        self.logger = rcutils_logger.RcutilsLogger(name="General-pipeline")

        return 0

    @final
    def update_params(self, params: dict) -> None:
        """Updates parameters used by the HappyPose.

        :param params: Parameters used to initialize the HappyPose pipeline.
            On runtime to update inference parameters.
        :type params: dict
        """
        self.logger.info("Updating parameters")

        self._inference_args = params["cosypose"]["inference"]
        self._inference_args["labels_to_keep"] = (
            self._inference_args["labels_to_keep"]
            if self._inference_args["labels_to_keep"] != [""]
            else None
        )

    @abstractmethod
    def get_dataset(self) -> RigidObjectDataset:
        """TODO"""
        pass

    @abstractmethod
    def __call__(self, observation: ObservationTensor) -> Union[None, dict]:
        """TODO"""
        pass


class CosyPosePipeline(InferencePipeline):
    """Object wrapping HappyPose pipeline extracting its calls from the main ROS node."""

    def __init__(self, params: dict) -> None:
        """Creates HappyPosePipeline object and starts loading Torch models to the memory.

        :param params: Parameters used to initialize the HappyPose pipeline.
        :type params: dict
        """
        self.logger = rcutils_logger.RcutilsLogger(name="CHPose-pipeline")
        self.logger.info("Starting CosyPose inference pipeline")

        super().__init__(params)
        self._params = params
        self._device = self._params["device"]

        self._wrapper = CosyPoseWrapper(
            dataset_name=self._params["cosypose"]["dataset_name"],
            model_type=self._params["cosypose"]["model_type"],
            depth_refiner_type=(
                self._params["cosypose"]["depth_refiner_type"]
                if self._params["use_depth"]
                else None
            ),
            **self._params["cosypose"]["renderer"],
        )

        self.update_params(self._params)

        self._multiview = len(self._params["camera_names"]) > 1
        if self._multiview:
            dir = self._wrapper.object_dataset.ds_dir.as_posix()
            label_format = self._params["cosypose"]["dataset_name"] + "-{label}"
            object_ds = BOPObjectDataset(dir, label_format)
            mesh_db = MeshDataBase.from_object_ds(object_ds)
            self._mv_predictor = MultiviewScenePredictor(mesh_db)

    def get_dataset(self) -> RigidObjectDataset:
        """Returns rigid object dataset used by HappyPose pose estimator

        :return: Dataset used by HappyPose pose estimator
        :rtype: RigidObjectDataset
        """
        dataset = self._wrapper.object_dataset
        if self._inference_args["labels_to_keep"] is None:
            return dataset
        return dataset.filter_objects(self._inference_args["labels_to_keep"])

    def __call__(self, observation: ObservationTensor) -> Union[None, dict]:
        """Performs sequence of actions to estimate pose and optionally merge
        multiview results.

        :param observation: Tensor containing camera information and incoming images.
        :type observation: happypose.toolbox.inference.types.ObservationTensor
        :return: Dictionary with final detections. If pipeline failed or nothing
            was detected None is returned
        :rtype: Union[None, dict]
        """
        timings = {}
        t1 = time.perf_counter()

        detections = self._wrapper.pose_predictor.detector_model.get_detections(
            observation,
            output_masks=False,
            **self._inference_args["detector"],
        )

        t2 = time.perf_counter()
        timings["detections"] = t2 - t1

        detections = filter_detections(
            detections, self._inference_args["labels_to_keep"]
        )

        if len(detections.infos) == 0:
            return None

        cosypose_predictions, _ = self._wrapper.pose_predictor.run_inference_pipeline(
            observation,
            detections=detections,
            run_detector=False,
            data_TCO_init=None,
            **self._inference_args["pose_estimator"],
        )

        t3 = time.perf_counter()
        timings["single_view"] = t3 - t2

        if self._params["use_depth"]:
            object_predictions, extra_data_depth_ref = (
                self._wrapper.depth_refiner.refine_poses(
                    predictions=cosypose_predictions,
                    depth=observation.depth,
                    K=observation.K,
                    **self._inference_args[
                        self._params["cosypose"]["depth_refiner_type"]
                    ],
                )
            )

            # Select only valid ICP results (retval of value 0)
            valid_icp_ids = np.logical_not(extra_data_depth_ref["retvals_icp"])
            object_predictions = object_predictions[valid_icp_ids]

        else:
            object_predictions = cosypose_predictions

        t4 = time.perf_counter()
        timings["depth_refinement"] = t4 - t3

        if not self._multiview:
            object_predictions.cpu()
            timings["total"] = time.perf_counter() - t1

            return {
                "infos": object_predictions.infos,
                "poses": object_predictions.poses,
                "bboxes": detections.tensors["bboxes"].int().cpu(),
                "timings": timings,
            }

        object_predictions.infos = object_predictions.infos.rename(
            columns={"batch_im_id": "view_id"}
        )
        # Arbitrary scene_id and group_id
        object_predictions.infos["scene_id"] = 42
        object_predictions.infos["group_id"] = 0

        cameras = PandasTensorCollection(
            K=observation.K,
            infos=pd.DataFrame({"view_id": np.arange(observation.batch_size)}),
        )
        cameras.infos["scene_id"] = 1
        cameras.infos["batch_im_id"] = 0

        predictions = self._mv_predictor.predict_scene_state(
            candidates=object_predictions,
            cameras=cameras,
            use_known_camera_poses=False,
            **self._inference_args["multiview"],
        )

        object_predictions = predictions["scene/objects"].cpu()
        cameras_pred = predictions["scene/cameras"].cpu()

        if len(predictions["scene/objects"].infos) == 0:
            return None

        # Choose view group with the maximum score, as the most likely one
        df_tmp = object_predictions.infos.groupby("view_group").sum(["score"])
        max_view_group = df_tmp["score"].idxmax()
        object_predictions.infos = object_predictions.infos[
            object_predictions.infos["view_group"] == max_view_group
        ]
        cameras_pred.infos = cameras_pred.infos[
            cameras_pred.infos["view_group"] == max_view_group
        ]

        # Normalize score to range 0 - 1
        predictions["scene/objects"].infos["score"] /= predictions[
            "scene/objects"
        ].infos["n_cand"]

        t5 = time.perf_counter()
        timings["depth_refinement"] = t5 - t4
        timings["total"] = t5 - t1

        return {
            "infos": object_predictions.infos,
            "poses": object_predictions.TWO,
            "bboxes": None,
            "camera_infos": cameras_pred.infos,
            "camera_poses": cameras_pred.TWC,
            "camera_K": cameras_pred.K,
            "timings": timings,
        }


class MegaPosePipeline(InferencePipeline):
    """TODO"""

    def __init__(self, params: dict) -> None:
        """Creates MegaPosePipeline object and starts loading Torch models to the memory.

        :param params: Parameters used to initialize the HappyPose pipeline.
        :type params: dict
        """

        self.logger = rcutils_logger.RcutilsLogger(name="MHPose-pipeline")
        self.logger.info("Starting MegaPose inference pipeline")

        super().__init__(params)
        self._params = params
        self._device = self._params["device"]

        self.update_params(self._params)

        object_dataset = self.get_dataset()

        self.logger.info(
            "Loading model " + str(self._params["megapose"]["model_name"]) + "."
        )
        self.model_info = NAMED_MODELS[self._params["megapose"]["model_name"]]
        self.pose_estimator = load_named_model(
            self._params["megapose"]["model_name"], object_dataset
        ).to(self._device)
        self.pose_estimator._SO3_grid = self.pose_estimator._SO3_grid[::8]

        # load yolo model
        yolo_model_path = "/docker_files/happypose_ros_data/yolo-checkpoints/yolo11n.pt"  # "/docker_files/happypose_ros_data/yolo-checkpoints/bar-holder-stripped-bi-v2.pt"   # to pass as param or not?
        self.yolo_model = YOLO(yolo_model_path)

    def get_dataset(self) -> RigidObjectDataset:
        """Returns rigid object dataset used by MegaPose pose estimator

        :return: Dataset used by MegaPose pose estimator
        :rtype: RigidObjectDataset
        """
        rigid_objects = []
        mesh_dir = Path(self._params["megapose"]["mesh_dir"])
        assert mesh_dir.exists(), f"Missing mesh directory {mesh_dir}"

        for mesh_path in mesh_dir.iterdir():
            if mesh_path.suffix in {".obj", ".ply"}:
                obj_name = mesh_path.with_suffix("").name
                rigid_objects.append(
                    RigidObject(
                        label=obj_name,
                        mesh_path=mesh_path,
                        mesh_units=self._params["megapose"]["mesh_units"],
                    ),
                )
        rigid_object_dataset = RigidObjectDataset(rigid_objects)
        return rigid_object_dataset

    def __call__(self, observation: ObservationTensor) -> Union[None, dict]:
        """TODO"""

        """Performs sequence of actions to estimate pose and optionally merge
        multiview results.

        :param observation: Tensor containing camera information and incoming images.
        :type observation: happypose.toolbox.inference.types.ObservationTensor
        :return: Dictionary with final detections. If pipeline failed or nothing
            was detected None is returned
        :rtype: Union[None, dict]
        """
        timings = {}
        t1 = time.perf_counter()

        # replace by a yolo for now
        # get detections with a yolo
        # rgb_tensor = observation.rgb # TODO: add to toolbox
        rgb_tensor = observation.images[
            :, 0:3
        ].cpu()  # * copy to cpu before doing operations on it
        rgb_image = rgb_tensor.numpy()
        rgb_image = rgb_image[0, :, :, :]
        rgb_image = np.moveaxis(rgb_image, [0, 1], [2, 0])

        # conversion from float32 to unit8 required for YOLO
        rgb_image *= 255
        rgb_image = rgb_image.astype(np.uint8)
<<<<<<< HEAD

        # ! debug ==============================================================
        plt.imshow(rgb_image)
        plt.show()
        # ! ====================================================================

=======


>>>>>>> 5d3f437 (cleanup the visualisation for debug)
        # detections = self.yolo_detector(self.yolo_model, rgb_image) #! runs but does not return a detection
        # test =========================================================================
        yolo_model_path = "/docker_files/happypose_ros_data/yolo-checkpoints/yolo11n.pt"  # "/docker_files/happypose_ros_data/yolo-checkpoints/bar-holder-stripped-bi-v2.pt"   # to pass as param or not?
        yolo_model = YOLO(yolo_model_path)
        yolo_results = yolo_model(rgb_image, stream=True)

        self.logger.info(str(yolo_results))
        for r in yolo_results:
            boxes = r.boxes
            min_confidence = 0
            box_w_max_conf = None
            self.logger.info("boxes len:" + str(len(boxes)))

            if len(boxes) > 0:
                self.logger.info("yolo found something")
                for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    # ! debug ==================================================
                    x1, y1, x2, y2 = box.xyxy[0]
                    self.logger.info(
                        "confidence: "
                        + str(confidence)
                        + "\t"
                        + str(int(x1))
                        + " "
                        + str(int(x2))
                        + " "
                        + str(int(y1))
                        + " "
                        + str(int(y2))
                    )
                    # ! ========================================================

                    if confidence > min_confidence:
                        box_w_max_conf = box
                        min_confidence = confidence

                # bounding box coordinates
                x1, y1, x2, y2 = box_w_max_conf.xyxy[0]
                x1, y1, x2, y2 = (
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                )  # convert to int values
                self.logger.info(
                    "Max confidence: "
                    + str(int(x1))
                    + " "
                    + str(int(x2))
                    + " "
                    + str(int(y1))
                    + " "
                    + str(int(y2))
                )
                # ! debug ======================================================
                # # Create a Rectangle patch for debug
                # image = Image.fromarray(rgb_image.astype('uint8'), 'RGB')
                # fig, ax = plt.subplots()
                # ax.imshow(image)
<<<<<<< HEAD
                # rect = patches.Rectangle((x1, y2), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')

=======
                # rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')

>>>>>>> 5d3f437 (cleanup the visualisation for debug)
                # # Add the patch to the Axes
                # ax.add_patch(rect)
                # plt.show()
                # ! ============================================================

            else:
                self.logger.info("No detection")
                return None

        label = "bar-holder-stripped-bi-v3"
        object_data = ObjectData(label=label, bbox_modal=np.array([x1, y1, x2, y2]))
        self.logger.info("Data: " + str(object_data))

        object_data = [object_data]

        detections = make_detections_from_object_data(object_data).to(self._device)

        # ==============================================================================
        # self.logger.info(detections)

        # temp replacement =====================================================
        # x1 = int(392)
        # y1 = int(217)
        # x2 = int(782)
        # y2 = int(434)
        # label ='bar-holder-stripped-bi-v3' # ! MUST be the name of the mesh

        # object_data = ObjectData(label=label, bbox_modal=np.array([x1,y1,x2,y2]))
        # object_data = [object_data]
        # detections = make_detections_from_object_data(object_data).to(self._device)

        # end of temp replacement =============================================

        if detections is None:
            return None
        # TODO change to raise error

        t2 = time.perf_counter()
        timings["detections"] = t2 - t1

        if len(detections.infos) == 0:  # ? redundant ?
            return None

        observation.to(self._device)

        data_TCO_final, extra_data = self.pose_estimator.run_inference_pipeline(
            observation,
            detections=detections,
            **self.model_info["inference_parameters"],  #! change this
        )

        object_predictions = data_TCO_final.cpu()  # ? not sure why there is a .cpu here

        t3 = time.perf_counter()
        timings["single_view"] = t3 - t2

        object_predictions.cpu()
        timings["total"] = time.perf_counter() - t1

        self.logger.info("Inference results")
        self.logger.info(str(object_predictions.infos.to_string()))

        object_predictions.infos.rename(
            columns={"pose_score": "score"}, inplace=True
        )  # todo: find a better way to do it
        object_predictions.infos["score"] = object_predictions.infos["score"].astype(
            float
        )

        return {
            "infos": object_predictions.infos,
            "poses": object_predictions.poses,
            "bboxes": detections.tensors["bboxes"].int().cpu(),
            "timings": timings,
        }

    def yolo_detector(self, yolo_model, color_image) -> DetectionsType:
        """
        TODO
        """
        # yolo_model_path = "/docker_files/happypose_ros_data/yolo-checkpoints/yolo11n.pt" #"/docker_files/happypose_ros_data/yolo-checkpoints/bar-holder-stripped-bi-v2.pt"
        # yolo_model = YOLO(yolo_model_path)
        yolo_results = yolo_model(color_image, stream=True)

        for r in yolo_results:
            boxes = r.boxes
            min_confidence = 0
            box_w_max_conf = 0
            self.logger.info("boxes len:" + str(len(boxes)))

            if len(boxes) > 0:
                # self.logger.info("yolo found something")
                for box in boxes:
                    confidence = math.ceil((box.conf[0] * 100)) / 100

                    if confidence > min_confidence:
                        box_w_max_conf = box
                        min_confidence = confidence

                    # bounding box coordinates
                    x1, y1, x2, y2 = box_w_max_conf.xyxy[0]
                    x1, y1, x2, y2 = (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    )  # convert to int values
                    # Create a Rectangle patch
                    rect = patches.Rectangle(
                        (x1, y2),
                        x2 - x1,
                        y2 - y1,
                        linewidth=1,
                        edgecolor="r",
                        facecolor="none",
                    )

                    # Add the patch to the Axes
                    color_image.add_patch(rect)
                    plt.imshow(color_image)
                    plt.show()

            else:
                self.logger.info("No detection")
                return None

        label = "bar-holder"
        object_data = ObjectData(label=label, bbox_modal=np.array([x1, y1, x2, y2]))
        self.logger.info("Data: " + str(object_data))

        object_data = [object_data]

        detections = make_detections_from_object_data(object_data).to(self._device)
        return detections
