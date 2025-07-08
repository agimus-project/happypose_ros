import time
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from abc import ABC, abstractmethod
from typing import final

from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.inference.utils import (
    filter_detections,
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

from happypose_ros.megapose_detector import Detector


class InferencePipeline(ABC):
    """Abstract class from which the CosyPosePipeline and MegaPosePipeline inherit."""

    def __init__(self, params: dict) -> None:
        """Creates the pipeline object and loads model to memory."""
        return 0

    @final
    def update_params(self, params: dict) -> None:
        """Updates parameters used by the HappyPose.

        :param params: Parameters used to initialize the HappyPose pipeline.
            On runtime to update inference parameters.
        :type params: dict
        """
        self._inference_args = params["cosypose"]["inference"]
        self._inference_args["labels_to_keep"] = (
            self._inference_args["labels_to_keep"]
            if self._inference_args["labels_to_keep"] != [""]
            else None
        )

    @abstractmethod
    def get_dataset(self) -> RigidObjectDataset:
        """Returns rigid object dataset used by the pose estimator

        :return: Dataset used by HappyPose pose estimator
        :rtype: RigidObjectDataset"""
        pass

    @abstractmethod
    def __call__(self, observation: ObservationTensor) -> Union[None, dict]:
        """Performs sequence of actions to estimate pose of the detected object(s)
        and (optional) multiple cameras.

        :param observation: Tensor containing camera information and incoming images.
        :type observation: happypose.toolbox.inference.types.ObservationTensor
        :return: Dictionary with final detections. If pipeline failed or nothing
            was detected None is returned
        :rtype: Union[None, dict]
        """
        pass


class CosyPosePipeline(InferencePipeline):
    """Object wrapping HappyPose pipeline extracting its calls from the main ROS node."""

    def __init__(self, params: dict) -> None:
        """Creates HappyPosePipeline object and starts loading Torch models to the memory.

        :param params: Parameters used to initialize the HappyPose pipeline.
        :type params: dict
        """
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
        super().__init__(params)
        self._params = params
        self._device = self._params["device"]

        self.update_params(self._params)

        object_dataset = self.get_dataset()
        self.model_info = NAMED_MODELS[self._params["megapose"]["model_name"]]
        self.pose_estimator = load_named_model(
            self._params["megapose"]["model_name"], object_dataset
        ).to(self._device)
        self.pose_estimator._SO3_grid = self.pose_estimator._SO3_grid[::8]

        # load yolo model
        self.detector = Detector(self._params)

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
        """Performs sequence of actions to estimate pose.

        :param observation: Tensor containing camera information and incoming images.
        :type observation: happypose.toolbox.inference.types.ObservationTensor
        :return: Dictionary with final detections. If pipeline failed or nothing
            was detected None is returned
        :rtype: Union[None, dict]
        """

        timings = {}
        t1 = time.perf_counter()

        detections = self.detector.run(observation)

        if detections is None:
            return None
        # TODO change to raise error

        t2 = time.perf_counter()
        timings["detections"] = t2 - t1

        if len(detections.infos) == 0:
            return None

        observation.to(self._device)

        data_TCO_final, extra_data = self.pose_estimator.run_inference_pipeline(
            observation,
            detections=detections,
            **self.model_info["inference_parameters"],
        )

        object_predictions = data_TCO_final.cpu()

        t3 = time.perf_counter()
        timings["single_view"] = t3 - t2

        object_predictions.cpu()
        timings["total"] = time.perf_counter() - t1

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
