import numpy as np
import pandas as pd
from typing import Union

from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.inference.utils import filter_detections

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


class HappyPosePipeline:
    def __init__(self, params: dict) -> None:
        super().__init__()
        self._params = params
        self._device = self._params["device"]

        # Currently only cosypose is supported
        self._wrapper = CosyPoseWrapper(
            dataset_name=self._params["cosypose"]["dataset_name"],
            **self._params["cosypose"]["renderer"],
        )

        self._inference_args = self._params["cosypose"]["inference"]
        self._inference_args["labels_to_keep"] = (
            self._inference_args["labels_to_keep"]
            if self._inference_args["labels_to_keep"] != [""]
            else None
        )

        self._multiview = len(self._params["camera_names"]) > 1
        if self._multiview:
            dir = self._wrapper.object_dataset.ds_dir.as_posix()
            label_format = self._params["cosypose"]["dataset_name"] + "-{label}"
            object_ds = BOPObjectDataset(dir, label_format)
            mesh_db = MeshDataBase.from_object_ds(object_ds)
            self._mv_predictor = MultiviewScenePredictor(mesh_db)

    def __call__(self, observation: ObservationTensor) -> Union[None, dict]:
        detections = self._wrapper.pose_predictor.detector_model.get_detections(
            observation,
            output_masks=False,
            **self._inference_args["detector"],
        )

        detections = filter_detections(
            detections, self._inference_args["labels_to_keep"]
        )

        if len(detections.infos) == 0:
            return None

        object_predictions, _ = self._wrapper.pose_predictor.run_inference_pipeline(
            observation,
            detections=detections,
            run_detector=False,
            data_TCO_init=None,
            **self._inference_args["pose_estimator"],
        )

        if not self._multiview:
            object_predictions.cpu()
            return {
                "infos": object_predictions.infos,
                "poses": object_predictions.poses,
                "bboxes": detections.tensors["bboxes"].int().cpu(),
            }

        object_predictions.infos = object_predictions.infos.rename(
            columns={"batch_im_id": "view_id"}
        )
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
        cameras_pred.TWC = cameras_pred.TWC[cameras_pred.infos.index]
        cameras_pred.K = cameras_pred.TWC[cameras_pred.infos.index]

        # Normalize score to range 0 - 1
        predictions["scene/objects"].infos["score"] /= predictions[
            "scene/objects"
        ].infos["n_cand"]

        return {
            "infos": object_predictions.infos,
            "poses": object_predictions.TWO,
            "bboxes": None,
            "camera_infos": cameras_pred.infos,
            "camera_poses": cameras_pred.TWC,
            "camera_K": cameras_pred.K,
        }
