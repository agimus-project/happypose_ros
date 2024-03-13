import torch
import yaml

from happypose.pose_estimators.cosypose.cosypose.integrated.detector import Detector
from happypose.toolbox.inference.types import ObservationTensor
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import (
    check_update_config as check_update_params_detector,
    create_model_detector,
)
from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    load_model_cosypose,
)
from happypose.toolbox.datasets.bop_object_datasets import BOPObjectDataset
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer

from happypose_ros.utils import unwrap_ros_path
from happypose_ros.happypose_ros_parameters import happypose_ros


class CosyPoseLoader:
    @staticmethod
    def _load_detector(params, device: str = "cpu") -> Detector:
        config_path = unwrap_ros_path(params.config_path)
        cfg = check_update_params_detector(
            yaml.load(
                (config_path / "config.yaml").read_text(), Loader=yaml.UnsafeLoader
            ),
        )
        label_to_category_id = cfg.label_to_category_id
        ckpt = torch.load(config_path / "checkpoint.pth.tar", map_location=device)[
            "state_dict"
        ]

        model = create_model_detector(cfg, len(label_to_category_id))
        model.load_state_dict(ckpt)
        model = model.to(device).eval()
        model.cfg = cfg
        model.params = cfg
        return Detector(model, params.dataset_name)

    @staticmethod
    def load_pose_estimator(params, device: str) -> PoseEstimator:
        kwargs = dict(params.object_dataset.__dict__)
        kwargs["ds_dir"] = unwrap_ros_path(params.object_dataset.config_path)
        del kwargs["config_path"]
        object_dataset = BOPObjectDataset(**kwargs)

        renderer = Panda3dBatchRenderer(
            object_dataset, **dict(params.pose_estimator.renderer.__dict__)
        )

        mesh_db = MeshDataBase.from_object_ds(object_dataset)
        mesh_db_batched = mesh_db.batched().to(device)
        kwargs = {
            "renderer": renderer,
            "mesh_db_batched": mesh_db_batched,
            "device": device,
        }
        coarse_path = unwrap_ros_path(params.pose_estimator.coarse.config_path)
        coarse_model = load_model_cosypose(coarse_path, **kwargs)

        refiner_path = unwrap_ros_path(params.pose_estimator.refiner.config_path)
        refiner_model = load_model_cosypose(refiner_path, **kwargs)

        return PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=CosyPoseLoader._load_detector(params.detector, device),
            # TODO get more information what is this
            bsz_objects=1,
            bsz_images=1,
        )


class HappyposePipeline:
    def __init__(self, params: happypose_ros.Params) -> None:
        super().__init__()
        self._device = params.device
        if params.pose_estimator_type == "cosypose":
            loader = CosyPoseLoader
            self._params = params.cosypose

        self._pose_estimator = loader.load_pose_estimator(self._params, self._device)

    def __call__(self, observation: ObservationTensor) -> tuple:
        kwargs = dict(self._params.inference.__dict__)
        preds, preds_extra = self._pose_estimator.run_inference_pipeline(
            observation=observation, run_detector=True, **kwargs
        )
        return preds, preds_extra
