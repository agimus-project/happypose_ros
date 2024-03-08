import torch
import pathlib
import yaml
from typing import Any, Union

from rclpy.node import Node
from rclpy.parameter import Parameter

from happypose.pose_estimators.cosypose.cosypose.integrated.detector import Detector
from happypose.pose_estimators.cosypose.cosypose.integrated.pose_estimator import (
    PoseEstimator,
)
from happypose.pose_estimators.cosypose.cosypose.training.detector_models_cfg import (
    check_update_config as check_update_config_detector,
    create_model_detector,
)

from happypose.pose_estimators.cosypose.cosypose.training.pose_models_cfg import (
    load_model_cosypose,
)

from happypose.toolbox.datasets.bop_object_datasets import BOPObjectDataset
from happypose.toolbox.lib3d.rigid_mesh_database import MeshDataBase
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer


class CosyPoseLoader:
    @staticmethod
    def ros_param_path(node: Node, param_name: str) -> pathlib.Path:
        path = node.get_parameter(param_name).get_parameter_value().string_value
        path = pathlib.Path(path)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist!")
        return path

    @staticmethod
    def _load_param_dict(
        node: Node, ns: str, params_definition: tuple[str, Parameter.Type, Any]
    ) -> dict[str, Union[float, list[str], int]]:
        params_definition = [
            pd for pd in params_definition if node.has_parameter(pd[0])
        ]
        node.declare_parameters(ns, params_definition)
        param_vals = node.get_parameters([ns + p[0] for p in params_definition])
        param_dict = {
            definition[0]: value
            for definition, value in zip(params_definition, param_vals)
        }
        return param_dict

    @staticmethod
    def _load_detector(node: Node, device: str) -> Detector:
        ns = "detector"
        node.declare_parameter(ns + "/dataset")
        node.declare_parameter(ns + "/config_path")

        dataset = node.get_parameter(ns + "/dataset").get_parameter_value().string_value

        config_path = CosyPoseLoader.ros_param_path(node, ns + "/config_path")

        # TODO discuss exposing it as ROS parameters
        cfg = check_update_config_detector(
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
        model.config = cfg
        return Detector(model, dataset)

    @staticmethod
    def load_pose_estimator(node: Node, device: str) -> PoseEstimator:
        ns = "pose_estimator"
        node.declare_parameter(ns + "/dataset_path")
        node.declare_parameter(ns + "/label_format")
        node.declare_parameter(ns + "/batch_renderer/n_workers", 1)
        node.declare_parameter(ns + "/coarse/config_path")
        node.declare_parameter(ns + "/refiner/config_path")

        dataset_path = CosyPoseLoader.ros_param_path(node, ns + "/dataset_path")
        coarse_config_path = CosyPoseLoader.ros_param_path(
            node, ns + "/coarse/config_path"
        )
        refiner_config_path = CosyPoseLoader.ros_param_path(
            node, ns + "/refiner/config_path"
        )

        label_format = (
            node.get_parameter(ns + "/label_format").get_parameter_value().string_value
        )

        object_dataset = BOPObjectDataset(
            dataset_path,
            label_format=label_format,
        )

        ns = "renderer"
        params_definition = [
            ("n_workers", Parameter.Type.INTEGER, 8),
            ("preload_cache", Parameter.Type.BOOL, True),
            ("split_objects", Parameter.Type.BOOL, False),
        ]
        renderer_params = CosyPoseLoader._load_param_dict(node, ns, params_definition)
        renderer = Panda3dBatchRenderer(object_dataset, **renderer_params)

        mesh_db = MeshDataBase.from_object_ds(object_dataset)
        mesh_db_batched = mesh_db.batched().to(device)
        kwargs = {"renderer": renderer, "mesh_db": mesh_db_batched, "device": device}
        coarse_model = load_model_cosypose(coarse_config_path, **kwargs)
        refiner_model = load_model_cosypose(refiner_config_path, **kwargs)

        return PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=CosyPoseLoader._load_detector(node, device),
            bsz_objects=1,
            bsz_images=1,
        )

    @staticmethod
    def load_inference_params(node: Node) -> dict[str, Union[float, list[str], int]]:
        ns = "inference"
        params_definition = [
            ("n_refiner_iterations", Parameter.Type.INTEGER, 1),
            ("n_coarse_iterations", Parameter.Type.INTEGER, 1),
            ("bsz_images", Parameter.Type.INTEGER, None),
            ("bsz_objects", Parameter.Type.INTEGER, None),
            ("detection_th", Parameter.Type.DOUBLE, 0.7),
            ("mask_th", Parameter.Type.DOUBLE, 0.8),
            ("labels_to_keep", Parameter.Type.STRING_ARRAY, []),
        ]
        return CosyPoseLoader._load_param_dict(node, ns, params_definition)
