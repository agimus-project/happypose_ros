import time
from typing import Union

import torch
import numpy as np
import pandas as pd
import open3d as o3d

from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.inference.utils import filter_detections
from happypose.toolbox.datasets.object_dataset import RigidObjectDataset
from happypose.toolbox.renderer.panda3d_batch_renderer import Panda3dBatchRenderer

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

from happypose_ros.detector_utils import get_multicrop_detections
from happypose_ros.utils import ObservationMixedTensor
from happypose_ros.icp_utils import create_o3d_poincloud_from_depth, orient_normals_toward_camera, crop_pcd_sphere, ICPConvergeCriteria, icp_registration_o3d
from happypose_ros.icp_utils import extract_np_from_renderings, render_ts, get_panda3d_ambient

class HappyPosePipeline:
    """Object wrapping HappyPose pipeline extracting its calls from the main ROS node."""

    def __init__(self, params: dict) -> None:
        """Creates HappyPosePipeline object and starts loading Torch models to the memory.

        :param params: Parameters used to initialize the HappyPose pipeline.
        :type params: dict
        """
        super().__init__()
        self._params = params
        self._device = self._params["device"]

        # Currently only cosypose is supported
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

    def get_dataset(self) -> RigidObjectDataset:
        """Returns rigid object dataset used by HappyPose pose estimator

        :return: Dataset used by HappyPose pose estimator
        :rtype: RigidObjectDataset
        """
        dataset = self._wrapper.object_dataset
        if self._inference_args["labels_to_keep"] is None:
            return dataset
        return dataset.filter_objects(self._inference_args["labels_to_keep"])

    def __call__(self, observation: ObservationMixedTensor) -> Union[None, dict]:
        """Performs sequence of actions to estimate pose and optionally merge
        multiview results.

        :param observation: Tensor containing camera information and incoming images.
        :type observation: TODO
        :return: Dictionary with final detections. If pipeline failed or nothing
            was detected None is returned
        :rtype: Union[None, dict]
        """
        timings = {}
        t1 = time.perf_counter()
        # if color and depth are not aligned, create happypose observation only with color images
        obs_happy = ObservationTensor.from_torch_batched(
            rgb=observation.rgb, 
            depth=observation.depth if self._params["aligned_depth"] else None, 
            K=observation.K_color
        )
        obs_happy.to(self._device)

        if self._inference_args["use_multicrop_detector"]:
            detections = get_multicrop_detections(
                self._wrapper.pose_predictor.detector_model,
                obs_happy.images,
                obs_happy.K,
                self._device,
                tile_detection_scale=self._inference_args["tile_detection_scale"],
                detector_args=self._inference_args["detector"],
            )
        else:
            detections = self._wrapper.pose_predictor.detector_model.get_detections(
                obs_happy,
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
            obs_happy,
            detections=detections,
            run_detector=False,
            data_TCO_init=None,
            **self._inference_args["pose_estimator"],
        )
        t3 = time.perf_counter()
        timings["single_view"] = t3 - t2


        # if depth refinement is enabled and depth and color are aligned, 
        # use the happypose depth refiner 
        if self._params["use_depth"]:
            if self._params["aligned_depth"]:
                object_predictions, extra_data_depth_ref = (
                    self._wrapper.depth_refiner.refine_poses(
                        predictions=cosypose_predictions,
                        depth=obs_happy.depth,
                        K=obs_happy.K,
                        **self._inference_args[
                            self._params["cosypose"]["depth_refiner_type"]
                        ],
                    )
                )

                # Select only valid ICP results (retval of value 0)
                valid_icp_ids = np.logical_not(extra_data_depth_ref["retvals_icp"])
                object_predictions = object_predictions[valid_icp_ids]
            else:
                po3d = self._params["cosypose"]["icp_open3d"]
                voxel_size = po3d["voxel_size"]
                mesh_radius = 0.12  # TODO: needed for all objects
                dist_threshold = voxel_size * po3d["dist_thresh_factor"]

                # loop over camera views
                renderer: Panda3dBatchRenderer = self._wrapper.pose_predictor.refiner_model.renderer
                light_datas = [get_panda3d_ambient()]

                object_predictions = cosypose_predictions.clone().cpu()
                for view_id in range(obs_happy.batch_size):
                    # Get camera extrinsics for this particular camera/view
                    T_dc = observation.T_depth_color[view_id].float()
                    T_cd = T_dc.inverse().float()

                    # Create point cloud from depth image
                    depth_meas = observation.depth[view_id].cpu().numpy().squeeze(0)
                    K_depth = observation.K_depth[view_id].cpu().numpy()

                    pcd_ct = create_o3d_poincloud_from_depth(depth_meas, K_depth)
                    pcd_ct.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(3.0*voxel_size, 40))
                    pcd_ct = pcd_ct.voxel_down_sample(voxel_size=voxel_size)
                    h, w = depth_meas.shape

                    # loop over detections in the current view
                    for det_id in range(len(object_predictions.infos)):
                        if object_predictions.infos.iloc[det_id]["batch_im_id"] != view_id:
                            continue
                        T_co_init = object_predictions.poses[det_id]
                        T_do_init = T_dc @ T_co_init 
                        label = object_predictions.infos.label.iloc[det_id]
                        renderings = renderer.render([label], render_ts(T_do_init), render_ts(K_depth), [light_datas], (h, w), render_depth=True)
                        ren = extract_np_from_renderings(renderings, 0, ["depth"])
                        pcd_cp = create_o3d_poincloud_from_depth(ren["depth"], K_depth)
                        pcd_cp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(3.0*voxel_size, 40))
                        pcd_cp = orient_normals_toward_camera(pcd_cp)
                        pcd_cp = pcd_cp.voxel_down_sample(voxel_size=voxel_size)

                        if po3d["crop"]:
                            pcd_ct_crop = crop_pcd_sphere(pcd_ct, center=T_do_init[:3,3], radius=mesh_radius, margin=po3d["margin_sphere_crop"])
                            icp_res_ct_cp = icp_registration_o3d(pcd_cp, pcd_ct_crop, np.eye(4), dist_threshold, po3d["icp_method"], ICPConvergeCriteria())
                        else:
                            icp_res_ct_cp = icp_registration_o3d(pcd_cp, pcd_ct, np.eye(4), dist_threshold, po3d["icp_method"], ICPConvergeCriteria())

                        T_ct_cp_icp = icp_res_ct_cp.transformation

                        T_do_ref = torch.from_numpy(T_ct_cp_icp.copy()).float() @ T_do_init
                        object_predictions.poses[det_id] = T_cd @ T_do_ref

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
            K=obs_happy.K,
            infos=pd.DataFrame({"view_id": np.arange(obs_happy.batch_size)}),
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
