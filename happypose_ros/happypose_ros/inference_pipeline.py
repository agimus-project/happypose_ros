from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.inference.utils import filter_detections

from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper,
)

from happypose.toolbox.utils.logging import get_logger

logger = get_logger(__name__)


class HappyposePipeline:
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
            if self._inference_args["labels_to_keep"]
            else None
        )

    def __call__(self, observation: ObservationTensor) -> dict:
        try:
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

            final_preds, _ = self._wrapper.pose_predictor.run_inference_pipeline(
                observation,
                detections=detections,
                run_detector=False,
                data_TCO_init=None,
                **self._inference_args["pose_estimator"],
            )

            final_preds.cpu()
            return {
                "infos": final_preds.infos,
                "poses": final_preds.poses,
                "bboxes": detections.tensors["bboxes"].int().cpu(),
            }
        except Exception as e:
            logger.error(f"Error: {str(e)}")
