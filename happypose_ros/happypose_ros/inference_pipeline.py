from happypose.toolbox.inference.types import ObservationTensor


from happypose.pose_estimators.cosypose.cosypose.utils.cosypose_wrapper import (
    CosyPoseWrapper,
)


class HappyposePipeline:
    def __init__(self, params: dict) -> None:
        super().__init__()
        self._params = params
        self._device = self._params["device"]

        # Currently only cosypose is supported
        self._wrapper = CosyPoseWrapper(
            dataset_name=self._params["cosypose"]["dataset_name"],
            **self._params["renderer"],
        )

        self._inference_args = self._params["cosypose"]["inference"]
        self._inference_args["labels_to_keep"] = (
            self._inference_args["labels_to_keep"]
            if self._inference_args["labels_to_keep"]
            else None
        )

    def __call__(self, observation: ObservationTensor) -> tuple:
        final_preds, _ = self._wrapper.pose_predictor.run_inference_pipeline(
            observation,
            detections=None,
            run_detector=True,
            data_TCO_init=None,
            **self._inference_args,
        )
        return final_preds
