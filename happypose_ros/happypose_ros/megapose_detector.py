from ultralytics import YOLO
import numpy as np
import math

from happypose.toolbox.inference.types import ObservationTensor, DetectionsType
from happypose.toolbox.datasets.scene_dataset import ObjectData
from happypose.toolbox.inference.utils import (
    make_detections_from_object_data,
)

from rclpy.impl import rcutils_logger


class Detector():
    def __init__(self, params ) -> None:
            """
                TODO
            """
            super().__init__()
            self.logger = rcutils_logger.RcutilsLogger(name="Megapose detector")

            super().__init__()
            self._params = params
            self._device = self._params["device"]

            self.detector_path = "/docker_files/happypose_ros_data/yolo-checkpoints/yolo11n.pt"
            self.label = "bar-holder-stripped-bi-v3"

            yolo_model_path = self.detector_path
            self.yolo_model = YOLO(yolo_model_path)
            

    def run(self, observation: ObservationTensor) -> DetectionsType :
        """
            TODO
        """
        image = self.convert_obstensor_to_image(observation)
        results = self.yolo_model(image, stream=True)

        for r in results:
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
                # rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                
                # # Add the patch to the Axes
                # ax.add_patch(rect)
                # plt.show()
                # ! ============================================================

            else:
                self.logger.info("No detection")
                return None

        object_data = ObjectData(label=self.label, bbox_modal=np.array([x1, y1, x2, y2]))
        self.logger.info("Data: " + str(object_data))

        object_data = [object_data]

        detections = make_detections_from_object_data(object_data).to(self._device)

        return detections
    
    def convert_obstensor_to_image(self, observation: ObservationTensor) -> np.array: # to utils?
        """
            TODO
        """
        rgb_tensor = observation.images[
            :, 0:3
        ].cpu() 

        rgb_image = rgb_tensor.numpy()

        # remove extra dimension
        rgb_image = rgb_image[0, :, :, :] 

        # rearrange axis to have RGB image
        rgb_image = np.moveaxis(rgb_image, [0, 1], [2, 0])

        # conversion from float32 to unit8 required for YOLO
        rgb_image *= 255
        rgb_image = rgb_image.astype(np.uint8)

        return rgb_image