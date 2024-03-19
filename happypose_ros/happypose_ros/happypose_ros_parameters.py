# flake8: noqa

# auto-generated DO NOT EDIT

from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
from rcl_interfaces.msg import FloatingPointRange, IntegerRange
from rclpy.clock import Clock
from rclpy.exceptions import InvalidParameterValueException
from rclpy.time import Time
import copy
import rclpy
from generate_parameter_library_py.python_validators import ParameterValidators


class happypose_ros:
    class Params:
        # for detecting if the parameter struct has been updated
        stamp_ = Time()

        pose_estimator_type = "cosypose"
        device = "cpu"

        class __Renderer:
            renderer_type = "panda3d"
            n_workers = 8
            gpu_renderer = True

        renderer = __Renderer()

        class __Cosypose:
            dataset_name = ""

            class __Inference:
                n_refiner_iterations = 1
                n_coarse_iterations = 1
                detection_th = 0.7
                mask_th = 0.8
                labels_to_keep = []

            inference = __Inference()

        cosypose = __Cosypose()

        class __Cameras:
            timeout = 0.0
            n_min_cameras = 1
            names = None

            class __MapNames:
                compressed = False
                image_topic = ""
                info_topic = ""
                k_matrix = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            __map_type = __MapNames

            def add_entry(self, name):
                if not hasattr(self, name):
                    setattr(self, name, self.__map_type())
                return getattr(self, name)

            def get_entry(self, name):
                return getattr(self, name)

        cameras = __Cameras()

    class ParamListener:
        def __init__(self, node, prefix=""):
            self.prefix_ = prefix
            self.params_ = happypose_ros.Params()
            self.node_ = node
            self.logger_ = rclpy.logging.get_logger("happypose_ros." + prefix)

            self.declare_params()

            self.node_.add_on_set_parameters_callback(self.update)
            self.clock_ = Clock()

        def get_params(self):
            tmp = self.params_.stamp_
            self.params_.stamp_ = None
            paramCopy = copy.deepcopy(self.params_)
            paramCopy.stamp_ = tmp
            self.params_.stamp_ = tmp
            return paramCopy

        def is_old(self, other_param):
            return self.params_.stamp_ != other_param.stamp_

        def refresh_dynamic_parameters(self):
            updated_params = self.get_params()
            # TODO remove any destroyed dynamic parameters

            # declare any new dynamic parameters

            for value_1 in updated_params.cameras.names:
                updated_params.cameras.add_entry(value_1)
                entry = updated_params.cameras.get_entry(value_1)
                param_name = f"{self.prefix_}cameras.{value_1}.compressed"
                if not self.node_.has_parameter(self.prefix_ + param_name):
                    descriptor = ParameterDescriptor(
                        description="Expect compressed messages from given camera.",
                        read_only=False,
                    )
                    parameter = entry.compressed
                    self.node_.declare_parameter(param_name, parameter, descriptor)
                param = self.node_.get_parameter(param_name)
                self.logger_.debug(
                    param.name + ": " + param.type_.name + " = " + str(param.value)
                )
                entry.compressed = param.value

            for value_1 in updated_params.cameras.names:
                updated_params.cameras.add_entry(value_1)
                entry = updated_params.cameras.get_entry(value_1)
                param_name = f"{self.prefix_}cameras.{value_1}.image_topic"
                if not self.node_.has_parameter(self.prefix_ + param_name):
                    descriptor = ParameterDescriptor(
                        description="Topic name to subscribe for images. If empty, defaults to '<camera_name>_raw' for not compressed image and '<camera_name>/compressed' for compressed images.",
                        read_only=False,
                    )
                    parameter = entry.image_topic
                    self.node_.declare_parameter(param_name, parameter, descriptor)
                param = self.node_.get_parameter(param_name)
                self.logger_.debug(
                    param.name + ": " + param.type_.name + " = " + str(param.value)
                )
                entry.image_topic = param.value

            for value_1 in updated_params.cameras.names:
                updated_params.cameras.add_entry(value_1)
                entry = updated_params.cameras.get_entry(value_1)
                param_name = f"{self.prefix_}cameras.{value_1}.info_topic"
                if not self.node_.has_parameter(self.prefix_ + param_name):
                    descriptor = ParameterDescriptor(
                        description="Topic name to subscribe for camera info. If empty defaults to '<camera_name>/info'.",
                        read_only=False,
                    )
                    parameter = entry.info_topic
                    self.node_.declare_parameter(param_name, parameter, descriptor)
                param = self.node_.get_parameter(param_name)
                self.logger_.debug(
                    param.name + ": " + param.type_.name + " = " + str(param.value)
                )
                entry.info_topic = param.value

            for value_1 in updated_params.cameras.names:
                updated_params.cameras.add_entry(value_1)
                entry = updated_params.cameras.get_entry(value_1)
                param_name = f"{self.prefix_}cameras.{value_1}.k_matrix"
                if not self.node_.has_parameter(self.prefix_ + param_name):
                    descriptor = ParameterDescriptor(
                        description="Camera intrinsic matrix. If not equal to all values of 0.0, overwrites values from info ROS topic.",
                        read_only=False,
                    )
                    descriptor.floating_point_range.append(FloatingPointRange())
                    descriptor.floating_point_range[-1].from_value = 0.0
                    descriptor.floating_point_range[-1].to_value = float("inf")
                    parameter = entry.k_matrix
                    self.node_.declare_parameter(param_name, parameter, descriptor)
                param = self.node_.get_parameter(param_name)
                self.logger_.debug(
                    param.name + ": " + param.type_.name + " = " + str(param.value)
                )
                validation_result = ParameterValidators.fixed_size(param, 9)
                if validation_result:
                    raise InvalidParameterValueException(
                        "cameras.__map_names.k_matrix",
                        param.value,
                        "Invalid value set during initialization for parameter cameras.__map_names.k_matrix: "
                        + validation_result,
                    )
                validation_result = ParameterValidators.lower_element_bounds(param, 0.0)
                if validation_result:
                    raise InvalidParameterValueException(
                        "cameras.__map_names.k_matrix",
                        param.value,
                        "Invalid value set during initialization for parameter cameras.__map_names.k_matrix: "
                        + validation_result,
                    )
                entry.k_matrix = param.value

        def update(self, parameters):
            updated_params = self.get_params()

            for param in parameters:
                if param.name == self.prefix_ + "pose_estimator_type":
                    validation_result = ParameterValidators.one_of(param, ["cosypose"])
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.pose_estimator_type = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "device":
                    validation_result = ParameterValidators.one_of(
                        param, ["cpu", "cuda:0"]
                    )
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.device = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "renderer.renderer_type":
                    validation_result = ParameterValidators.one_of(
                        param, ["panda3d", "bullet"]
                    )
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.renderer.renderer_type = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "renderer.n_workers":
                    validation_result = ParameterValidators.gt_eq(param, 1)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.renderer.n_workers = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "renderer.gpu_renderer":
                    updated_params.renderer.gpu_renderer = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "cosypose.dataset_name":
                    validation_result = ParameterValidators.one_of(
                        param, ["hope", "tless", "ycbv"]
                    )
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cosypose.dataset_name = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if (
                    param.name
                    == self.prefix_ + "cosypose.inference.n_refiner_iterations"
                ):
                    validation_result = ParameterValidators.gt_eq(param, 1)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cosypose.inference.n_refiner_iterations = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if (
                    param.name
                    == self.prefix_ + "cosypose.inference.n_coarse_iterations"
                ):
                    validation_result = ParameterValidators.gt_eq(param, 1)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cosypose.inference.n_coarse_iterations = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "cosypose.inference.detection_th":
                    validation_result = ParameterValidators.bounds(param, 0.0, 1.0)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cosypose.inference.detection_th = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "cosypose.inference.mask_th":
                    validation_result = ParameterValidators.bounds(param, 0.0, 1.0)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cosypose.inference.mask_th = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "cosypose.inference.labels_to_keep":
                    validation_result = ParameterValidators.unique(param)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cosypose.inference.labels_to_keep = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "cameras.timeout":
                    validation_result = ParameterValidators.gt_eq(param, 0.0)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cameras.timeout = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "cameras.n_min_cameras":
                    validation_result = ParameterValidators.gt_eq(param, 1)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cameras.n_min_cameras = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

                if param.name == self.prefix_ + "cameras.names":
                    validation_result = ParameterValidators.size_gt(param, 0)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    validation_result = ParameterValidators.unique(param)
                    if validation_result:
                        return SetParametersResult(
                            successful=False, reason=validation_result
                        )
                    updated_params.cameras.names = param.value
                    self.logger_.debug(
                        param.name + ": " + param.type_.name + " = " + str(param.value)
                    )

            # update dynamic parameters
            for param in parameters:
                for value_1 in updated_params.cameras.names:
                    param_name = f"{self.prefix_}cameras.{value_1}.compressed"
                    if param.name == param_name:
                        updated_params.cameras.get_entry(
                            value_1
                        ).compressed = param.value
                        self.logger_.debug(
                            param.name
                            + ": "
                            + param.type_.name
                            + " = "
                            + str(param.value)
                        )

                for value_1 in updated_params.cameras.names:
                    param_name = f"{self.prefix_}cameras.{value_1}.image_topic"
                    if param.name == param_name:
                        updated_params.cameras.get_entry(
                            value_1
                        ).image_topic = param.value
                        self.logger_.debug(
                            param.name
                            + ": "
                            + param.type_.name
                            + " = "
                            + str(param.value)
                        )

                for value_1 in updated_params.cameras.names:
                    param_name = f"{self.prefix_}cameras.{value_1}.info_topic"
                    if param.name == param_name:
                        updated_params.cameras.get_entry(
                            value_1
                        ).info_topic = param.value
                        self.logger_.debug(
                            param.name
                            + ": "
                            + param.type_.name
                            + " = "
                            + str(param.value)
                        )

                for value_1 in updated_params.cameras.names:
                    param_name = f"{self.prefix_}cameras.{value_1}.k_matrix"
                    if param.name == param_name:
                        validation_result = ParameterValidators.fixed_size(param, 9)
                        if validation_result:
                            return SetParametersResult(
                                successful=False, reason=validation_result
                            )
                        validation_result = ParameterValidators.lower_element_bounds(
                            param, 0.0
                        )
                        if validation_result:
                            return SetParametersResult(
                                successful=False, reason=validation_result
                            )
                        updated_params.cameras.get_entry(value_1).k_matrix = param.value
                        self.logger_.debug(
                            param.name
                            + ": "
                            + param.type_.name
                            + " = "
                            + str(param.value)
                        )

            updated_params.stamp_ = self.clock_.now()
            self.update_internal_params(updated_params)
            return SetParametersResult(successful=True)

        def update_internal_params(self, updated_params):
            self.params_ = updated_params

        def declare_params(self):
            updated_params = self.get_params()
            # declare all parameters and give default values to non-required ones
            if not self.node_.has_parameter(self.prefix_ + "pose_estimator_type"):
                descriptor = ParameterDescriptor(
                    description="Specifies which pose estimator to use in the pipeline.",
                    read_only=False,
                )
                parameter = updated_params.pose_estimator_type
                self.node_.declare_parameter(
                    self.prefix_ + "pose_estimator_type", parameter, descriptor
                )

            if not self.node_.has_parameter(self.prefix_ + "device"):
                descriptor = ParameterDescriptor(
                    description="Device to which the models will be loaded.",
                    read_only=False,
                )
                parameter = updated_params.device
                self.node_.declare_parameter(
                    self.prefix_ + "device", parameter, descriptor
                )

            if not self.node_.has_parameter(self.prefix_ + "renderer.renderer_type"):
                descriptor = ParameterDescriptor(
                    description="Specifies which renderer to use in the pipeline.",
                    read_only=False,
                )
                parameter = updated_params.renderer.renderer_type
                self.node_.declare_parameter(
                    self.prefix_ + "renderer.renderer_type", parameter, descriptor
                )

            if not self.node_.has_parameter(self.prefix_ + "renderer.n_workers"):
                descriptor = ParameterDescriptor(
                    description="Number of CPU cores to use during rendering.",
                    read_only=False,
                )
                descriptor.integer_range.append(IntegerRange())
                descriptor.integer_range[-1].from_value = 1
                descriptor.integer_range[-1].to_value = 2**31 - 1
                parameter = updated_params.renderer.n_workers
                self.node_.declare_parameter(
                    self.prefix_ + "renderer.n_workers", parameter, descriptor
                )

            if not self.node_.has_parameter(self.prefix_ + "renderer.gpu_renderer"):
                descriptor = ParameterDescriptor(
                    description="Render objects with a GPU", read_only=False
                )
                parameter = updated_params.renderer.gpu_renderer
                self.node_.declare_parameter(
                    self.prefix_ + "renderer.gpu_renderer", parameter, descriptor
                )

            if not self.node_.has_parameter(self.prefix_ + "cosypose.dataset_name"):
                descriptor = ParameterDescriptor(
                    description="Name of a dataset used during training.",
                    read_only=False,
                )
                parameter = updated_params.cosypose.dataset_name
                self.node_.declare_parameter(
                    self.prefix_ + "cosypose.dataset_name", parameter, descriptor
                )

            if not self.node_.has_parameter(
                self.prefix_ + "cosypose.inference.n_refiner_iterations"
            ):
                descriptor = ParameterDescriptor(
                    description="Number of iterations for the refiner.", read_only=False
                )
                descriptor.integer_range.append(IntegerRange())
                descriptor.integer_range[-1].from_value = 1
                descriptor.integer_range[-1].to_value = 2**31 - 1
                parameter = updated_params.cosypose.inference.n_refiner_iterations
                self.node_.declare_parameter(
                    self.prefix_ + "cosypose.inference.n_refiner_iterations",
                    parameter,
                    descriptor,
                )

            if not self.node_.has_parameter(
                self.prefix_ + "cosypose.inference.n_coarse_iterations"
            ):
                descriptor = ParameterDescriptor(
                    description="Number of iterations for the coarse estimate.",
                    read_only=False,
                )
                descriptor.integer_range.append(IntegerRange())
                descriptor.integer_range[-1].from_value = 1
                descriptor.integer_range[-1].to_value = 2**31 - 1
                parameter = updated_params.cosypose.inference.n_coarse_iterations
                self.node_.declare_parameter(
                    self.prefix_ + "cosypose.inference.n_coarse_iterations",
                    parameter,
                    descriptor,
                )

            if not self.node_.has_parameter(
                self.prefix_ + "cosypose.inference.detection_th"
            ):
                descriptor = ParameterDescriptor(
                    description="Detection threshold of an object used by detector.",
                    read_only=False,
                )
                descriptor.floating_point_range.append(FloatingPointRange())
                descriptor.floating_point_range[-1].from_value = 0.0
                descriptor.floating_point_range[-1].to_value = 1.0
                parameter = updated_params.cosypose.inference.detection_th
                self.node_.declare_parameter(
                    self.prefix_ + "cosypose.inference.detection_th",
                    parameter,
                    descriptor,
                )

            if not self.node_.has_parameter(
                self.prefix_ + "cosypose.inference.mask_th"
            ):
                descriptor = ParameterDescriptor(description="?", read_only=False)
                descriptor.floating_point_range.append(FloatingPointRange())
                descriptor.floating_point_range[-1].from_value = 0.0
                descriptor.floating_point_range[-1].to_value = 1.0
                parameter = updated_params.cosypose.inference.mask_th
                self.node_.declare_parameter(
                    self.prefix_ + "cosypose.inference.mask_th", parameter, descriptor
                )

            if not self.node_.has_parameter(
                self.prefix_ + "cosypose.inference.labels_to_keep"
            ):
                descriptor = ParameterDescriptor(
                    description="Labels of detected objects to keep.", read_only=False
                )
                parameter = updated_params.cosypose.inference.labels_to_keep
                self.node_.declare_parameter(
                    self.prefix_ + "cosypose.inference.labels_to_keep",
                    parameter,
                    descriptor,
                )

            if not self.node_.has_parameter(self.prefix_ + "cameras.timeout"):
                descriptor = ParameterDescriptor(
                    description="Timeout, after which frame from a camera is considered too old. Value '0.0' disables timeout.",
                    read_only=False,
                )
                descriptor.floating_point_range.append(FloatingPointRange())
                descriptor.floating_point_range[-1].from_value = 0.0
                descriptor.floating_point_range[-1].to_value = float("inf")
                parameter = updated_params.cameras.timeout
                self.node_.declare_parameter(
                    self.prefix_ + "cameras.timeout", parameter, descriptor
                )

            if not self.node_.has_parameter(self.prefix_ + "cameras.n_min_cameras"):
                descriptor = ParameterDescriptor(
                    description="Minimum number of cameras to consider during single view.",
                    read_only=False,
                )
                descriptor.integer_range.append(IntegerRange())
                descriptor.integer_range[-1].from_value = 1
                descriptor.integer_range[-1].to_value = 2**31 - 1
                parameter = updated_params.cameras.n_min_cameras
                self.node_.declare_parameter(
                    self.prefix_ + "cameras.n_min_cameras", parameter, descriptor
                )

            if not self.node_.has_parameter(self.prefix_ + "cameras.names"):
                descriptor = ParameterDescriptor(
                    description="List of names of cameras to subscribe.", read_only=True
                )
                parameter = rclpy.Parameter.Type.STRING_ARRAY
                self.node_.declare_parameter(
                    self.prefix_ + "cameras.names", parameter, descriptor
                )

            # TODO: need validation
            # get parameters and fill struct fields
            param = self.node_.get_parameter(self.prefix_ + "pose_estimator_type")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.one_of(param, ["cosypose"])
            if validation_result:
                raise InvalidParameterValueException(
                    "pose_estimator_type",
                    param.value,
                    "Invalid value set during initialization for parameter pose_estimator_type: "
                    + validation_result,
                )
            updated_params.pose_estimator_type = param.value
            param = self.node_.get_parameter(self.prefix_ + "device")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.one_of(param, ["cpu", "cuda:0"])
            if validation_result:
                raise InvalidParameterValueException(
                    "device",
                    param.value,
                    "Invalid value set during initialization for parameter device: "
                    + validation_result,
                )
            updated_params.device = param.value
            param = self.node_.get_parameter(self.prefix_ + "renderer.renderer_type")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.one_of(param, ["panda3d", "bullet"])
            if validation_result:
                raise InvalidParameterValueException(
                    "renderer.renderer_type",
                    param.value,
                    "Invalid value set during initialization for parameter renderer.renderer_type: "
                    + validation_result,
                )
            updated_params.renderer.renderer_type = param.value
            param = self.node_.get_parameter(self.prefix_ + "renderer.n_workers")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.gt_eq(param, 1)
            if validation_result:
                raise InvalidParameterValueException(
                    "renderer.n_workers",
                    param.value,
                    "Invalid value set during initialization for parameter renderer.n_workers: "
                    + validation_result,
                )
            updated_params.renderer.n_workers = param.value
            param = self.node_.get_parameter(self.prefix_ + "renderer.gpu_renderer")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            updated_params.renderer.gpu_renderer = param.value
            param = self.node_.get_parameter(self.prefix_ + "cosypose.dataset_name")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.one_of(
                param, ["hope", "tless", "ycbv"]
            )
            if validation_result:
                raise InvalidParameterValueException(
                    "cosypose.dataset_name",
                    param.value,
                    "Invalid value set during initialization for parameter cosypose.dataset_name: "
                    + validation_result,
                )
            updated_params.cosypose.dataset_name = param.value
            param = self.node_.get_parameter(
                self.prefix_ + "cosypose.inference.n_refiner_iterations"
            )
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.gt_eq(param, 1)
            if validation_result:
                raise InvalidParameterValueException(
                    "cosypose.inference.n_refiner_iterations",
                    param.value,
                    "Invalid value set during initialization for parameter cosypose.inference.n_refiner_iterations: "
                    + validation_result,
                )
            updated_params.cosypose.inference.n_refiner_iterations = param.value
            param = self.node_.get_parameter(
                self.prefix_ + "cosypose.inference.n_coarse_iterations"
            )
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.gt_eq(param, 1)
            if validation_result:
                raise InvalidParameterValueException(
                    "cosypose.inference.n_coarse_iterations",
                    param.value,
                    "Invalid value set during initialization for parameter cosypose.inference.n_coarse_iterations: "
                    + validation_result,
                )
            updated_params.cosypose.inference.n_coarse_iterations = param.value
            param = self.node_.get_parameter(
                self.prefix_ + "cosypose.inference.detection_th"
            )
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.bounds(param, 0.0, 1.0)
            if validation_result:
                raise InvalidParameterValueException(
                    "cosypose.inference.detection_th",
                    param.value,
                    "Invalid value set during initialization for parameter cosypose.inference.detection_th: "
                    + validation_result,
                )
            updated_params.cosypose.inference.detection_th = param.value
            param = self.node_.get_parameter(
                self.prefix_ + "cosypose.inference.mask_th"
            )
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.bounds(param, 0.0, 1.0)
            if validation_result:
                raise InvalidParameterValueException(
                    "cosypose.inference.mask_th",
                    param.value,
                    "Invalid value set during initialization for parameter cosypose.inference.mask_th: "
                    + validation_result,
                )
            updated_params.cosypose.inference.mask_th = param.value
            param = self.node_.get_parameter(
                self.prefix_ + "cosypose.inference.labels_to_keep"
            )
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.unique(param)
            if validation_result:
                raise InvalidParameterValueException(
                    "cosypose.inference.labels_to_keep",
                    param.value,
                    "Invalid value set during initialization for parameter cosypose.inference.labels_to_keep: "
                    + validation_result,
                )
            updated_params.cosypose.inference.labels_to_keep = param.value
            param = self.node_.get_parameter(self.prefix_ + "cameras.timeout")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.gt_eq(param, 0.0)
            if validation_result:
                raise InvalidParameterValueException(
                    "cameras.timeout",
                    param.value,
                    "Invalid value set during initialization for parameter cameras.timeout: "
                    + validation_result,
                )
            updated_params.cameras.timeout = param.value
            param = self.node_.get_parameter(self.prefix_ + "cameras.n_min_cameras")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.gt_eq(param, 1)
            if validation_result:
                raise InvalidParameterValueException(
                    "cameras.n_min_cameras",
                    param.value,
                    "Invalid value set during initialization for parameter cameras.n_min_cameras: "
                    + validation_result,
                )
            updated_params.cameras.n_min_cameras = param.value
            param = self.node_.get_parameter(self.prefix_ + "cameras.names")
            self.logger_.debug(
                param.name + ": " + param.type_.name + " = " + str(param.value)
            )
            validation_result = ParameterValidators.size_gt(param, 0)
            if validation_result:
                raise InvalidParameterValueException(
                    "cameras.names",
                    param.value,
                    "Invalid value set during initialization for parameter cameras.names: "
                    + validation_result,
                )
            validation_result = ParameterValidators.unique(param)
            if validation_result:
                raise InvalidParameterValueException(
                    "cameras.names",
                    param.value,
                    "Invalid value set during initialization for parameter cameras.names: "
                    + validation_result,
                )
            updated_params.cameras.names = param.value

            # declare and set all dynamic parameters

            for value_1 in updated_params.cameras.names:
                updated_params.cameras.add_entry(value_1)
                entry = updated_params.cameras.get_entry(value_1)
                param_name = f"{self.prefix_}cameras.{value_1}.compressed"
                if not self.node_.has_parameter(self.prefix_ + param_name):
                    descriptor = ParameterDescriptor(
                        description="Expect compressed messages from given camera.",
                        read_only=False,
                    )
                    parameter = entry.compressed
                    self.node_.declare_parameter(param_name, parameter, descriptor)
                param = self.node_.get_parameter(param_name)
                self.logger_.debug(
                    param.name + ": " + param.type_.name + " = " + str(param.value)
                )
                entry.compressed = param.value

            for value_1 in updated_params.cameras.names:
                updated_params.cameras.add_entry(value_1)
                entry = updated_params.cameras.get_entry(value_1)
                param_name = f"{self.prefix_}cameras.{value_1}.image_topic"
                if not self.node_.has_parameter(self.prefix_ + param_name):
                    descriptor = ParameterDescriptor(
                        description="Topic name to subscribe for images. If empty, defaults to '<camera_name>_raw' for not compressed image and '<camera_name>/compressed' for compressed images.",
                        read_only=False,
                    )
                    parameter = entry.image_topic
                    self.node_.declare_parameter(param_name, parameter, descriptor)
                param = self.node_.get_parameter(param_name)
                self.logger_.debug(
                    param.name + ": " + param.type_.name + " = " + str(param.value)
                )
                entry.image_topic = param.value

            for value_1 in updated_params.cameras.names:
                updated_params.cameras.add_entry(value_1)
                entry = updated_params.cameras.get_entry(value_1)
                param_name = f"{self.prefix_}cameras.{value_1}.info_topic"
                if not self.node_.has_parameter(self.prefix_ + param_name):
                    descriptor = ParameterDescriptor(
                        description="Topic name to subscribe for camera info. If empty defaults to '<camera_name>/info'.",
                        read_only=False,
                    )
                    parameter = entry.info_topic
                    self.node_.declare_parameter(param_name, parameter, descriptor)
                param = self.node_.get_parameter(param_name)
                self.logger_.debug(
                    param.name + ": " + param.type_.name + " = " + str(param.value)
                )
                entry.info_topic = param.value

            for value_1 in updated_params.cameras.names:
                updated_params.cameras.add_entry(value_1)
                entry = updated_params.cameras.get_entry(value_1)
                param_name = f"{self.prefix_}cameras.{value_1}.k_matrix"
                if not self.node_.has_parameter(self.prefix_ + param_name):
                    descriptor = ParameterDescriptor(
                        description="Camera intrinsic matrix. If not equal to all values of 0.0, overwrites values from info ROS topic.",
                        read_only=False,
                    )
                    descriptor.floating_point_range.append(FloatingPointRange())
                    descriptor.floating_point_range[-1].from_value = 0.0
                    descriptor.floating_point_range[-1].to_value = float("inf")
                    parameter = entry.k_matrix
                    self.node_.declare_parameter(param_name, parameter, descriptor)
                param = self.node_.get_parameter(param_name)
                self.logger_.debug(
                    param.name + ": " + param.type_.name + " = " + str(param.value)
                )
                validation_result = ParameterValidators.fixed_size(param, 9)
                if validation_result:
                    raise InvalidParameterValueException(
                        "cameras.__map_names.k_matrix",
                        param.value,
                        "Invalid value set during initialization for parameter cameras.__map_names.k_matrix: "
                        + validation_result,
                    )
                validation_result = ParameterValidators.lower_element_bounds(param, 0.0)
                if validation_result:
                    raise InvalidParameterValueException(
                        "cameras.__map_names.k_matrix",
                        param.value,
                        "Invalid value set during initialization for parameter cameras.__map_names.k_matrix: "
                        + validation_result,
                    )
                entry.k_matrix = param.value

            self.update_internal_params(updated_params)
