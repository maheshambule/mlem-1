import warnings
from functools import cached_property
from typing import IO, Any, ClassVar, Optional, Text, Union, List


import numpy as np
import onnx
import onnxruntime as onnxrt
import pandas as pd
from google import protobuf
from numpy.typing import DTypeLike
from onnx import ModelProto, ValueInfoProto, load_model, TypeProto, TensorProto, save_model
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE, STORAGE_ELEMENT_TYPE_TO_FIELD, OPTIONAL_ELEMENT_TYPE_TO_FIELD

from mlem.contrib.numpy import NumpyNdarrayType
from mlem.core.artifacts import Storage, Artifacts
from mlem.core.data_type import ArrayType, PrimitiveType, DictType
from mlem.core.hooks import IsInstanceHookMixin
from mlem.core.model import (
    ModelHook,
    ModelIO,
    ModelType,
    Signature,
)
from mlem.core.requirements import InstallableRequirement, Requirements
from mlem.utils.module import get_object_requirements


def convert_to_numpy(
        data: Union[np.ndarray, pd.DataFrame], dtype: DTypeLike
) -> np.ndarray:
    """Converts input data to numpy"""
    if isinstance(data, np.ndarray):
        pass
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    else:
        raise TypeError(f"input data type: {type(data)} is not supported")
    return data.astype(dtype=dtype)


def get_onnx_to_numpy_type(value_info: ValueInfoProto) -> DTypeLike:
    """Returns numpy equivalent type of onnx value info"""
    onnx_type = value_info.type.tensor_type.elem_type
    return TENSOR_TYPE_TO_NP_TYPE[onnx_type]


class ONNXWrappedModel:
    """
    Wrapper for `onnx` models and onnx runtime.
    """
    model: ModelProto
    runtime_session: onnxrt.InferenceSession

    def __init__(self, model: Union[ModelProto, IO[bytes], Text]):
        self.model = (
            model if isinstance(model, ModelProto) else load_model(model)
        )

    @property
    def runtime_session(self) -> onnxrt.InferenceSession:
        """Creates onnx runtime inference session"""
        # TODO - add support for runtime providers, options. add support for GPU devices.
        return onnxrt.InferenceSession(self.model.SerializeToString())

    def predict(self, data: Union[List, np.ndarray, pd.DataFrame]) -> Any:
        """Returns inference output for given input data"""
        model_inputs = self.runtime_session.get_inputs()

        if not isinstance(data, list):
            data = [data]

        if len(model_inputs) != len(data):
            raise ValueError(
                f"no of inputs provided: {len(data)}, "
                f"expected: {len(model_inputs)}"
            )

        input_dict = {}
        for model_input, input_data in zip(self.model.graph.input, data):
            input_dict[model_input.name] = convert_to_numpy(
                input_data, get_onnx_to_numpy_type(model_input)
            )

        label_names = [out.name for out in self.runtime_session.get_outputs()]
        pred_onnx = self.runtime_session.run(label_names, input_dict)

        output = []
        for input_data in pred_onnx:
            if isinstance(
                    input_data, list
            ):  # TODO - temporary workaround to fix fastapi model issues
                output.append(pd.DataFrame(input_data).to_numpy())
            else:
                output.append(input_data)

        return output


def get_type(elem_type):
    """Converts onnx integer datatype to numpy datatype"""
    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))

    try:
        from onnx import TensorProto
    except ImportError as e:
        raise ImportError("Unable to import TensorProto from onnx {}".format(e))

    # Onnx mapping converts bfloat16 to float16 because
    # numpy does not have a bfloat16 data type. However,
    # tvm has one, so we force the return type to be bfloat16
    # if elem_type == int(TensorProto.BFLOAT16):
    #     return "bfloat16"
    #

    if elem_type in TENSOR_TYPE_TO_NP_TYPE:
        return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])
    elif elem_type in STORAGE_ELEMENT_TYPE_TO_FIELD:
        return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])
    elif elem_type in OPTIONAL_ELEMENT_TYPE_TO_FIELD:
        return str(OPTIONAL_ELEMENT_TYPE_TO_FIELD[elem_type])


TENSOR_TYPE_TO_PYTHON_TYPE = {
    int(TensorProto.FLOAT): float,
    int(TensorProto.UINT8): int,
    int(TensorProto.INT8): int,
    int(TensorProto.UINT16): np.dtype('uint16'),
    int(TensorProto.INT16): np.dtype('int16'),
    int(TensorProto.INT32): np.dtype('int32'),
    int(TensorProto.INT64): np.dtype('int64'),
    int(TensorProto.BOOL): np.dtype('bool'),
    int(TensorProto.FLOAT16): np.dtype('float16'),
    int(TensorProto.BFLOAT16): np.dtype('float16'),  # native numpy does not support bfloat16
    int(TensorProto.DOUBLE): np.dtype('float64'),
    int(TensorProto.COMPLEX64): np.dtype('complex64'),
    int(TensorProto.COMPLEX128): np.dtype('complex128'),
    int(TensorProto.UINT32): np.dtype('uint32'),
    int(TensorProto.UINT64): np.dtype('uint64'),
    int(TensorProto.STRING): np.dtype('object')
}


def get_info(info_proto):
    """Extract the shape from a ValueInfoProto."""
    shape = []
    shape_name = []
    for dim in info_proto.type.tensor_type.shape.dim:
        name = dim.dim_param
        value = dim.dim_value
        if value is None or value == 0:
            value = Any
            shape_name.append(name)
        else:
            shape_name.append(value)
        shape.append(value)

    name = info_proto.name
    dtype = None
    if info_proto.type.tensor_type.elem_type is not None:
        dtype = get_type(info_proto.type.tensor_type.elem_type)


def get_tensor_type_shape(tensor_type):
    # TODO - add support for dim.dim_param
    return (dim.dim_value for dim in tensor_type.shape.dim)


def get_type_of(dtype: TypeProto):
    if dtype.HasField('sequence_type'):
        # elem_type is of type TypeProto
        return ArrayType(dtype=get_type_of(dtype.sequence_type.elem_type), size=-1)
    elif dtype.HasField('tensor_type'):
        return NumpyNdarrayType(dtype=TENSOR_TYPE_TO_NP_TYPE[dtype.tensor_type.elem_type],
                                shape=get_tensor_type_shape(dtype.tensor_type))
    elif dtype.HasField('map_type'):
        # dtype.tensor_type.key_type is either TensorProto.STRING or TensorProto int type
        key_type = str if dtype.tensor_type.key_type == TensorProto.STRING else int

        # value_type is of type TypeProto
        value_type = get_type_of(dtype.tensor_type.value_type)
        return DictType(item_types)
    else:
        raise TypeError(
            f"unsupported type: {dtype}, "
            f"expected: 'sequence_type', 'tensor_type', 'map_type'"
        )

    return name, shape, dtype, shape_name


def new_var(name_hint, type_annotation="", shape="", dtype="float32"):
    return name_hint + type_annotation + str(shape) + str(dtype)


class ONNXModelSignature:

    def __init__(self):
        self._nodes = {}
        self._params = {}
        self._inputs = {}
        self._outputs = {}
        self._renames = {}
        self._num_input = 0
        self._num_output = 0
        self._num_param = 0
        self._shape = {}
        self._input_names = []
        self._output_names = []
        # self._dtype = None
        self.opset = None
        # self._freeze_params = True

    def from_onnx(self, graph):
        self._parse_graph_initializers(graph)
        self._parse_graph_input(graph)
        self._check_user_inputs_in_outermost_graph_scope()
        self._parse_graph_output(graph)

        self._inputs
        self._outputs

    def _parse_graph_initializers(self, graph):
        """Parse network inputs to relay, aka parameters."""
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            array = self._parse_array(init_tensor)
            if self._freeze_params:
                self._nodes[init_tensor.name] = array
            else:
                self._params[init_tensor.name] = array
                self._nodes[init_tensor.name] = new_var(
                    init_tensor.name,
                    shape=self._params[init_tensor.name].shape,
                    dtype=self._params[init_tensor.name].dtype,
                )

    def _parse_graph_input(self, graph):
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            if i_name in self._params:
                # i is a param instead of input
                self._num_param += 1
                self._nodes[i_name] = new_var(
                    i_name, shape=self._params[i_name].shape, dtype=self._params[i_name].dtype
                )
            elif i_name in self._nodes:
                continue
            else:
                self._num_input += 1
                self._input_names.append(i_name)
                if i_name in self._shape:
                    i_shape = self._shape[i_name]
                else:
                    if "?" in str(i_shape):
                        warning_msg = (
                                "Input %s has unknown dimension shapes: %s. "
                                "Specifying static values may improve performance"
                                % (i_name, str(i_shape_name))
                        )
                        warnings.warn(warning_msg)
                # if isinstance(self._dtype, dict):
                #     dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                # else:
                #     dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=i_shape, dtype=d_type)
            self._inputs[i_name] = self._nodes[i_name]

    def _parse_graph_output(self, graph):
        for i in graph.output:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            if i_name in self._nodes:
                continue
            else:
                self._num_output += 1
                self._output_names.append(i_name)
                if "-1" in str(i_shape):
                    warning_msg = (
                            "Input %s has unknown dimension shapes: %s. "
                            "Specifying static values may improve performance"
                            % (i_name, str(i_shape_name))
                    )
                    warnings.warn(warning_msg)
                # if isinstance(self._dtype, dict):
                #     dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                # else:
                #     dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=i_shape, dtype=d_type)
            self._outputs[i_name] = self._nodes[i_name]

    def _check_user_inputs_in_outermost_graph_scope(self):
        """Only check user inputs in the outer-most graph scope."""
        # if self._old_manager is None:
        assert all(
            [name in self._input_names for name in self._shape.keys()]
        ), "User specified the shape for inputs that weren't found in the graph: " + str(
            self._shape
        )

    def _parse_value_proto(self, value_proto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name


class ModelProtoIO(ModelIO):
    """IO for ONNX model object"""

    type: ClassVar[str] = "model_proto"

    def dump(self, storage: Storage, path: str, model) -> Artifacts:
        with storage.open(path) as (f, art):
            save_model(model, f)
        return {self.art_name: art}

    def load(self, artifacts: Artifacts):
        if len(artifacts) != 1:
            raise ValueError("Invalid artifacts: should be one .onx file")
        with artifacts[self.art_name].open() as f:
            return load_model(f)


class ONNXModel(ModelType, ModelHook, IsInstanceHookMixin):
    """
    :class:`mlem.core.model.ModelType` implementation for `onnx` models
    """

    type: ClassVar[str] = "onnx"
    io: ModelIO = ModelProtoIO()
    valid_types: ClassVar = (ModelProto,)

    class Config:
        keep_untouched = (cached_property,)

    @classmethod
    def process(
            cls, obj: Any, sample_data: Optional[Any] = None, **kwargs
    ) -> ModelType:

        model = ONNXModel(io=ModelProtoIO(), methods={}).bind(obj)
        # TODO - use ONNX infer shapes.
        # ONNXModelSignature().from_onnx(obj.model.graph)
        onnxrt_predict = Signature.from_method(
            model.predict, auto_infer=sample_data is not None, data=sample_data
        )
        model.methods = {
            "predict": onnxrt_predict,
        }

        return model

    @cached_property
    def runtime_session(self) -> onnxrt.InferenceSession:
        """Provides onnx runtime inference session"""
        # TODO - add support for runtime providers, options. add support for GPU devices.
        return onnxrt.InferenceSession(self.model.SerializeToString())

    def predict(self, data: Union[List, np.ndarray, pd.DataFrame]) -> Any:
        """Returns inference output for given input data"""
        model_inputs = self.runtime_session.get_inputs()

        if not isinstance(data, list):
            data = [data]

        if len(model_inputs) != len(data):
            raise ValueError(
                f"no of inputs provided: {len(data)}, "
                f"expected: {len(model_inputs)}"
            )

        input_dict = {}
        for model_input, input_data in zip(self.model.graph.input, data):
            input_dict[model_input.name] = convert_to_numpy(
                input_data, get_onnx_to_numpy_type(model_input)
            )

        label_names = [out.name for out in self.runtime_session.get_outputs()]
        pred_onnx = self.runtime_session.run(label_names, input_dict)

        output = []
        for input_data in pred_onnx:
            if isinstance(
                    input_data, list
            ):  # TODO - temporary workaround to fix fastapi model issues
                output.append(pd.DataFrame(input_data).to_numpy())
            else:
                output.append(input_data)

        return output

    def get_requirements(self) -> Requirements:
        return super().get_requirements() + InstallableRequirement.from_module(
            onnx
        ) + get_object_requirements(
            self.predict
        ) + Requirements.new(InstallableRequirement(module="protobuf", version="3.20.1"))
        # https://github.com/protocolbuffers/protobuf/issues/10051




