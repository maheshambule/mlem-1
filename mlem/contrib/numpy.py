from types import ModuleType
from typing import Any, ClassVar, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, conlist, create_model

from mlem.core.artifacts import Artifacts, Storage
from mlem.core.data_type import (
    DataHook,
    DataReader,
    DataSerializer,
    DataType,
    DataWriter,
)
from mlem.core.errors import DeserializationError, SerializationError
from mlem.core.requirements import LibRequirementsMixin


def python_type_from_np_string_repr(string_repr: str) -> type:
    np_type = np_type_from_string(string_repr)
    return python_type_from_np_type(np_type)


def python_type_from_np_type(np_type: Union[Type, np.dtype]):
    if np_type.type == np.object_:
        return object
    value = np_type.type()
    if np_type.type.__module__ == "numpy":
        value = value.item()
    return type(value)


def np_type_from_string(string_repr) -> np.dtype:
    try:
        return np.dtype(string_repr)
    except TypeError as e:
        raise ValueError(f"Unknown numpy type {string_repr}") from e


class NumpyNumberType(
    LibRequirementsMixin, DataType, DataSerializer, DataHook
):
    """
    :class:`.DataType` implementation for `numpy.number` objects which
    converts them to built-in Python numbers and vice versa.

    :param dtype: `numpy.number` data type as string
    """

    libraries: ClassVar[List[ModuleType]] = [np]
    type: ClassVar[str] = "number"
    dtype: str

    # def get_spec(self) -> ArgList:
    #     return [Field(None, python_type_from_np_string_repr(self.dtype), False)]

    def deserialize(self, obj: dict) -> Any:
        return self.actual_type(obj)  # pylint: disable=not-callable

    def serialize(self, instance: np.number) -> object:  # type: ignore
        self.check_type(instance, np.number, ValueError)
        return instance.item()

    @property
    def actual_type(self) -> Type:
        return np_type_from_string(self.dtype).type

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, np.number)

    @classmethod
    def process(cls, obj: np.number, **kwargs) -> "NumpyNumberType":
        return NumpyNumberType(dtype=obj.dtype.name)

    def get_writer(self, project: str = None, filename: str = None, **kwargs):
        return NumpyNumberWriter(**kwargs)

    def get_model(self, prefix: str = "") -> Type:
        return python_type_from_np_string_repr(self.dtype)


class NumpyNdarrayType(
    LibRequirementsMixin, DataType, DataHook, DataSerializer
):
    """
    :class:`.DataType` implementation for `np.ndarray` objects
    which converts them to built-in Python lists and vice versa.

    :param shape: shape of `numpy.ndarray` objects in data
    :param dtype: data type of `numpy.ndarray` objects in data
    """

    type: ClassVar[str] = "ndarray"
    libraries: ClassVar[List[ModuleType]] = [np]

    shape: Tuple[Optional[int], ...]
    dtype: str

    @staticmethod
    def _abstract_shape(shape):
        return (None,) + shape[1:]

    @classmethod
    def process(cls, obj, **kwargs) -> DataType:
        return NumpyNdarrayType(
            shape=cls._abstract_shape(obj.shape), dtype=obj.dtype.name
        )

    @classmethod
    def is_object_valid(cls, obj: Any) -> bool:
        return isinstance(obj, np.ndarray)

    def deserialize(self, obj):
        try:
            ret = np.array(obj, dtype=np_type_from_string(self.dtype))
        except (ValueError, TypeError) as e:
            raise DeserializationError(
                f"given object: {obj} could not be converted to array "
                f"of type: {np_type_from_string(self.dtype)}"
            ) from e
        self._check_shape(ret, DeserializationError)
        return ret

    def _subtype(self, subshape: Tuple[Optional[int], ...]):
        if len(subshape) == 0:
            return python_type_from_np_string_repr(self.dtype)
        return conlist(
            self._subtype(subshape[1:]),
            min_items=subshape[0],
            max_items=subshape[0],
        )

    def get_model(self, prefix: str = "") -> Type[BaseModel]:
        # TODO: https://github.com/iterative/mlem/issues/33
        return create_model(
            prefix + "NumpyNdarray", __root__=(List[self._subtype(self.shape[1:])], ...)  # type: ignore
        )

    def serialize(self, instance: np.ndarray):
        self.check_type(instance, np.ndarray, SerializationError)
        exp_type = np_type_from_string(self.dtype)
        if instance.dtype != exp_type:
            raise SerializationError(
                f"given array is of type: {instance.dtype}, expected: {exp_type}"
            )
        self._check_shape(instance, SerializationError)
        return instance.tolist()

    def _check_shape(self, array, exc_type):
        if tuple(array.shape)[1:] != self.shape[1:]:
            raise exc_type(
                f"given array is of shape: {(None,) + tuple(array.shape)[1:]}, expected: {self.shape}"
            )

    def get_writer(self, project: str = None, filename: str = None, **kwargs):
        return NumpyArrayWriter()


DATA_KEY = "data"


class NumpyNumberWriter(DataWriter):
    type: ClassVar[str] = "numpy_number"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            f.write(str(data.data).encode("utf-8"))
        return NumpyNumberReader(data_type=data), {self.art_name: art}


class NumpyNumberReader(DataReader):
    type: ClassVar[str] = "numpy_number"
    data_type: NumpyNumberType

    def read(self, artifacts: Artifacts) -> DataType:
        if DataWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DataWriter.art_name} file"
            )
        with artifacts[DataWriter.art_name].open() as f:
            res = f.read()
            data = self.data_type.actual_type(res)
            return self.data_type.copy().bind(data)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError


class NumpyArrayWriter(DataWriter):
    """DataWriter implementation for numpy ndarray"""

    type: ClassVar[str] = "numpy"

    def write(
        self, data: DataType, storage: Storage, path: str
    ) -> Tuple[DataReader, Artifacts]:
        with storage.open(path) as (f, art):
            np.savez_compressed(f, **{DATA_KEY: data.data})
        return NumpyArrayReader(data_type=data), {self.art_name: art}


class NumpyArrayReader(DataReader):
    """DataReader implementation for numpy ndarray"""

    type: ClassVar[str] = "numpy"

    def read(self, artifacts: Artifacts) -> DataType:
        if DataWriter.art_name not in artifacts:
            raise ValueError(
                f"Wrong artifacts {artifacts}: should be one {DataWriter.art_name} file"
            )
        with artifacts[DataWriter.art_name].open() as f:
            data = np.load(f)[DATA_KEY]
        return self.data_type.copy().bind(data)

    def read_batch(
        self, artifacts: Artifacts, batch_size: int
    ) -> Iterator[DataType]:
        raise NotImplementedError
