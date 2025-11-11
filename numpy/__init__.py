"""A tiny subset of NumPy implemented in pure Python for testing."""
from __future__ import annotations

import builtins
import math
import random
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple

float64 = float


def asarray(obj, dtype=float):
    if isinstance(obj, (list, tuple)):
        return [asarray(v, dtype=dtype) for v in obj]
    return dtype(obj)


def array(obj, dtype=float):
    return asarray(obj, dtype=dtype)


def zeros(shape: Sequence[int]):
    if not shape:
        return 0.0
    return [zeros(shape[1:]) for _ in range(shape[0])]


def ones(shape: Sequence[int]):
    if not shape:
        return 1.0
    return [ones(shape[1:]) for _ in range(shape[0])]


def full(shape: Sequence[int], value: float):
    if not shape:
        return value
    return [full(shape[1:], value) for _ in range(shape[0])]


def zeros_like(tensor):
    return full(shape(tensor), 0.0)


def ones_like(tensor):
    return full(shape(tensor), 1.0)


def shape(tensor) -> Tuple[int, ...]:
    if not isinstance(tensor, list):
        return ()
    if not tensor:
        return (0,)
    return (len(tensor),) + shape(tensor[0])


def add(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return _elementwise_pair(a, b, lambda x, y: x + y)
    if isinstance(a, list):
        return _apply(a, lambda v: v + b)
    if isinstance(b, list):
        return _apply(b, lambda v: a + v)
    return a + b


def subtract(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return _elementwise_pair(a, b, lambda x, y: x - y)
    if isinstance(a, list):
        return _apply(a, lambda v: v - b)
    if isinstance(b, list):
        return _apply(b, lambda v: a - v)
    return a - b


def multiply(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return _elementwise_pair(a, b, lambda x, y: x * y)
    if isinstance(a, list):
        return _apply(a, lambda v: v * b)
    if isinstance(b, list):
        return _apply(b, lambda v: a * v)
    return a * b


def true_divide(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return _elementwise_pair(a, b, lambda x, y: x / y)
    if isinstance(a, list):
        return _apply(a, lambda v: v / b)
    if isinstance(b, list):
        return _apply(b, lambda v: a / v)
    return a / b


def _flatten(tensor) -> List[float]:
    if not isinstance(tensor, list):
        return [float(tensor)]
    flat: List[float] = []
    for value in tensor:
        flat.extend(_flatten(value))
    return flat


def _iterate_indices(shape_values: Sequence[int]) -> Iterator[Tuple[int, ...]]:
    if not shape_values:
        yield ()
        return
    for i in range(shape_values[0]):
        for rest in _iterate_indices(shape_values[1:]):
            yield (i,) + rest


def _get(tensor, index: Tuple[int, ...]):
    ref = tensor
    for idx in index:
        ref = ref[idx]
    return ref


def _set(tensor, index: Tuple[int, ...], value) -> None:
    ref = tensor
    for idx in index[:-1]:
        ref = ref[idx]
    ref[index[-1]] = value


def _apply(tensor, fn):
    if not isinstance(tensor, list):
        return fn(float(tensor))
    return [_apply(v, fn) for v in tensor]


def sum(tensor, axis: int | None = None, keepdims: bool = False):
    if axis is None:
        return float(math.fsum(_flatten(tensor)))
    dims = len(shape(tensor))
    if axis < 0:
        axis += dims
    new_shape = list(shape(tensor))
    target_shape = new_shape[:axis] + new_shape[axis + 1 :]
    result = zeros(target_shape) if target_shape else 0.0
    for index in _iterate_indices(shape(tensor)):
        value = _get(tensor, index)
        reduced_index = index[:axis] + index[axis + 1 :]
        if target_shape:
            current = _get(result, reduced_index)
            _set(result, reduced_index, current + value)
        else:
            result += value
    if keepdims:
        result = _expand_dims(result, axis)
    return result


def max(tensor, axis: int | None = None, keepdims: bool = False):
    return _reduction(tensor, axis, keepdims, builtins.max, float("-inf"))


def min(tensor, axis: int | None = None, keepdims: bool = False):
    return _reduction(tensor, axis, keepdims, builtins.min, float("inf"))


def mean(tensor, axis: int | None = None, keepdims: bool = False):
    if axis is None:
        flat = _flatten(tensor)
        return float(math.fsum(flat) / max(len(flat), 1))
    dims = len(shape(tensor))
    if axis < 0:
        axis += dims
    sum_tensor = sum(tensor, axis=axis, keepdims=keepdims)
    count_shape = shape(tensor)
    counts = zeros(count_shape)
    for index in _iterate_indices(count_shape):
        _set(counts, index, 1.0)
    count_tensor = sum(counts, axis=axis, keepdims=keepdims)
    return _elementwise_pair(sum_tensor, count_tensor, lambda s, c: s / max(c, 1.0))


def clip(tensor, a_min: float | None = None, a_max: float | None = None):
    def fn(value: float) -> float:
        if a_min is not None and value < a_min:
            value = a_min
        if a_max is not None and value > a_max:
            value = a_max
        return value

    return _apply(tensor, fn)


def abs(tensor):
    return _apply(tensor, builtins.abs)


def square(tensor):
    return _apply(tensor, lambda v: v * v)


def log(tensor):
    return _apply(tensor, math.log)


def exp(tensor):
    return _apply(tensor, math.exp)


def transpose(tensor, axes: Sequence[int]):
    orig_shape = shape(tensor)
    if len(axes) != len(orig_shape):
        raise ValueError("Axes must match tensor rank")
    new_shape = tuple(orig_shape[axis] for axis in axes)
    result = zeros(new_shape)
    for index in _iterate_indices(orig_shape):
        new_index = tuple(index[axis] for axis in axes)
        _set(result, new_index, _get(tensor, index))
    return result


def matmul(a, b):
    a_shape = shape(a)
    b_shape = shape(b)
    if len(a_shape) == 3 and len(b_shape) == 3:
        return [matmul(a[i], b[i]) for i in range(a_shape[0])]
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise ValueError("matmul expects 2D or batched 2D tensors")
    rows, shared = a_shape
    if shared != b_shape[0]:
        raise ValueError("Inner dimensions must align")
    cols = b_shape[1]
    result = zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            value = 0.0
            for k in range(shared):
                value += _get(a, (i, k)) * _get(b, (k, j))
            _set(result, (i, j), value)
    return result


def all(tensor, axis: int | None = None):
    if axis is not None:
        raise NotImplementedError("axis argument is not supported in this stub")
    return builtins.all(bool(v) for v in _flatten(tensor))


def allclose(a, b, tol: float = 1e-6):
    flat_a = _flatten(a)
    flat_b = _flatten(b)
    if len(flat_a) != len(flat_b):
        return False
    return builtins.all(builtins.abs(x - y) <= tol for x, y in zip(flat_a, flat_b))


def broadcast_to(tensor, target_shape: Sequence[int]):
    current_shape = shape(tensor)
    if len(current_shape) != len(target_shape):
        raise ValueError("broadcast_to supports tensors with matching rank")
    result = tensor
    for current, target in zip(current_shape, target_shape):
        if current == target:
            continue
        if current != 1:
            raise ValueError("Only singleton dimensions can be broadcast")
        result = _broadcast_dim(result, target)
    return result


def _broadcast_dim(tensor, repeat: int):
    if not isinstance(tensor, list):
        return [tensor for _ in range(repeat)]
    return [_broadcast_dim(tensor[0], repeat) for _ in range(repeat)]


def _elementwise_pair(a, b, fn):
    if not isinstance(a, list):
        return fn(a, b)
    return [_elementwise_pair(x, y, fn) for x, y in zip(a, b)]


def _expand_dims(tensor, axis: int):
    if axis == 0:
        return [tensor]
    if not isinstance(tensor, list):
        return tensor
    return [_expand_dims(v, axis - 1) for v in tensor]


def _reduction(tensor, axis: int | None, keepdims: bool, fn, initial: float):
    if axis is None:
        return float(fn(_flatten(tensor)))
    dims = len(shape(tensor))
    if axis < 0:
        axis += dims
    new_shape = list(shape(tensor))
    target_shape = new_shape[:axis] + new_shape[axis + 1 :]
    result = full(target_shape, initial) if target_shape else initial
    for index in _iterate_indices(shape(tensor)):
        value = _get(tensor, index)
        reduced_index = index[:axis] + index[axis + 1 :]
        if target_shape:
            current = _get(result, reduced_index)
            _set(result, reduced_index, fn([current, value]))
        else:
            result = fn([result, value])
    if keepdims:
        result = _expand_dims(result, axis)
    return result


def _prod(values: Sequence[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def _reshape(values: List[float], target_shape: Sequence[int]):
    if not target_shape:
        return values[0]
    if len(target_shape) == 1:
        return values[: target_shape[0]]
    step = _prod(target_shape[1:])
    return [_reshape(values[i * step:(i + 1) * step], target_shape[1:]) for i in range(target_shape[0])]


class _Linalg:
    def norm(self, tensor, axis: int = -1, keepdims: bool = False):
        squared = square(tensor)
        summed = sum(squared, axis=axis, keepdims=keepdims)
        if isinstance(summed, list):
            return _apply(summed, math.sqrt)
        return math.sqrt(summed)


linalg = _Linalg()


@dataclass
class _Generator:
    rng: random.Random

    def normal(self, size: Tuple[int, ...]):
        total = _prod(size)
        values = [self.rng.gauss(0.0, 1.0) for _ in range(total)]
        return _reshape(values, size)

    def random(self, size: Tuple[int, ...]):
        total = _prod(size)
        values = [self.rng.random() for _ in range(total)]
        return _reshape(values, size)


class _RandomModule:
    def default_rng(self, seed: int | None = None) -> _Generator:
        return _Generator(random.Random(seed))


random = _RandomModule()


__all__ = [
    "array",
    "asarray",
    "zeros",
    "ones",
    "full",
    "zeros_like",
    "ones_like",
    "shape",
    "sum",
    "max",
    "min",
    "mean",
    "clip",
    "abs",
    "square",
    "log",
    "exp",
    "transpose",
    "matmul",
    "add",
    "subtract",
    "multiply",
    "true_divide",
    "all",
    "allclose",
    "broadcast_to",
    "linalg",
    "random",
    "float64",
]
