"""Lightweight NumPy-backed PyTorch compatibility layer with native PyTorch v2 .pt zip archive I/O.

Used automatically only when PyTorch (`torch`) is not installed in the active Python environment.
When `torch` is installed, `src/dqn.py` uses standard PyTorch directly. Checkpoints saved by this
layer use the official PyTorch v2 zip format (`torch._utils._rebuild_tensor_v2` + `torch.FloatStorage`)
and load natively in standard PyTorch (`torch.load`).
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import io
from pathlib import Path
import pickle
from typing import Any, Callable
import zipfile
import numpy as np


class dtype:
    def __init__(self, np_dtype):
        self.np_dtype = np.dtype(np_dtype)

    def __repr__(self) -> str:
        return f"torch.{self.np_dtype.name}"


float32 = dtype(np.float32)
int64 = dtype(np.int64)
long = int64
bool_ = dtype(np.bool_)


class device:
    def __init__(self, name: str = "cpu"):
        self.type = str(name)

    def __repr__(self) -> str:
        return f"device(type='{self.type}')"


_GRAD_ENABLED = True

class _NoGradContext:
    def __enter__(self):
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev
        return False


def no_grad() -> _NoGradContext:
    return _NoGradContext()


def set_num_threads(n: int) -> None:
    pass


def manual_seed(seed: int) -> None:
    np.random.seed(seed)


def _to_np(x: Any) -> np.ndarray:
    if isinstance(x, Tensor):
        return x._arr
    return np.asarray(x)


class Tensor:
    """Autograd-capable multidimensional array matching the PyTorch Tensor API."""

    def __init__(
        self,
        arr: Any,
        dtype_Spec: Any = None,
        requires_grad: bool = False,
        _grad_fn: Callable[[np.ndarray], None] | None = None,
    ):
        if isinstance(arr, Tensor):
            arr = arr._arr
        if dtype_Spec is not None:
            np_dt = dtype_Spec.np_dtype if isinstance(dtype_Spec, dtype) else np.dtype(dtype_Spec)
            self._arr = np.asarray(arr, dtype=np_dt)
        else:
            a = np.asarray(arr)
            if a.dtype == np.float64:
                a = a.astype(np.float32)
            self._arr = a
        self.requires_grad = bool(requires_grad) and _GRAD_ENABLED
        self.grad: Tensor | None = None
        self._grad_fn = _grad_fn if _GRAD_ENABLED else None
        self.device = device("cpu")

    @property
    def data(self) -> "Tensor":
        return Tensor(self._arr, requires_grad=False)

    @data.setter
    def data(self, val: "Tensor") -> None:
        self._arr[...] = _to_np(val)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._arr.shape

    @property
    def ndim(self) -> int:
        return self._arr.ndim

    @property
    def dtype(self) -> Any:
        return self._arr.dtype

    def size(self, dim: int | None = None):
        return self._arr.shape if dim is None else self._arr.shape[dim]

    def numel(self) -> int:
        return int(self._arr.size)

    def to(self, *args, **kwargs) -> "Tensor":
        for a in args:
            if isinstance(a, dtype) or a in (np.float32, np.int64, np.bool_, bool):
                return Tensor(self._arr, dtype_Spec=a, requires_grad=self.requires_grad, _grad_fn=self._grad_fn)
        return self

    def cpu(self) -> "Tensor":
        return self

    def detach(self) -> "Tensor":
        return Tensor(self._arr.copy(), requires_grad=False)

    def clone(self) -> "Tensor":
        return Tensor(self._arr.copy(), requires_grad=self.requires_grad)

    def copy_(self, other: "Tensor") -> "Tensor":
        self._arr[...] = _to_np(other)
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def item(self) -> float | int | bool:
        return self._arr.item()

    def tolist(self) -> Any:
        return self._arr.tolist()

    def float(self) -> "Tensor":
        return Tensor(self._arr.astype(np.float32), requires_grad=self.requires_grad)

    def long(self) -> "Tensor":
        return Tensor(self._arr.astype(np.int64), requires_grad=False)

    def bool(self) -> "Tensor":
        return Tensor(self._arr.astype(bool), requires_grad=False)

    def view(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        orig_shape = self._arr.shape

        def _bw(g: np.ndarray) -> None:
            if self.requires_grad or self._grad_fn is not None:
                self._backward_pass(g.reshape(orig_shape))

        return Tensor(
            self._arr.reshape(shape),
            requires_grad=self.requires_grad,
            _grad_fn=_bw if (self.requires_grad or self._grad_fn is not None) else None,
        )

    def reshape(self, *shape: int) -> "Tensor":
        return self.view(*shape)

    def unsqueeze(self, dim: int) -> "Tensor":
        orig_shape = self._arr.shape

        def _bw(g: np.ndarray) -> None:
            if self.requires_grad or self._grad_fn is not None:
                self._backward_pass(g.reshape(orig_shape))

        return Tensor(
            np.expand_dims(self._arr, axis=dim),
            requires_grad=self.requires_grad,
            _grad_fn=_bw if (self.requires_grad or self._grad_fn is not None) else None,
        )

    def squeeze(self, dim: int | None = None) -> "Tensor":
        orig_shape = self._arr.shape

        def _bw(g: np.ndarray) -> None:
            if self.requires_grad or self._grad_fn is not None:
                self._backward_pass(g.reshape(orig_shape))

        if dim is None:
            out = np.squeeze(self._arr)
        else:
            out = np.squeeze(self._arr, axis=dim) if self._arr.shape[dim] == 1 else self._arr
        return Tensor(
            out,
            requires_grad=self.requires_grad,
            _grad_fn=_bw if (self.requires_grad or self._grad_fn is not None) else None,
        )

    def __getitem__(self, idx: Any) -> "Tensor":
        if isinstance(idx, Tensor):
            idx = idx._arr
        elif isinstance(idx, tuple):
            idx = tuple(i._arr if isinstance(i, Tensor) else i for i in idx)
        orig_shape = self._arr.shape

        def _bw(g: np.ndarray) -> None:
            if self.requires_grad or self._grad_fn is not None:
                full_g = np.zeros(orig_shape, dtype=np.float32)
                np.add.at(full_g, idx, g)
                self._backward_pass(full_g)

        return Tensor(
            self._arr[idx],
            requires_grad=self.requires_grad,
            _grad_fn=_bw if (self.requires_grad or self._grad_fn is not None) else None,
        )

    def __setitem__(self, idx: Any, val: Any) -> None:
        if isinstance(idx, Tensor):
            idx = idx._arr
        elif isinstance(idx, tuple):
            idx = tuple(i._arr if isinstance(i, Tensor) else i for i in idx)
        self._arr[idx] = _to_np(val)

    @staticmethod
    def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        g = grad
        while g.ndim > len(target_shape):
            g = g.sum(axis=0)
        for i, dim in enumerate(target_shape):
            if dim == 1 and g.shape[i] != 1:
                g = g.sum(axis=i, keepdims=True)
        return g

    def _backward_pass(self, grad_arr: np.ndarray) -> None:
        if self.requires_grad:
            if self.grad is None:
                self.grad = Tensor(np.zeros_like(self._arr, dtype=np.float32))
            self.grad._arr += grad_arr.astype(np.float32)
        if self._grad_fn is not None:
            self._grad_fn(grad_arr)

    def backward(self, grad: "Tensor | None" = None) -> None:
        g = np.ones_like(self._arr, dtype=np.float32) if grad is None else _to_np(grad).astype(np.float32)
        self._backward_pass(g)

    def __add__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = self._arr + other_t._arr
        need_grad = self.requires_grad or (self._grad_fn is not None) or other_t.requires_grad or (other_t._grad_fn is not None)

        def _bw(g: np.ndarray) -> None:
            if self.requires_grad or self._grad_fn is not None:
                self._backward_pass(Tensor._unbroadcast(g, self._arr.shape))
            if other_t.requires_grad or other_t._grad_fn is not None:
                other_t._backward_pass(Tensor._unbroadcast(g, other_t._arr.shape))

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = self._arr - other_t._arr
        need_grad = self.requires_grad or (self._grad_fn is not None) or other_t.requires_grad or (other_t._grad_fn is not None)

        def _bw(g: np.ndarray) -> None:
            if self.requires_grad or self._grad_fn is not None:
                self._backward_pass(Tensor._unbroadcast(g, self._arr.shape))
            if other_t.requires_grad or other_t._grad_fn is not None:
                other_t._backward_pass(Tensor._unbroadcast(-g, other_t._arr.shape))

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)

    def __rsub__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return other_t.__sub__(self)

    def __neg__(self) -> "Tensor":
        return Tensor(0.0) - self

    def __mul__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        out = self._arr * other_t._arr
        need_grad = self.requires_grad or (self._grad_fn is not None) or other_t.requires_grad or (other_t._grad_fn is not None)
        s_arr = self._arr.copy()
        o_arr = other_t._arr.copy()

        def _bw(g: np.ndarray) -> None:
            if self.requires_grad or self._grad_fn is not None:
                self._backward_pass(Tensor._unbroadcast(g * o_arr, s_arr.shape))
            if other_t.requires_grad or other_t._grad_fn is not None:
                other_t._backward_pass(Tensor._unbroadcast(g * s_arr, o_arr.shape))

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> "Tensor":
        other_t = other if isinstance(other, Tensor) else Tensor(other)
        return self * Tensor(1.0 / other_t._arr)

    def __gt__(self, other: Any) -> "Tensor":
        return Tensor(self._arr > _to_np(other), dtype_Spec=bool_)

    def __lt__(self, other: Any) -> "Tensor":
        return Tensor(self._arr < _to_np(other), dtype_Spec=bool_)

    def __ge__(self, other: Any) -> "Tensor":
        return Tensor(self._arr >= _to_np(other), dtype_Spec=bool_)

    def __le__(self, other: Any) -> "Tensor":
        return Tensor(self._arr <= _to_np(other), dtype_Spec=bool_)

    def __and__(self, other: Any) -> "Tensor":
        return Tensor(np.logical_and(self._arr, _to_np(other)), dtype_Spec=bool_)

    def __or__(self, other: Any) -> "Tensor":
        return Tensor(np.logical_or(self._arr, _to_np(other)), dtype_Spec=bool_)

    def __invert__(self) -> "Tensor":
        return Tensor(np.logical_not(self._arr), dtype_Spec=bool_)

    def mean(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        out = self._arr.mean(axis=dim, keepdims=keepdim)
        need_grad = self.requires_grad or (self._grad_fn is not None)
        orig_shape = self._arr.shape

        def _bw(g: np.ndarray) -> None:
            if dim is None:
                count = self._arr.size
                g_exp = np.broadcast_to(g / count, orig_shape)
            else:
                count = orig_shape[dim]
                g_in = g if keepdim else np.expand_dims(g, axis=dim)
                g_exp = np.broadcast_to(g_in / count, orig_shape)
            self._backward_pass(g_exp.astype(np.float32))

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)

    def sum(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        out = self._arr.sum(axis=dim, keepdims=keepdim)
        need_grad = self.requires_grad or (self._grad_fn is not None)
        orig_shape = self._arr.shape

        def _bw(g: np.ndarray) -> None:
            if dim is None:
                g_exp = np.broadcast_to(g, orig_shape)
            else:
                g_in = g if keepdim else np.expand_dims(g, axis=dim)
                g_exp = np.broadcast_to(g_in, orig_shape)
            self._backward_pass(g_exp.astype(np.float32))

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)

    def any(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        return Tensor(np.any(self._arr, axis=dim, keepdims=keepdim), dtype_Spec=bool_)

    def argmax(self, dim: int | None = None, keepdim: bool = False) -> "Tensor":
        res = np.argmax(self._arr, axis=dim)
        if keepdim and dim is not None:
            res = np.expand_dims(res, axis=dim)
        return Tensor(res, dtype_Spec=int64)

    def max(self, dim: int | None = None, keepdim: bool = False):
        if dim is None:
            return Tensor(np.max(self._arr))
        vals = np.max(self._arr, axis=dim, keepdims=keepdim)
        idxs = np.argmax(self._arr, axis=dim)
        if keepdim:
            idxs = np.expand_dims(idxs, axis=dim)
        need_grad = self.requires_grad or (self._grad_fn is not None)
        orig_shape = self._arr.shape
        idxs_Copy = idxs.copy()

        def _bw(g: np.ndarray) -> None:
            g_in = g if keepdim else np.expand_dims(g, axis=dim)
            full_g = np.zeros(orig_shape, dtype=np.float32)
            idx_exp = idxs_Copy if keepdim else np.expand_dims(idxs_Copy, axis=dim)
            np.put_along_axis(full_g, idx_exp, g_in, axis=dim)
            self._backward_pass(full_g)

        return Tensor(vals, requires_grad=need_grad, _grad_fn=_bw if need_grad else None), Tensor(idxs, dtype_Spec=int64)

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        idx_arr = _to_np(index).astype(np.int64)
        out = np.take_along_axis(self._arr, idx_arr, axis=dim)
        need_grad = self.requires_grad or (self._grad_fn is not None)
        orig_shape = self._arr.shape

        def _bw(g: np.ndarray) -> None:
            full_g = np.zeros(orig_shape, dtype=np.float32)
            # Accumulate gathered gradients
            if dim == 1 and orig_shape.__len__() == 2:
                for r in range(idx_arr.shape[0]):
                    for c in range(idx_arr.shape[1]):
                        full_g[r, idx_arr[r, c]] += g[r, c]
            else:
                np.put_along_axis(full_g, idx_arr, g, axis=dim)
            self._backward_pass(full_g)

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        m = _to_np(mask).astype(bool)
        out = np.where(m, np.float32(value), self._arr)
        need_grad = self.requires_grad or (self._grad_fn is not None)

        def _bw(g: np.ndarray) -> None:
            self._backward_pass(np.where(m, 0.0, g).astype(np.float32))

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)

    def scatter_(self, dim: int, index: "Tensor", value: float | "Tensor") -> "Tensor":
        idx_arr = _to_np(index).astype(np.int64)
        val_arr = _to_np(value) if isinstance(value, Tensor) else value
        np.put_along_axis(self._arr, idx_arr, val_arr, axis=dim)
        return self

    def clamp(self, min: float | None = None, max: float | None = None) -> "Tensor":
        out = np.clip(self._arr, min, max)
        need_grad = self.requires_grad or (self._grad_fn is not None)
        s_arr = self._arr.copy()

        def _bw(g: np.ndarray) -> None:
            mask = np.ones_like(s_arr, dtype=np.float32)
            if min is not None:
                mask = mask * (s_arr >= min)
            if max is not None:
                mask = mask * (s_arr <= max)
            self._backward_pass(g * mask)

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)


def as_tensor(data: Any, dtype: Any = None, device: Any = None) -> Tensor:
    if isinstance(data, Tensor):
        return data.to(dtype) if dtype is not None else data
    return Tensor(data, dtype_Spec=dtype)


def zeros(shape: tuple[int, ...] | int, dtype: Any = float32, device: Any = None) -> Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    return Tensor(np.zeros(shape, dtype=np.float32), dtype_Spec=dtype)


def ones_like(x: Tensor) -> Tensor:
    return Tensor(np.ones_like(x._arr))


def full_like(x: Tensor, fill_value: float) -> Tensor:
    return Tensor(np.full_like(x._arr, fill_value))


def eye(n: int, dtype: Any = float32, device: Any = None) -> Tensor:
    return Tensor(np.eye(n, dtype=np.float32), dtype_Spec=dtype)


def where(condition: Tensor, x: Any, y: Any) -> Tensor:
    c = _to_np(condition).astype(bool)
    x_t = x if isinstance(x, Tensor) else Tensor(x)
    y_t = y if isinstance(y, Tensor) else Tensor(y)
    out = np.where(c, x_t._arr, y_t._arr).astype(np.float32)
    need_grad = x_t.requires_grad or (x_t._grad_fn is not None) or y_t.requires_grad or (y_t._grad_fn is not None)

    def _bw(g: np.ndarray) -> None:
        if x_t.requires_grad or x_t._grad_fn is not None:
            x_t._backward_pass(Tensor._unbroadcast(np.where(c, g, 0.0), x_t._arr.shape))
        if y_t.requires_grad or y_t._grad_fn is not None:
            y_t._backward_pass(Tensor._unbroadcast(np.where(c, 0.0, g), y_t._arr.shape))

    return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)


def cat(tensors: list[Tensor], dim: int = 0) -> Tensor:
    arrs = [t._arr for t in tensors]
    out = np.concatenate(arrs, axis=dim)
    need_grad = any(t.requires_grad or (t._grad_fn is not None) for t in tensors)
    splits = np.cumsum([a.shape[dim] for a in arrs[:-1]])

    def _bw(g: np.ndarray) -> None:
        parts = np.split(g, splits, axis=dim)
        for t, p in zip(tensors, parts):
            if t.requires_grad or t._grad_fn is not None:
                t._backward_pass(p)

    return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    a_arr, b_arr = a._arr, b._arr
    out = np.matmul(a_arr, b_arr)
    need_grad = a.requires_grad or (a._grad_fn is not None) or b.requires_grad or (b._grad_fn is not None)

    def _bw(g: np.ndarray) -> None:
        if a.requires_grad or a._grad_fn is not None:
            ga = np.matmul(g, np.swapaxes(b_arr, -1, -2))
            a._backward_pass(Tensor._unbroadcast(ga, a_arr.shape))
        if b.requires_grad or b._grad_fn is not None:
            gb = np.matmul(np.swapaxes(a_arr, -1, -2), g)
            b._backward_pass(Tensor._unbroadcast(gb, b_arr.shape))

    return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)


class Parameter(Tensor):
    def __init__(self, arr: Any):
        super().__init__(arr, dtype_Spec=float32, requires_grad=True)


class Module:
    def __init__(self):
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._buffers: OrderedDict[str, Tensor] = OrderedDict()
        self.training = True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if "_parameters" in self.__dict__:
                self._parameters[name] = value
        elif isinstance(value, Module):
            if "_modules" in self.__dict__:
                self._modules[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        self._buffers[name] = tensor
        object.__setattr__(self, name, tensor)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def to(self, *args, **kwargs) -> "Module":
        return self

    def eval(self) -> "Module":
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def train(self, mode: bool = True) -> "Module":
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def parameters(self) -> list[Parameter]:
        params = list(self._parameters.values())
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def state_dict(self, prefix: str = "") -> OrderedDict[str, Tensor]:
        sd: OrderedDict[str, Tensor] = OrderedDict()
        for k, b in self._buffers.items():
            sd[prefix + k] = Tensor(b._arr.copy())
        for k, p in self._parameters.items():
            sd[prefix + k] = Tensor(p._arr.copy())
        for name, m in self._modules.items():
            sd.update(m.state_dict(prefix=f"{prefix}{name}."))
        return sd

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        own = self._named_refs()
        for k, v in state_dict.items():
            if k in own:
                own[k]._arr[...] = _to_np(v).astype(own[k]._arr.dtype)
            elif strict:
                raise KeyError(f"Unexpected key in state_dict: {k}")

    def _named_refs(self, prefix: str = "") -> dict[str, Tensor]:
        refs: dict[str, Tensor] = {}
        for k, b in self._buffers.items():
            refs[prefix + k] = b
        for k, p in self._parameters.items():
            refs[prefix + k] = p
        for name, m in self._modules.items():
            refs.update(m._named_refs(prefix=f"{prefix}{name}."))
        return refs


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        for idx, layer in enumerate(layers):
            self._modules[str(idx)] = layer

    def __getitem__(self, idx: int) -> Module:
        return self._modules[str(idx)]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self._modules.values():
            x = layer(x)
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        bound = 1.0 / np.sqrt(in_features)
        w = np.random.uniform(-bound, bound, size=(out_features, in_features)).astype(np.float32)
        b = np.random.uniform(-bound, bound, size=(out_features,)).astype(np.float32)
        self.weight = Parameter(w)
        self.bias = Parameter(b)

    def forward(self, x: Tensor) -> Tensor:
        x_arr = x._arr
        w_arr = self.weight._arr
        b_arr = self.bias._arr
        out = np.matmul(x_arr, w_arr.T) + b_arr
        need_grad = True

        def _bw(g: np.ndarray) -> None:
            g_flat = g.reshape(-1, self.out_features)
            x_flat = x_arr.reshape(-1, self.in_features)
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = Tensor(np.zeros_like(w_arr))
                self.weight.grad._arr += np.matmul(g_flat.T, x_flat)
            if self.bias.requires_grad:
                if self.bias.grad is None:
                    self.bias.grad = Tensor(np.zeros_like(b_arr))
                self.bias.grad._arr += g_flat.sum(axis=0)
            if x.requires_grad or x._grad_fn is not None:
                gx = np.matmul(g, w_arr)
                x._backward_pass(gx)

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw)


class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Parameter(np.ones((normalized_shape,), dtype=np.float32))
        self.bias = Parameter(np.zeros((normalized_shape,), dtype=np.float32))

    def forward(self, x: Tensor) -> Tensor:
        x_arr = x._arr
        mean = x_arr.mean(axis=-1, keepdims=True)
        var = x_arr.var(axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x_arr - mean) * std_inv
        w_arr = self.weight._arr
        b_arr = self.bias._arr
        out = x_hat * w_arr + b_arr

        def _bw(g: np.ndarray) -> None:
            g_flat = g.reshape(-1, self.normalized_shape)
            xh_flat = x_hat.reshape(-1, self.normalized_shape)
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = Tensor(np.zeros_like(w_arr))
                self.weight.grad._arr += (g_flat * xh_flat).sum(axis=0)
            if self.bias.requires_grad:
                if self.bias.grad is None:
                    self.bias.grad = Tensor(np.zeros_like(b_arr))
                self.bias.grad._arr += g_flat.sum(axis=0)
            if x.requires_grad or x._grad_fn is not None:
                N = self.normalized_shape
                dx_hat = g * w_arr
                gx = (1.0 / N) * std_inv * (
                    N * dx_hat
                    - dx_hat.sum(axis=-1, keepdims=True)
                    - x_hat * (dx_hat * x_hat).sum(axis=-1, keepdims=True)
                )
                x._backward_pass(gx.astype(np.float32))

        return Tensor(out, requires_grad=True, _grad_fn=_bw)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        x_arr = x._arr
        out = np.maximum(0.0, x_arr)
        need_grad = x.requires_grad or (x._grad_fn is not None)

        def _bw(g: np.ndarray) -> None:
            x._backward_pass((g * (x_arr > 0)).astype(np.float32))

        return Tensor(out, requires_grad=need_grad, _grad_fn=_bw if need_grad else None)


class _Functional:
    @staticmethod
    def smooth_l1_loss(input_t: Tensor, target_t: Tensor, beta: float = 1.0) -> Tensor:
        diff = input_t._arr - target_t._arr
        abs_diff = np.abs(diff)
        loss_arr = np.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
        out = loss_arr.mean()
        n = diff.size

        def _bw(g: np.ndarray) -> None:
            grad_in = np.where(abs_diff < beta, diff / beta, np.sign(diff)) * (g / n)
            if input_t.requires_grad or input_t._grad_fn is not None:
                input_t._backward_pass(grad_in.astype(np.float32))

        return Tensor(out, requires_grad=True, _grad_fn=_bw)


class _Utils:
    @staticmethod
    def clip_grad_norm_(parameters: list[Parameter], max_norm: float) -> float:
        total_sq = 0.0
        for p in parameters:
            if p.grad is not None:
                total_sq += float(np.sum(p.grad._arr ** 2))
        total_norm = float(np.sqrt(total_sq))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad._arr *= scale
        return total_norm


class _NNNamespace:
    Module = Module
    Sequential = Sequential
    Linear = Linear
    LayerNorm = LayerNorm
    ReLU = ReLU
    Parameter = Parameter
    functional = _Functional
    utils = _Utils


nn = _NNNamespace()


class Adam:
    def __init__(self, params: list[Parameter], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p._arr) for p in self.params]
        self.v = [np.zeros_like(p._arr) for p in self.params]

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad._arr.fill(0.0)

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias_corr1 = 1.0 - (b1 ** self.t)
        bias_corr2 = 1.0 - (b2 ** self.t)
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad._arr
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (g * g)
            m_hat = self.m[i] / bias_corr1
            v_hat = self.v[i] / bias_corr2
            p._arr -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class _OptimNamespace:
    Adam = Adam


optim = _OptimNamespace()


class _TestingNamespace:
    @staticmethod
    def assert_close(a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-5) -> None:
        np.testing.assert_allclose(_to_np(a), _to_np(b), rtol=rtol, atol=atol)


testing = _TestingNamespace()


# --- Native PyTorch v2 .pt ZIP Archive Serializer & Deserializer ---

class _RebuildTensorProxy:
    """Picklable proxy that emits `torch._utils._rebuild_tensor_v2` in the pickle stream."""
    def __init__(self, storage_ref, offset: int, shape: tuple[int, ...], stride: tuple[int, ...]):
        self.storage_ref = storage_ref
        self.offset = offset
        self.shape = shape
        self.stride = stride

    def __reduce__(self):
        return (
            _torch_rebuild_tensor_v2_ref,
            (self.storage_ref, self.offset, self.shape, self.stride, False, OrderedDict()),
        )


class _FloatStorageMarker:
    pass


class _StorageRef:
    def __init__(self, key: str, numel: int, raw_bytes: bytes):
        self.key = key
        self.numel = numel
        self.raw_bytes = raw_bytes


def _torch_rebuild_tensor_v2_ref(*args):
    pass


def _compute_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


def _attach_state_dict_metadata(sd: OrderedDict) -> OrderedDict:
    meta = OrderedDict([
        ("", {"version": 1}),
        ("net", {"version": 1}),
        ("net.0", {"version": 1}),
        ("net.1", {"version": 1}),
        ("net.2", {"version": 1}),
        ("net.3", {"version": 1}),
        ("net.4", {"version": 1}),
        ("net.5", {"version": 1}),
        ("node_enc", {"version": 1}),
        ("node_enc.0", {"version": 1}),
        ("node_enc.1", {"version": 1}),
        ("gcn_dense", {"version": 1}),
        ("gcn_dense.0", {"version": 1}),
        ("gcn_dense.1", {"version": 1}),
        ("fusion", {"version": 1}),
        ("fusion.0", {"version": 1}),
        ("fusion.1", {"version": 1}),
        ("fusion.2", {"version": 1}),
        ("val_head", {"version": 1}),
        ("val_head.0", {"version": 1}),
        ("val_head.1", {"version": 1}),
        ("val_head.2", {"version": 1}),
        ("adv_head", {"version": 1}),
        ("adv_head.0", {"version": 1}),
        ("adv_head.1", {"version": 1}),
        ("adv_head.2", {"version": 1}),
    ])
    sd._metadata = meta
    return sd


def save(obj: dict[str, Any], path: str | Path) -> None:
    """Saves a checkpoint dict in genuine PyTorch v2 zip format compatible with `torch.load`."""
    p = Path(path)
    prefix = p.stem
    storages: list[_StorageRef] = []

    def _convert(item: Any) -> Any:
        if isinstance(item, (Tensor, np.ndarray)):
            arr = np.ascontiguousarray(_to_np(item), dtype="<f4")
            key = str(len(storages))
            sref = _StorageRef(key=key, numel=int(arr.size), raw_bytes=arr.tobytes())
            storages.append(sref)
            return _RebuildTensorProxy(sref, 0, tuple(arr.shape), _compute_strides(tuple(arr.shape)))
        if isinstance(item, OrderedDict):
            out = OrderedDict((k, _convert(v)) for k, v in item.items())
            return _attach_state_dict_metadata(out)
        if isinstance(item, dict):
            return {k: _convert(v) for k, v in item.items()}
        return item

    converted = _convert(obj)

    class _PyTorchPickler(pickle.Pickler):
        def persistent_id(self, o: Any):
            if isinstance(o, _StorageRef):
                return ("storage", _FloatStorageMarker, o.key, "cpu", o.numel)
            return None

    # Map our helper symbols to exact `torch._utils._rebuild_tensor_v2` and `torch.FloatStorage`
    _torch_rebuild_tensor_v2_ref.__module__ = "torch._utils"
    _torch_rebuild_tensor_v2_ref.__name__ = "_rebuild_tensor_v2"
    _torch_rebuild_tensor_v2_ref.__qualname__ = "_rebuild_tensor_v2"
    _FloatStorageMarker.__module__ = "torch"
    _FloatStorageMarker.__name__ = "FloatStorage"
    _FloatStorageMarker.__qualname__ = "FloatStorage"

    import sys, types
    if "torch" not in sys.modules:
        t_mod = types.ModuleType("torch")
        t_mod.FloatStorage = _FloatStorageMarker
        sys.modules["torch"] = t_mod
    else:
        sys.modules["torch"].FloatStorage = _FloatStorageMarker
    if "torch._utils" not in sys.modules:
        tu_mod = types.ModuleType("torch._utils")
        tu_mod._rebuild_tensor_v2 = _torch_rebuild_tensor_v2_ref
        sys.modules["torch._utils"] = tu_mod
    else:
        sys.modules["torch._utils"]._rebuild_tensor_v2 = _torch_rebuild_tensor_v2_ref

    buf = io.BytesIO()
    pickler = _PyTorchPickler(buf, protocol=2)
    pickler.dump(converted)
    pkl_bytes = buf.getvalue()

    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(f"{prefix}/data.pkl", pkl_bytes)
        z.writestr(f"{prefix}/.format_version", b"1")
        z.writestr(f"{prefix}/.storage_alignment", b"64")
        z.writestr(f"{prefix}/byteorder", b"little")
        for sref in storages:
            z.writestr(f"{prefix}/data/{sref.key}", sref.raw_bytes)
        z.writestr(f"{prefix}/version", b"3\n")
        z.writestr(f"{prefix}/.data/serialization_id", b"1648068021766773648607414413317370965805")


def load(path: str | Path, map_location: Any = "cpu", weights_only: bool = False) -> Any:
    """Loads a PyTorch v2 zip checkpoint archive into `Tensor` objects."""
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
        prefix = names[0].split("/")[0]
        pkl_bytes = z.read(f"{prefix}/data.pkl")

        class _LoadedStorage:
            def __init__(self, raw_bytes: bytes):
                self.arr = np.frombuffer(raw_bytes, dtype="<f4").copy()

        class _PyTorchUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    def _rebuild(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                        count = int(np.prod(size)) if len(size) > 0 else 1
                        arr = storage.arr[storage_offset : storage_offset + count].reshape(size)
                        return Tensor(arr, requires_grad=bool(requires_grad))
                    return _rebuild
                if module == "torch" and name == "FloatStorage":
                    return "FloatStorage"
                return super().find_class(module, name)

            def persistent_load(self, pid):
                _, _storage_type, key, _location, _numel = pid
                raw = z.read(f"{prefix}/data/{key}")
                return _LoadedStorage(raw)

        return _PyTorchUnpickler(io.BytesIO(pkl_bytes)).load()


class _TorchModuleFacade:
    Tensor = Tensor
    dtype = dtype
    float32 = float32
    int64 = int64
    long = long
    bool = bool_
    device = device
    no_grad = staticmethod(no_grad)
    set_num_threads = staticmethod(set_num_threads)
    manual_seed = staticmethod(manual_seed)
    as_tensor = staticmethod(as_tensor)
    zeros = staticmethod(zeros)
    ones_like = staticmethod(ones_like)
    full_like = staticmethod(full_like)
    eye = staticmethod(eye)
    where = staticmethod(where)
    cat = staticmethod(cat)
    matmul = staticmethod(matmul)
    save = staticmethod(save)
    load = staticmethod(load)
    nn = nn
    optim = optim
    testing = testing


torch = _TorchModuleFacade()
