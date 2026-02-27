from __future__ import annotations

import numpy as np
from typing import (Any, Iterable, Callable)

class Tensor:
    def __init__(
            self, 
            data: Any, 
            requires_grad: bool = False,
            _children: Iterable["Tensor"] = (),
            op: str = ""
            ) -> None:
        self.data: np.ndarray = np.asanyarray(data, dtype=np.float64)
        self.requires_grad: bool = bool(requires_grad)
        self.grad: np.ndarray | None = None
        self._prev: set[Tensor] = set(_children)
        self._backward: Callable[[], None] = lambda: None
        self.op: str = op

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape
    
    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype
    
    def __repr__(self) -> str:
        return(
            f"Tensor(data={self.data!r}, requires_grad={self.requires_grad}, "
            f"op={self.op!r})"
        )
    
    @staticmethod
    def _ensure_tensor(other: Any) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)
    
    # ops
    def __add__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return Tensor(self.data + other.data, _children=(self, other), op="add")
    
    def __radd__(self, other: Any) -> "Tensor":
        return self + other
    
    def __neg__(self) -> "Tensor":
        return Tensor(-self.data, _children=(self,), op="neg")
    
    def __sub__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return Tensor(self.data - other.data, _children=(self, other), op="sub")
    
    def __rsub__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return other - self
    
    def __mul__(self, other: Any) -> "Tensor":
        other = self._ensure_tensor(other)
        return Tensor(self.data * other.data, _children=(self, other), op="mul")
    
    def __rmul__(self, other: Any) -> "Tensor":
        return self * other
    
    def pow(self, exponent: float) -> "Tensor":
        return Tensor(self.data ** exponent, _children=(self,), op="pow")
    
    def sum(self) -> "Tensor":
        return Tensor(self.data.sum(), _children=(self,), op="sum")
    
    def mean(self) -> "Tensor":
        return Tensor(self.data.mean(), _children=(self,), op="mean")