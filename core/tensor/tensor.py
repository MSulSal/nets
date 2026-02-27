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