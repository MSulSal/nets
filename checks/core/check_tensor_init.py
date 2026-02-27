import numpy as np

from core.tensor.tensor import Tensor

def check(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)

def run() -> None:
    # Scalar input
    t0 = Tensor(3.14)
    check(isinstance(t0.data, np.ndarray), "t0.data should be ndarray")
    check(t0.dtype == np.float64, "t0.dtype should be float64")
    check(t0.shape == (), "t0.shape should be ()")
    check(t0.grad is None, "t0.grad should be none")
    check(len(t0._prev) == 0, "t0._prev should be 0")
    check(t0.op == "", "t0.op should be ''")

    # List input
    t1 = Tensor([1, 2, 3], requires_grad=True)
    check(isinstance(t1.data, np.ndarray), "t1.data should be ndarray")
    check(t1.dtype == np.float64, "t1.dtype should be float64")
    check(t1.shape == (3,), "t1.shape should be (3,)")
    check(t1.requires_grad is True, "t1.requires_grad should be True")
    check(t1.grad is None, "t1.grad should be None")
    check(len(t1._prev) == 0, "t1._prev should be 0")

    # ndarray input
    arr = np.array([1.0, 2.0], dtype=np.float32)
    t2 = Tensor(arr, requires_grad=False)
    check(t2.dtype == np.float64, "t2.dtype should be np.float64")
    check(t2.requires_grad is False, "t2.requires_grad should be False")


if __name__ == "__main__":
    run()
    print("check_tensor_init: PASS")
