import numpy as np

from core.tensor.tensor import Tensor


def check(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def check_close(actual: np.ndarray, expected: np.ndarray, msg: str) -> None:
    if not np.allclose(actual, expected):
        raise AssertionError(f"{msg} | actual={actual} expected={expected}")


def run() -> None:
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])

    # add / radd
    c = a + b
    check(c.op == "add", "add op label should be 'add'")
    check(len(c._prev) == 2, "add should have 2 parents")
    check(a in c._prev and b in c._prev, "add parents should include both operands")
    check_close(c.data, np.array([5.0, 7.0, 9.0]), "tensor + tensor result mismatch")

    c2 = 2.0 + a
    check(c2.op == "add", "radd op label should be 'add'")
    check_close(c2.data, np.array([3.0, 4.0, 5.0]), "scalar + tensor result mismatch")

    # neg
    n = -a
    check(n.op == "neg", "neg op label should be 'neg'")
    check(len(n._prev) == 1 and a in n._prev, "neg should have exactly one parent")
    check_close(n.data, np.array([-1.0, -2.0, -3.0]), "neg result mismatch")

    # sub / rsub
    s = a - 1.5
    check(s.op == "sub", "sub op label should be 'sub'")
    check_close(s.data, np.array([-0.5, 0.5, 1.5]), "tensor - scalar result mismatch")

    s2 = 10.0 - a
    check(s2.op == "sub", "rsub op label should be 'sub'")
    check_close(s2.data, np.array([9.0, 8.0, 7.0]), "scalar - tensor result mismatch")

    # mul / rmul
    m = a * b
    check(m.op == "mul", "mul op label should be 'mul'")
    check(len(m._prev) == 2, "mul should have 2 parents")
    check_close(m.data, np.array([4.0, 10.0, 18.0]), "tensor * tensor result mismatch")

    m2 = 3.0 * a
    check(m2.op == "mul", "rmul op label should be 'mul'")
    check_close(m2.data, np.array([3.0, 6.0, 9.0]), "scalar * tensor result mismatch")

    # pow
    p = a.pow(2)
    check(p.op == "pow", "pow op label should be 'pow'")
    check(len(p._prev) == 1 and a in p._prev, "pow should have exactly one parent")
    check_close(p.data, np.array([1.0, 4.0, 9.0]), "pow result mismatch")

    # sum
    su = a.sum()
    check(su.op == "sum", "sum op label should be 'sum'")
    check(su.shape == (), "sum should return scalar-shaped tensor")
    check_close(su.data, np.array(6.0), "sum result mismatch")

    # mean
    me = a.mean()
    check(me.op == "mean", "mean op label should be 'mean'")
    check(me.shape == (), "mean should return scalar-shaped tensor")
    check_close(me.data, np.array(2.0), "mean result mismatch")

    # input tensors should not be mutated by ops
    check_close(a.data, np.array([1.0, 2.0, 3.0]), "a should remain unchanged")
    check_close(b.data, np.array([4.0, 5.0, 6.0]), "b should remain unchanged")


if __name__ == "__main__":
    run()
    print("check_ops_basic: PASS")
