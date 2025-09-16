import numpy as np
import pytest

from lanthanide import Lanthanide, Coupling


def is_orthogonal(A):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2

    N = A.shape[1]
    identity = np.eye(N)
    return np.allclose(identity, A.T @ A, atol=1e-12)


@pytest.mark.parametrize("num", range(1, 14))
def test_orthogonal(num):
    coupling = Coupling.SLJ
    with Lanthanide(num, coupling) as ion:
        V = ion.states(Coupling.SLJM).transform
        assert is_orthogonal(V)
        V = ion.states(Coupling.SLJ).transform
        assert is_orthogonal(V)
