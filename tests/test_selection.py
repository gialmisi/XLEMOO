import pytest

import numpy.testing as npt
import numpy as np
from XLEMOO.selection import SelectNBest


@pytest.mark.selection
def test_select_n_best():
    n_desired = 3
    selection_op = SelectNBest(None, n_desired)
    fitnesses = np.atleast_2d([10, 20, 5, 1, -1, 15]).T
    selected = selection_op.do(None, fitnesses)

    assert len(selected) == n_desired
    npt.assert_allclose(selected, np.atleast_2d([4, 3, 2]).T)

    # n_desired bigger than pop size
    n_desired = 10
    selection_op = SelectNBest(None, n_desired)
    fitnesses = np.atleast_2d([10, 20, 5, 1, -1, 15]).T
    selected = selection_op.do(None, fitnesses)

    assert len(selected) == 6
    npt.assert_allclose(selected, np.atleast_2d([4, 3, 2, 0, 5, 1]).T)
