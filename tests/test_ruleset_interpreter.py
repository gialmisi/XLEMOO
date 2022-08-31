import numpy as np
import numpy.testing as npt
import pytest
from XLEMOO.ruleset_interpreter import instantiate_ruleset_rules


@pytest.mark.ruleset
def test_instantiate_ruleset_rules():
    dummy_dict = {("X_1", "<"): "1000", ("X_2", ">="): "300"}
    res = instantiate_ruleset_rules(dummy_dict)

    npt.assert_almost_equal(res, np.array([0]))
