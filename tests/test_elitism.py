import numpy as np
import numpy.testing as npt
import pytest
from XLEMOO.elitism import simple_elitism
from desdeo_emo.population import Population
from XLEMOO.LEMOO import DummyPopulation


@pytest.mark.elitism
def test_simple_elitism_errors():
    pop1 = DummyPopulation(np.ones((100, 3)), None)
    pop2 = DummyPopulation(np.zeros((40, 3)), None)
    pop3 = DummyPopulation(np.ones((100, 2)), None)

    fitness_1 = np.array(range(0, 100))
    fitness_2 = np.array(range(50, 150))

    # should be fine
    simple_elitism(pop1, fitness_1, pop2, fitness_2)

    # mismatch pop size
    with pytest.raises(ValueError) as err:
        simple_elitism(pop1, fitness_1, pop3, fitness_2)
