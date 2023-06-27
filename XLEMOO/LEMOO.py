from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Callable, Optional, Union, Dict

from desdeo_emo.population import (
    SurrogatePopulation,
    create_new_individuals,
    Population,
)
from desdeo_emo.recombination import SBX_xover, BP_mutation
from desdeo_emo.selection import TournamentSelection
from desdeo_problem.problem import MOProblem

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from imodels import SlipperClassifier, BoostedRulesClassifier, SkopeRulesClassifier

from XLEMOO.tree_interpreter import find_all_paths, instantiate_tree_rules
from XLEMOO.ruleset_interpreter import (
    extract_slipper_rules,
    instantiate_ruleset_rules,
    extract_skoped_rules,
)
from XLEMOO.selection import SelectNBest


class CrossOverOP(ABC):
    """An abstract class that defines the general interface for a crossover operator."""

    @abstractmethod
    def do(self, population: Population, mating_pop_ids: List[int]) -> np.ndarray:
        """The crossover operator should have a 'do' method as specified in the abstract class.

        Args:
            population (Population): A population of solutions.
            mating_pop_ids (List[int]): The indices selected for mating.

        Returns:
            np.ndarray: The offspring individuals resulting from applying the mating operator.
        """
        pass


CrossOverOP.register(SBX_xover)


class MutationOP(ABC):
    """An abstract class that defines the general interface for a mutation operator."""

    @abstractmethod
    def do(self, offsprings: np.ndarray) -> np.ndarray:
        """The mutation operator should have a 'do' method as specified in the abstract class.

        Args:
            offsprings (np.ndarray): The offspring (or individuals from a population) to be mutated.

        Returns:
            np.ndarray: The mutated offsprings.
        """
        pass


MutationOP.register(BP_mutation)


class SelectionOP(ABC):
    """An abstract class that defines the general interface for a selection operator."""

    @abstractmethod
    def do(self, pop: Population, fitness: np.ndarray) -> List[int]:
        """The selection operator should have a 'do' method as specified in the abstract class.

        Args:
            pop (Population): The population the selection operator is applied to.
            fitness (np.ndarray): The fitness of the individuals in the population.
                It is assumed the fitnesses are related to the individuals inthe population by index.

        Returns:
            List[int]: A list of indices indicating which individuals in the population have been selected.
        """
        pass


SelectionOP.register(TournamentSelection)
SelectionOP.register(SelectNBest)


class ElitismOP(ABC):
    """An abstract class that defines the general interface for an elitism operator."""

    @abstractmethod
    def do(
        self,
        pop1: Population,
        pop1_fitness: np.ndarray,
        pop2: Population,
        pop2_fitness: np.ndarray,
    ) -> np.ndarray:
        """The elitism operator should have a 'do' method as specified in the abstract class.
        Two populations are compared by the elitism operator from which the best individuals,
        according to their fitness values, are selected and returned.

        Args:
            pop1 (Population): The first population.
            pop1_fitness (np.ndarray): The fitness values in the first population.
                It is assumed the fitnesses are related to the individuals inthe population by index.
            pop2 (Population): The second population.
            pop2_fitness (np.ndarray): The fitness values in the second population.
                It is assumed the fitnesses are related to the individuals inthe population by index.

        Returns:
            np.ndarray: The elite individuals from resulting by applying the elitism operator.
        """
        pass


class MLModel(ABC):
    """An abstract class that defines the general interface for a machine learning model used in a LEMOO method's
    learning mode.

    """

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray):
        """The machine learning model should have a 'fit' method which allows training the model based on
        data.

        Args:
            X (np.ndarray): Training samples. E.g., n-dimensional vectors of real values floats.
            Y (np.ndarray): The targets. E.g., a 1-dimensional vector of 0s and 1s for binary classificaiton.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """The machine learning model should have a 'predict' method that can be used to predict the, e.g., class
        in binary classification, or a sample, or samples.

        Args:
            X (np.ndarray): Sample or sample a class should be predicted for.

        Returns:
            np.ndarray: The predicted classes for the samples.
        """
        pass


# Register supported ML models here
MLModel.register(DecisionTreeClassifier)
MLModel.register(SlipperClassifier)
MLModel.register(BoostedRulesClassifier)
MLModel.register(SkopeRulesClassifier)


class DummyPopulation:
    """Used for testing."""

    def __init__(self, individuals: np.ndarray, problem: MOProblem):
        self.individuals = individuals
        self.problem = problem


@dataclass
class EAParams:
    """A data class to store and pass parameter values related to the Darwinian mode of a LEMOO method.

    Args:
        population_size (int): The size of the population to be evolved.
        cross_over_op (CrossOverOP): The crossover operator.
        mutation_op (MutationOP): The mutation operator.
        selection_op (SelectionOP): The selection operator.
        population_init_design (str): Initialization strategy of the populatino. Should be 'Random' for random
            or 'LHSDesign' for latin hypercube sampling.
        iterations_per_cycle (int): How many times a population is evolved in a Darwinian mode before switching to a
            learning mode. Only relevant when a LEMOO method is run using the ``run_iterations`` method.
    """

    population_size: int
    cross_over_op: CrossOverOP
    mutation_op: MutationOP
    selection_op: SelectionOP
    population_init_design: str
    iterations_per_cycle: int


@dataclass
class MLParams:
    """A data class to store and pass parameter values related to the learning mode of a LEMOO method.

    Args:
        H_split (float): The splitting ratio of 'high performing' population members. E.g., a H_split of 0.10 means
            that 10% of the best performing population members are labeled as high perfomring during a learning mode.
        L_split (float): Same as H_split, but for the 'low performing' population members.
        ml_model (MLModel): The machine learning model to be used in a learning mode.
        instantation_factor (float): A multiplier used to determine how many new population members are instantiated
            in a learning mode after hypothesis forming. E.g., a factor of 2.0 means that 2.0*N_population new population
            members are instantiated based on the learned hypothesis, where N_population is the size of the population
            in the LEMOO method.
        generation_lookback (int): How many older generations to consider in a learning mode. E.g., a lookback of 5
            means that the 5 most recent population are considered when forming a hypothesis.
        ancestral_recall (int): This is like generation_lookback, but considers a specific number of the oldest
            populations. E.g., a recall of 5 will consider the five first populations.
        unique_only (bool): Whether to consider unique population memebrs only when learning a hypothesis.
        iterations_per_cycle (int): How many times a population is ''evolved'' in a learning mode before switching to
            a Darwinian mode. A good default is 1. Only relevant when a LEMOO method is run using the ``run_iterations`` method.
    """

    H_split: float
    L_split: float
    ml_model: MLModel
    instantation_factor: float
    generation_lookback: int
    ancestral_recall: int
    unique_only: bool
    iterations_per_cycle: int


@dataclass
class LEMParams:
    """A data class to store and pass general parameter values of a LEMOO method.

    Args:
        use_ml (bool): Whether to engage in a learning mode or not.
        use_darwin (bool): Whether to engage in a Darwinian mode or not.
        fitness_inicator (Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]): A fitness function that
            accepts a 2d numpy array that represents the population members in the objective space of the problem.
            Optionally, the decision variable values may also be passed for each population member.
        ml_probe (int): The maximum time a learning mode is executed when a threshold is not reached.
            Only relevant when a LEMOO method is executed using the ``run`` method.
        darwin_probe (int): like ``ml_probe`` but for a Darwininan mode.
        ml_threshold (float): The relative improvement of the best population member's fitness expected before switching out
            of a learning mode. E.g., a threshold of 1.05 means that executing a learning mode stops when the best population
            member has improved by 5% when compared to the previous population's best member.
        darwin_threshold (float): Like ``ml_threshold`` but for a Darwinian mode.
        total_iterations (int): Overall maximum number of iterations to be run. Only relevant when the ``run_iterations``
            method of a LEMOO model is used.

    """

    use_ml: bool
    use_darwin: bool
    fitness_indicator: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]
    ml_probe: int
    darwin_probe: int
    ml_threshold: float
    darwin_threshold: float
    total_iterations: int


@dataclass
class PastGeneration:
    """A helper data class representing past generation with the individuals (decison space) and their
    corresponsing objective fitness values and fitness function values.

    Args:
        individuals (np.ndarray): The individuals of a population in the decision variable space.
        individuals (np.ndarray): The individuals of a population in the objective function space.
        fitness_fun_values (np.ndarray): The fitness function values of each individual.

    """

    individuals: np.ndarray
    objectives_fitnesses: np.ndarray
    fitness_fun_values: np.ndarray


class LEMOO:
    def __init__(
        self,
        problem: MOProblem,
        lem_params: LEMParams,
        ea_params: EAParams,
        ml_params: MLParams,
    ):
        """A class to define LEMOO models.

        Args:
            problem (MOProblem): The multiobjective optimization problem to be solved as defined in the DESDEO
                framework.
            lem_params (LEMParams): A dataclass with parameters relevant to the LEM part of the LEMOO method.
                See the dataclass' documentation for additional details.
            ea_params (EAParams): A dataclass with parameters relevant to the Darwin mode of the LEMOO method.
                See the dataclass' documentation for additional details.
            ml_params (MLParams): A dataclass with parameters relevant to the learning mode of the LEMOO method.
                See the dataclass' documentation for additional details.

        Attributes:
            current_ml_model (MLModel): The current machine learing model employed in the learning mode.
            _populatoin (Union[None, SurrogatePopulation]): The current population of solutions.
            _best_fitness_fun_value (float): The current best fitness value found.
            _generation_history(List[PastGeneration]): A list to keep track fo the population histories during the
                executiong of the LEMOO model.

        """
        self._problem: MOProblem = problem
        self._lem_params: LEMParams = lem_params
        self._ea_params: EAParams = ea_params
        self._ml_params: MLParams = ml_params
        self.current_ml_model: MLModel = ml_params.ml_model
        self._population: Union[None, SurrogatePopulation] = None
        self._best_fitness_fun_value: float = np.inf  # lower is better!

        # initialize the population and the evolutionary operators
        self._generation_history: List[PastGeneration] = []
        self.initialize_population()

    def initialize_population(self) -> None:
        """Use the defined initialization design to init a new population, add the initial population to the history of populations."""
        initial_population = create_new_individuals(
            self._ea_params.population_init_design,
            self._problem,
            pop_size=self._ea_params.population_size,
        )

        self._population = SurrogatePopulation(
            self._problem,
            self._ea_params.population_size,
            initial_population,
            self._ea_params.cross_over_op,
            self._ea_params.mutation_op,
            None,
        )

        self.add_population_to_history()
        self.update_best_fitness()

        return

    def reset_population(self) -> None:
        """Reset the current population. Do not add the new population to history."""
        initial_population = create_new_individuals(
            self._ea_params.population_init_design,
            self._problem,
            pop_size=self._ea_params.population_size,
        )

        self._population = SurrogatePopulation(
            self._problem,
            self._ea_params.population_size,
            initial_population,
            self._ea_params.cross_over_op,
            self._ea_params.mutation_op,
            None,
        )

        return

    def reset_generation_history(self) -> None:
        """Reset the population history."""
        self._generation_history = []

    def update_best_fitness(self) -> bool:
        """Find the best fitness value in the current population and update the value stored in
        self._best_fitness_fun_value if the found fitness is better.

        Returns:
            bool: True if the best fitness was updated, otherwise False.

        """
        fitness_fun_values = self._lem_params.fitness_indicator(self._population.fitness, self._population.individuals)
        min_value = np.min(fitness_fun_values)

        if min_value < self._best_fitness_fun_value:
            self._best_fitness_fun_value = min_value

            return True

        else:
            return False

    def add_population_to_history(
        self,
        individuals: Optional[np.ndarray] = None,
        objectives_fitnesses: Optional[np.ndarray] = None,
    ) -> None:
        """Add a population to the history of the LEMOO model.

        Args:
            individuals (Optional[np.ndarray], optional): The decision variables values of the population members. Defaults to None.
            objectives_fitnesses (Optional[np.ndarray], optional): The corresponding objective function values of
            the population members. Defaults to None.

        Note:
            If both arguments are ``None``, then the current population in the LEMOO model is added to the history.

        """
        if individuals is not None and objectives_fitnesses is not None:
            # add supplied individuals and fitnesses to history
            fitness_fun_values = self._lem_params.fitness_indicator(objectives_fitnesses, individuals)
            gen = PastGeneration(individuals, objectives_fitnesses, fitness_fun_values)
            self._generation_history.append(gen)

            return

        else:
            # add current population to history
            fitness_fun_values = self._lem_params.fitness_indicator(
                self._population.fitness, self._population.individuals
            )
            gen = PastGeneration(
                self._population.individuals,
                self._population.fitness,
                fitness_fun_values,
            )
            self._generation_history.append(gen)

            return

    def update_population(self, new_individuals: np.ndarray) -> None:
        """Replace the current population of the LEMOO model with a new one.

        Args:
            new_individuals (np.ndarray): The new population members in the decision variable space.

        """
        self._population.delete(np.arange(len(self._population.individuals)))
        self._population.add(new_individuals)

        return

    def collect_n_past_generations(self, n: int, ancestral_recall: int = 0, unique_only=False):
        """Collect the n past generations into a single numpy array for easier handling.
        Returns the collected individuals, objective fitness values, and fitness function values.

        Args:
            n (int): number of past generations to collect.

        Returns:
            Tuple[nd.array, np.ndarray, np.ndarray]: A tuple with the collected individuals, objective fitness values, and
            fitness function values. Each array is 2-dimensional.

        """

        if n > len(self._generation_history):
            # n bigger than available history, truncate to length of history
            n = len(self._generation_history)
            # also set  ancestral recall to zero since it will be included in any case
            ancestral_recall = 0

        past_slice = self._generation_history[-n:]

        if ancestral_recall > 0:
            ancestral_slice = self._generation_history[0:ancestral_recall]

            past_slice = ancestral_slice + past_slice

        individuals = np.concatenate([gen.individuals for gen in past_slice])
        objectives_fitnesses = np.concatenate([gen.objectives_fitnesses for gen in past_slice])
        fitness_fun_values = np.concatenate([gen.fitness_fun_values for gen in past_slice])

        if unique_only:
            # return only unique individuals,
            _, unique_inds = np.unique(individuals, return_index=True, axis=0)
            individuals = individuals[unique_inds]
            objectives_fitnesses = objectives_fitnesses[unique_inds]
            fitness_fun_values = fitness_fun_values[unique_inds]

        return (
            np.atleast_2d(individuals),
            np.atleast_2d(objectives_fitnesses),
            np.atleast_2d(fitness_fun_values),
        )

    def darwinian_mode(self) -> None:
        """Execute Darwinian mode. The size of the population can vary."""
        # mate
        offspring = self._population.mate()
        self._population.add(offspring, False)

        fitness_fun_values = self._lem_params.fitness_indicator(self._population.fitness, self._population.individuals)
        selected = self._ea_params.selection_op.do(self._population, fitness_fun_values)

        self._population.keep(selected)

        return

    def learning_mode(self) -> None:
        """Execute learning mode."""
        instantiation_factor = self._ml_params.instantation_factor

        if self._ml_params.generation_lookback <= 0:
            lookback_n = len(self._generation_history)

        else:
            # if generation lookback is specified, then use at most that many past generations from the current generation
            lookback_n = (
                self._ml_params.generation_lookback
                if self._ml_params.generation_lookback < len(self._generation_history)
                else len(self._generation_history)
            )

        (
            all_individuals,
            all_objectives_fitnesses,
            all_fitness_fun_values,
        ) = self.collect_n_past_generations(lookback_n, unique_only=self._ml_params.unique_only)

        if not isinstance(self._ml_params.ml_model, MLModel):
            raise TypeError(f"MLModel of type {type(self._ml_params.ml_model)} is not supported in learning mode.")

        sorted_indices = np.argsort(np.squeeze(all_fitness_fun_values))

        # formulate the H and L groups
        # calculate the cut-off indices for both groups, the split might not be exactly
        # accurate due to the cast to int
        # if H or L split are more than 1, use that many samples in both groups, respectively
        if self._ml_params.H_split < 1.0:
            h_split_id = int(self._ml_params.H_split * len(sorted_indices))
        else:
            if self._ml_params.H_split > (all_individuals.shape[0] / 2):
                # defualt to half if H is alrger than half of the individuals
                h_split_id = int(all_individuals.shape[0] / 2)
            else:
                h_split_id = int(self._ml_params.H_split)

        if self._ml_params.L_split < 1.0:
            l_split_id = int(self._ml_params.L_split * len(sorted_indices))
        else:
            if self._ml_params.L_split > (all_individuals.shape[0] / 2):
                # default to half if L is larger than half of all individuals
                l_split_id = int(all_individuals.shape[0] / 2)
            else:
                l_split_id = int(self._ml_params.L_split)

        # because the indices are now sorted, we can just pick the top best and bottom worst
        # and set them as the H and L groups
        h_indices = sorted_indices[0:h_split_id]
        l_indices = sorted_indices[-l_split_id:][::-1]  # reversing might not really be needed here

        # pick the individuals according to the calculated indices
        h_group = all_individuals[h_indices]
        l_group = all_individuals[l_indices]

        # create training data to train an ML model:
        x_train = np.vstack((h_group, l_group))

        if isinstance(self._ml_params.ml_model, DecisionTreeClassifier):
            # for tree, 1 good, -1 bad
            y_train = np.hstack(
                (
                    np.ones(len(h_group), dtype=int),
                    -1 * np.ones(len(l_group), dtype=int),
                )
            )
        elif (
            isinstance(self._ml_params.ml_model, SlipperClassifier)
            or isinstance(self._ml_params.ml_model, BoostedRulesClassifier)
            or isinstance(self._ml_params.ml_model, SkopeRulesClassifier)
        ):
            # 1: target, 0: other
            y_train = np.hstack((np.ones(len(h_group), dtype=int), np.zeros(len(l_group), dtype=int)))
        else:
            raise TypeError(f"MLModel of type {type(self._ml_params.ml_model)} is not supported in learning mode.")

        n_to_instantiate = int(all_individuals.shape[0] * instantiation_factor)

        # TODO: this is tree specific, check if tree or ruleset!
        #
        if isinstance(self._ml_params.ml_model, DecisionTreeClassifier):
            # do tree stuff
            classifier = self._ml_params.ml_model.fit(x_train, y_train)
            self.current_ml_model = classifier

            # after training the model, it must be used to instantiate new individuals according to
            # the H-group description. I.e., find new individuals that have a classification of 1.
            # by default, instantiate twice as many new individuals as peresent in the whole history.
            # this factor is controlled by the instantiation_factor argument (default 2)

            paths = find_all_paths(classifier)
            instantiated = instantiate_tree_rules(
                paths,
                self._problem.n_of_variables,
                self._problem.get_variable_bounds(),
                n_to_instantiate,
                1,
            )

            # reshape to have just a list of new individuals
            instantiated = instantiated.reshape((-1, instantiated.shape[2]))

        elif (
            isinstance(self._ml_params.ml_model, SlipperClassifier)
            or isinstance(self._ml_params.ml_model, BoostedRulesClassifier)
            or isinstance(self._ml_params.ml_model, SkopeRulesClassifier)
        ):
            # do ruleset stuff
            classifier = self._ml_params.ml_model.fit(x_train, y_train)
            self.current_ml_model = classifier

            if isinstance(self._ml_params.ml_model, SlipperClassifier) or isinstance(
                self._ml_params.ml_model, BoostedRulesClassifier
            ):
                # Slipper rules
                ruleset, weights = extract_slipper_rules(classifier)
            elif isinstance(self._ml_params.ml_model, SkopeRulesClassifier):
                # Skoped rules
                ruleset, weights = extract_skoped_rules(classifier)
            else:
                raise ValueError(
                    f"The current classifier {self._ml_params.ml_model} has no supported rule extractions."
                )

            instantiated = instantiate_ruleset_rules(
                ruleset,
                weights,
                self._problem.n_of_variables,
                self._problem.get_variable_bounds(),
                n_to_instantiate,
            )

        else:
            raise TypeError(f"MLModel of type {type(self._ml_params.ml_model)} is not supported in learning mode.")

        # mix with the existing H-group
        instantiated_and_h = np.vstack((h_group, instantiated))

        # compute objectives fitnesses
        objective_fitnesses_new = self._problem.evaluate(instantiated_and_h).fitness

        # compute fitness fun values
        fitness_fun_values_new = self._lem_params.fitness_indicator(objective_fitnesses_new, instantiated_and_h)

        # sort the individuals according to their fitness value in ascending order
        sorted_indices_new = np.argsort(np.squeeze(fitness_fun_values_new))

        # pic the best fraction that is the same size as the last population
        selected_indices = sorted_indices_new[0 : self._population.individuals.shape[0]]
        selected_individuals = instantiated_and_h[selected_indices]

        # replace the current population with the new selected individuals
        self.update_population(selected_individuals)

        return

    def check_condition_best(self, n_lookback: int, threshold: float) -> bool:
        """
        Check whether the Darwin termination criterion is met. In the past n_lookback iterations.

        Return True and update current best value if the condition is met, just return False otherwise.

        Args:
            n_lookback (int): How many generations to look back to.
            threshold (float): The relative improvement expected in regard to the best fitness value. E.g.,
                a threshold of 1.05 means a 5% improvement is expected.

        Returns:
            bool: True if the threshold is met. False otherwise.

        """
        # get past generations
        (
            past_individuals,
            past_objectives_fitnesses,
            past_fitness_fun_values,
        ) = self.collect_n_past_generations(n_lookback)

        # find index of best (lowest fitness value) individual
        best_idx = np.argmin(past_fitness_fun_values)

        # check condition
        if ((past_fitness_fun_values[best_idx] / self._best_fitness_fun_value)) < threshold:
            self._best_fitness_fun_value = past_fitness_fun_values[best_idx][0]

            return True

        else:

            return False

    def run(self) -> Dict:
        """
        Run the LEMOO model. Switching between the Darwinian mode and learning mode happends
        when the fitness of the best population member has improved past a threshold or when a maximum number of
        iterations has been executed in a mode.

        Returns:
            Dict: A dictionary with counters indicating how many times the Darwinian and learning modes have been
                executed, and total iterations.

        """
        # counters:
        counters = {"darwin_mode": 0, "learning_mode": 0}

        keep_running = True

        while keep_running:
            improved_in_learning = False
            improved_in_darwin = False

            # start in ML mode
            if self._lem_params.use_ml:
                learning_iters = 0
                while learning_iters < self._lem_params.ml_probe:
                    # always assume previous population has been saved to history before
                    self.learning_mode()
                    self.add_population_to_history()
                    counters["learning_mode"] += 1
                    learning_iters += 1

                    # check generations saved so far in learning more if they meet the termination criterion
                    if self.check_condition_best(learning_iters, self._lem_params.ml_threshold):
                        improved_in_learning = True
                        break
            else:
                improved_in_learning = False

            # do Darwinian mode
            if self._lem_params.use_darwin:
                darwin_iters = 0
                while darwin_iters < self._lem_params.darwin_probe:
                    # we assume the population has been saved in a previous iteration
                    self.darwinian_mode()
                    self.add_population_to_history()
                    counters["darwin_mode"] += 1
                    darwin_iters += 1

                    # iterate until condition is True
                    # check the generations saved so far in Darwin mode, that is
                    # why we keep the darwin_iters counter.
                    if self.check_condition_best(darwin_iters, self._lem_params.darwin_threshold):
                        improved_in_darwin = True
                        break
            else:
                improved_in_darwin = False

            if not improved_in_learning and not improved_in_darwin:
                keep_running = False

        return counters

    def run_iterations(self) -> Dict:
        """
        Run the LEMOO model. The Darwinian mode and learning mode are always executed for a set number of
        iterations. Thresholds are ignored.

        Returns:
            Dict: A dictionary with counters indicating how many times the Darwinian and learning modes have been
                executed, and total iterations.

        """
        # counters:
        counters = {"darwin_mode": 0, "learning_mode": 0}

        total_iterations: int = 0

        while total_iterations < self._lem_params.total_iterations:

            # Darwinian (ea) mode
            if self._lem_params.use_darwin:
                iterations_in_ea: int = 0

                while iterations_in_ea < self._ea_params.iterations_per_cycle:
                    # we assume the population has been saved in a previous iteration
                    self.darwinian_mode()
                    self.add_population_to_history()
                    counters["darwin_mode"] += 1
                    iterations_in_ea += 1

            # start in ML mode
            if self._lem_params.use_ml:
                iterations_in_ml: int = 0

                while iterations_in_ml < self._ml_params.iterations_per_cycle:
                    # always assume previous population has been saved to history before
                    self.learning_mode()
                    self.add_population_to_history()
                    counters["learning_mode"] += 1
                    iterations_in_ml += 1

            total_iterations += 1

        counters.update({"total_iterations": total_iterations})

        return counters


if __name__ == "__main__":
    print(issubclass(SBX_xover, CrossOverOP))
