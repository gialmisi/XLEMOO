from .LEMOO import LEMOO
from .ruleset_interpreter import extract_skoped_rules
from desdeo_problem.problem import MOProblem

import numpy as np

def parse_skoped_rules(lemoo: LEMOO, problem: MOProblem) -> dict:
    """Parses the rules from a trained LEMOO model that utilizes skoped rules in its
    learning mode and returns a dict with an entry
    for each variable describing its upper and lower bounds according to the extracted
    rules.

    Args:
        lemoo (LEMOO): A trained LEMOO model.
        problem (MOProblem): A multiobjective optimization problem.

    Returns:
        dict: A dict with each key representing a variable rules have been parsed for.
        Each entry is a dict with keys representing either the '>' or '<=' operator.
        The value behind the keys represented by these operators is the corresponding
        lower or upper bound of the variable, respectively. Example: ``{'x0': {'>': 2.5, <=': 5.2}}``.
    """
    rules, accuracies = extract_skoped_rules(lemoo.current_ml_model)
    problem_lower_bounds = problem.get_variable_lower_bounds()
    problem_upper_bounds = problem.get_variable_upper_bounds()

    rules_for_vars = {f"X_{i}": {">": [problem_lower_bounds[i], -1], "<=": [problem_upper_bounds[i], -1]} for i in range(problem.n_of_variables)}

    for accuracy, rule in zip(accuracies, rules):
        for key in rule:
            var_name = key[0]
            op = key[1]

            # check accuracy
            if rules_for_vars[var_name][op][1] < accuracy:
                # update accuracy
                rules_for_vars[var_name][op][1] = accuracy
                if op == "<=":
                    # tighten rule, if necessary
                    if float(rule[(var_name, op)]) <= rules_for_vars[var_name][op][0]:
                        rules_for_vars[var_name][op][0] = float(rule[(var_name, op)])
                elif op == ">":
                    # tighten rule, if necessary
                    if float(rule[(var_name, op)]) > rules_for_vars[var_name][op][0]:
                        rules_for_vars[var_name][op][0] = float(rule[(var_name, op)])

    return rules_for_vars

def complete_missing_rules(lemoo: LEMOO, rules_for_vars: dict) -> dict:
    """Completes the missing rules contained in a dict. For the contents of the
    dict, see the function :func:`parse_skoped_rules`.
    The completing of the missing rules utilizes the
    final population generated by a trained LEMOO model by taking the lower
    and upper bounds of the population for each variables with missing bounds in the
    rules defined in `rules_for_vars`.

    Args:
        lemoo (LEMOO): A trained LEMOO model.
        rules_for_vars (dict): A dict with each key representing a variable. Each
            entry contains another dict representing a variable's upper and lower bounds.
            See :func:`parse_skoped_rules` for additional details.

    Returns:
        dict: A dictionary with completed missing rules. If a rule was completed utilizing
        the bounds in the population, its accuracy is set to -1.
    """
    lower_bounds = np.min(lemoo._generation_history[-1].individuals, axis=0)
    upper_bounds = np.max(lemoo._generation_history[-1].individuals, axis=0)

    # if there are no upper or lower bound for some vars in the rules, use the min or max from the population
    for var_i, var_name in enumerate(rules_for_vars):
        for op in rules_for_vars[var_name]:
            if op == "<=":
                if rules_for_vars[var_name][op][1] == -1:
                    # replace missing rule with bound from population
                    rules_for_vars[var_name][op][0] = upper_bounds[var_i]
            if op == ">":
                if rules_for_vars[var_name][op][1] == -1:
                    rules_for_vars[var_name][op][0] = lower_bounds[var_i]

    return rules_for_vars

def print_rules(lemoo: LEMOO, rules_for_vars: dict, tex: bool = False) -> None:
    """Prints the rules contained in a dict. Either as plain text or as a raw LaTeX table.
    For the structure of this dict, see
    the function :func:`parse_skoped_rules`.

    Args:
        lemoo (LEMOO): A trained LEMOO model.
        rules_for_vars (dict): A dict with rules for each variable. See :func:`parse_skoped_rules`.   
        tex (bool, optional): Whether to print the output as a raw LaTeX table or not. Defaults to False.

    Returns:
        None
    """
    lower_bounds = np.min(lemoo._generation_history[-1].individuals, axis=0)
    upper_bounds = np.max(lemoo._generation_history[-1].individuals, axis=0)

    print("RULES:")
    print("Var\tLower(R)\t\tUpper(R)\t\tLower(P)\t\tUpper(P)")
    for i, rule in enumerate(rules_for_vars):
        if not tex:
            msg = (f"{rule}\t{np.round(rules_for_vars[rule]['>'][0], 5)}\t({np.round(rules_for_vars[rule]['>'][1], 3)})\t\t{np.round(rules_for_vars[rule]['<='][0], 5)} ({np.round(rules_for_vars[rule]['<='][1], 3)})\t\t"
                   f"{np.round(lower_bounds[i], 5)}\t\t\t{np.round(upper_bounds[i], 5)}")

        else:
            msg = (f"${rule}$ & ${np.round(rules_for_vars[rule]['>'][0], 5)}$ & $({np.round(rules_for_vars[rule]['>'][1], 3)})$ & ${np.round(rules_for_vars[rule]['<='][0], 5)}$ & $({np.round(rules_for_vars[rule]['<='][1], 3)})$ &"
               f"${np.round(lower_bounds[i], 5)}$ & ${np.round(upper_bounds[i], 5)}$ \\\\")

        print(msg)

    return None