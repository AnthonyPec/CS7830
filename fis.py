import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# fuzzy inference system for washing machine
# New Antecedent/Consequent objects hold universe variables and membership
# functions
dirtiness = ctrl.Antecedent(np.arange(0, 101, 1), 'dirtiness')
type_dirt = ctrl.Antecedent(np.arange(0, 101, 1), 'type_dirt')

# weight in lbs
weight = ctrl.Antecedent(np.arange(0, 16, 1), 'weight')

time = ctrl.Consequent(np.arange(0, 81, 1), 'time', defuzzify_method='centroid')


# Auto-membership function with triangle membership


def create_triangular_membership_functions():
    type_dirt.automf(3, names=['not_greasy', 'medium', 'greasy'])
    dirtiness.automf(3, names=['not_dirty', 'dirty', 'very_dirty'])
    weight.automf(3, names=['light', 'medium', 'heavy'])

    time['short'] = fuzz.trimf(time.universe, [0, 0, 20])
    time['medium'] = fuzz.trimf(time.universe, [0, 20, 45])
    time['long'] = fuzz.trimf(time.universe, [20, 40, 60])
    time['very_long'] = fuzz.trimf(time.universe, [45, 60, 80])


def create_trapezoidal_membership_functions():
    type_dirt['not_greasy'] = fuzz.trapmf(type_dirt.universe, [0, 10, 30, 50])
    type_dirt['medium'] = fuzz.trapmf(type_dirt.universe, [20, 40, 60, 80])
    type_dirt['greasy'] = fuzz.trapmf(type_dirt.universe, [50, 70, 90, 100])

    dirtiness['not_dirty'] = fuzz.trapmf(dirtiness.universe, [0, 10, 30, 50])
    dirtiness['dirty'] = fuzz.trapmf(dirtiness.universe, [20, 40, 60, 80])
    dirtiness['very_dirty'] = fuzz.trapmf(dirtiness.universe, [50, 70, 90, 100])

    weight['light'] = fuzz.trapmf(weight.universe, [0, 10, 30, 50])
    weight['medium'] = fuzz.trapmf(weight.universe, [20, 40, 60, 80])
    weight['heavy'] = fuzz.trapmf(weight.universe, [50, 70, 90, 100])

    time['short'] = fuzz.trapmf(time.universe, [0, 0, 10, 20])
    time['medium'] = fuzz.trapmf(time.universe, [0, 20, 30, 45])
    time['long'] = fuzz.trapmf(time.universe, [20, 40, 50, 60])
    time['very_long'] = fuzz.trapmf(time.universe, [45, 60, 70, 80])


def create_rules(and_fun=np.min, or_fun=np.fmax):
    rule1 = ctrl.Rule(type_dirt['not_greasy'], time['short'], and_func=np.multiply)
    rule2 = ctrl.Rule(dirtiness['very_dirty'] & weight['heavy'], time['very_long'])
    rule3 = ctrl.Rule(type_dirt['greasy'], time['very_long'])
    rule4 = ctrl.Rule(weight['medium'] & dirtiness['dirty'], time['medium'], and_func=np.multiply)
    rule5 = ctrl.Rule(weight['medium'] | type_dirt['medium'], time['long'])
    rule6 = ctrl.Rule(weight['light'] & dirtiness['not_dirty'], time['short'], and_func=np.multiply)

    return [rule1, rule2, rule3, rule4, rule5, rule6]


# how to change membership functions


# type_clothes.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
def use_washing_machine(washing_machine):
    washing_machine.input['dirtiness'] = 99
    washing_machine.input['type_dirt'] = 99
    washing_machine.input['weight'] = 15

    # Crunch the numbers
    washing_machine.compute()
    print(washing_machine.output['time'])

    washing_machine.input['dirtiness'] = 1
    washing_machine.input['type_dirt'] = 1
    washing_machine.input['weight'] = 1
    washing_machine.compute()
    time.view(sim=washing_machine)
    print(washing_machine.output['time'])

    washing_machine.input['dirtiness'] = 50
    washing_machine.input['type_dirt'] = 50
    washing_machine.input['weight'] = 8
    washing_machine.compute()
    print(washing_machine.output['time'])


# create_trapezoidal_membership_functions()
create_triangular_membership_functions()
# You can see how these look with .view()
rules_list = create_rules()
washing_machine_ctrl = ctrl.ControlSystem(rules_list)
use_washing_machine(ctrl.ControlSystemSimulation(washing_machine_ctrl))

create_trapezoidal_membership_functions()
# You can see how these look with .view()
rules_list = create_rules()
washing_machine_ctrl = ctrl.ControlSystem(rules_list)
use_washing_machine(ctrl.ControlSystemSimulation(washing_machine_ctrl))
# skfuzzdefuzz

create_triangular_membership_functions()
# You can see how these look with .view()
rules_list = create_rules(and_fun=np.multiply)
washing_machine_ctrl = ctrl.ControlSystem(rules_list)
use_washing_machine(ctrl.ControlSystemSimulation(washing_machine_ctrl))
