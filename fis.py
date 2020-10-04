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

#type_clothes = ctrl.Antecedent(np.arange(0, 5, 1), 'type_clothes')

time = ctrl.Consequent(np.arange(0, 81, 1), 'time')

# Auto-membership function with triangle membership
type_dirt.automf(3, names=['not_greasy', 'medium', 'greasy'])

# type_dirt['not_greasy'] = fuzz.trimf(type_dirt.universe, [0, 0,6])
# type_dirt['medium'] = fuzz.trimf(type_dirt.universe, [0, 0,6])
# type_dirt['greasy'] = fuzz.trimf(type_dirt.universe, [0, 0,6])

dirtiness.automf(3, names=['not_dirty', 'dirty', 'very_dirty'])
weight.automf(3, names=['light', 'med', 'heavy'])

# how to change membership functions
# weight['light'] = fuzz.trimf(weight.universe, [0, 0,6])
# weight['medium'] = fuzz.trimf(weight.universe, [0, 6, 8])
# weight['heavy'] = fuzz.trimf(weight.universe, [6, 10, 15])

#type_clothes.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
time['short'] = fuzz.trimf(time.universe, [0, 0, 20])
time['medium'] = fuzz.trimf(time.universe, [0, 20, 45])
time['long'] = fuzz.trimf(time.universe, [20, 40, 60])
time['very_long'] = fuzz.trimf(time.universe, [45, 60, 80])


# You can see how these look with .view()
rule1 = ctrl.Rule(type_dirt['not_greasy'], time['short'])
rule2 = ctrl.Rule(dirtiness['very_dirty'] & weight['heavy'], time['very_long'])
rule3 = ctrl.Rule(type_dirt['greasy'], time['very_long'])
rule4 = ctrl.Rule(weight['med'] & dirtiness['dirty'], time['medium'])
rule5 = ctrl.Rule(weight['med'] | type_dirt['medium'], time['long'])
rule6 = ctrl.Rule(weight['light'] & dirtiness['not_dirty'], time['short'])


washing_machine_ctrl = ctrl.ControlSystem([rule1, rule2, rule3,rule4,rule5])
washing_machine = ctrl.ControlSystemSimulation(washing_machine_ctrl)


# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
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

print('')
