import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# fuzzy inference system for washing machine
# New Antecedent/Consequent objects hold universe variables and membership
# functions
quality = ctrl.Antecedent(np.arange(0, 11, 1), 'quality')
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
type_clothes = ctrl.Antecedent(np.arange(0, 4, 1), 'type_clothes')
time = ctrl.Consequent(np.arange(0, 26, 1), 'time')

service.view()
type_clothes.view()
# Auto-membership function population is possible with .automf(3, 5, or 7)
quality.automf(3)
service.automf(3)
type_clothes.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
time['low'] = fuzz.trimf(time.universe, [0, 0, 13])
time['medium'] = fuzz.trimf(time.universe, [0, 13, 25])
time['high'] = fuzz.trimf(time.universe, [13, 25, 25])

# You can see how these look with .view()
rule1 = ctrl.Rule(quality['poor'] | service['poor'], time['low'])
rule2 = ctrl.Rule(service['average'], time['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], time['high'])
rule4 = ctrl.Rule(type_clothes['poor'], time['high'])

# rule1.view()

washing_machine_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4])
washing_machine = ctrl.ControlSystemSimulation(washing_machine_ctrl)
# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
washing_machine.input['quality'] = 6.5
washing_machine.input['service'] = 9.8
washing_machine.input['type_clothes'] = 2
# Crunch the numbers
washing_machine.compute()
print(washing_machine.output['time'])
time.view(sim=washing_machine)
print('')
