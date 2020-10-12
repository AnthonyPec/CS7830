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

time = ctrl.Consequent(np.arange(0, 120, 1), 'time', defuzzify_method='centroid')


def create_triangular_membership_functions():
    '''
    use automf to auto generate all triangular membership functions
    '''
    type_dirt.automf(3, names=['not_greasy', 'medium', 'greasy'])
    dirtiness.automf(3, names=['not_dirty', 'dirty', 'very_dirty'])
    weight.automf(3, names=['light', 'medium', 'heavy'])
    time.automf(5, names=['very_short', 'short', 'medium', 'long', 'very_long'])


def create_trapezoidal_membership_functions():
    type_dirt['not_greasy'] = fuzz.trapmf(type_dirt.universe, [0, 0, 20, 50])
    type_dirt['medium'] = fuzz.trapmf(type_dirt.universe, [0, 40, 60, 100])
    type_dirt['greasy'] = fuzz.trapmf(type_dirt.universe, [50, 80, 100, 100])

    dirtiness['not_dirty'] = fuzz.trapmf(dirtiness.universe, [0, 0, 20, 50])
    dirtiness['dirty'] = fuzz.trapmf(dirtiness.universe, [0, 40, 60, 100])
    dirtiness['very_dirty'] = fuzz.trapmf(dirtiness.universe, [50, 80, 100, 100])

    weight['light'] = fuzz.trapmf(weight.universe, [0, 0, 4, 8])
    weight['medium'] = fuzz.trapmf(weight.universe, [0, 6, 9, 15])
    weight['heavy'] = fuzz.trapmf(weight.universe, [7, 11, 15, 15])

    time['short'] = fuzz.trapmf(time.universe, [0, 0, 10, 20])
    time['medium'] = fuzz.trapmf(time.universe, [0, 20, 30, 45])
    time['long'] = fuzz.trapmf(time.universe, [20, 40, 50, 60])
    time['very_long'] = fuzz.trapmf(time.universe, [45, 60, 70, 80])


def create_rules(and_fun=np.min, or_fun=np.fmax):
    rule1 = ctrl.Rule(type_dirt['not_greasy'] & (weight['light'] | dirtiness['not_dirty']), time['very_short'],
                      and_func=and_fun)
    rule2 = ctrl.Rule(type_dirt['not_greasy'] & (weight['medium'] | dirtiness['dirty']), time['short'],
                      and_func=and_fun)
    rule3 = ctrl.Rule(type_dirt['not_greasy'] & (weight['heavy'] | dirtiness['very_dirty']), time['medium'],
                      and_func=and_fun)

    rule4 = ctrl.Rule(type_dirt['medium'] & (weight['light'] | dirtiness['not_dirty']), time['short'], and_func=and_fun)
    rule5 = ctrl.Rule(type_dirt['medium'] & (weight['medium'] | dirtiness['dirty']), time['medium'],
                      and_func=and_fun)
    rule6 = ctrl.Rule(type_dirt['medium'] & (weight['heavy'] | dirtiness['very_dirty']), time['long'],
                      and_func=and_fun)

    rule7 = ctrl.Rule(type_dirt['greasy'] & (weight['light'] | dirtiness['not_dirty']), time['medium'],
                      and_func=and_fun)
    rule8 = ctrl.Rule(type_dirt['greasy'] & (weight['medium'] | dirtiness['dirty']), time['long'],
                      and_func=and_fun)
    rule9 = ctrl.Rule(type_dirt['greasy'] & (weight['heavy'] | dirtiness['very_dirty']), time['very_long'],
                      and_func=and_fun)

    return [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]


# Custom membership functions can be built interactively with a familiar,
# Pythonic API
def use_washing_machine(washing_machine, input_values):
    for i in input_values:
        washing_machine.input['dirtiness'] = i[0]
        washing_machine.input['type_dirt'] = i[1]
        washing_machine.input['weight'] = i[2]
        washing_machine.compute()
        time.view(sim=washing_machine)
        print(washing_machine.output['time'])


def view_membership_functions():
    weight.view()
    dirtiness.view()
    type_dirt.view()
    time.view()


input_list = [[98, 98, 14], [2, 2, 1], [50, 50, 8]]
if __name__ == '__main__':
    # 1a
    create_triangular_membership_functions()
    #view_membership_functions()

    # 3a
    create_triangular_membership_functions()
    rules_list = create_rules()
    washing_machine_ctrl = ctrl.ControlSystem(rules_list)
    use_washing_machine(ctrl.ControlSystemSimulation(washing_machine_ctrl), input_list)

    # 3b
    create_trapezoidal_membership_functions()
    rules_list = create_rules()
    washing_machine_ctrl = ctrl.ControlSystem(rules_list)
    use_washing_machine(ctrl.ControlSystemSimulation(washing_machine_ctrl), input_list)

    # 3c
    create_triangular_membership_functions()
    rules_list = create_rules(and_fun=np.multiply)
    washing_machine_ctrl = ctrl.ControlSystem(rules_list)
    use_washing_machine(ctrl.ControlSystemSimulation(washing_machine_ctrl), input_list)

    # 3d
    time.defuzzify_method = 'lom'
    create_triangular_membership_functions()
    rules_list = create_rules()
    washing_machine_ctrl = ctrl.ControlSystem(rules_list)
    use_washing_machine(ctrl.ControlSystemSimulation(washing_machine_ctrl), input_list)
