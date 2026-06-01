import sys

sys.path.append('.')

import os
import pathlib
import yaml
from typing import Dict
import numpy as np
import skfuzzy as skfuzzy
from skfuzzy import control as ctrl
import pandas as pd
import matplotlib.pyplot as plt
import fire


INPUT_RESOLUTION = 100
OUTPUT_RESOLUTION = 100


class TriggeringMechanism:

    INTERVENTION_MODES = [f for f in os.listdir(os.path.join("triggering", "modes")) if '.' not in f]

    def __init__(self):
        self.fuzzy_info = {mode: self.load_fuzzy_info(mode) for mode in self.INTERVENTION_MODES} 

    def load_fuzzy_info(self, intervention_mode: str):
        # Load antecedents
        with open(os.path.join("triggering", "modes", intervention_mode, "inputs", "memberships.yml"), 'r') as f:
            antecedents_data = yaml.safe_load(f)
        antecedents = {}
        for a in antecedents_data:
            try:
                universe = np.linspace(*a['range'], INPUT_RESOLUTION+1)
                antecedents[a['name']] = ctrl.Antecedent(universe,
                                                         label=a['name'])
                for term in a['terms']:
                    antecedents[a['name']][term['name']] = membership(universe, **term)
            except Exception as e:
                raise Exception(f'{a} ~ ') from e

        # Load consequents
        with open(os.path.join("triggering", "modes", intervention_mode, "outputs", "memberships.yml"), 'r') as f:
            consequents_data = yaml.safe_load(f)
        consequents = {}
        for c in consequents_data:
            universe = np.linspace(*c['range'], OUTPUT_RESOLUTION+1)
            consequents[c['name']] = ctrl.Consequent(universe,
                                                          label=c['name'],
                                                          defuzzify_method='som')
            for term in c['terms']:
                consequents[c['name']][term['name']] = \
                    membership(universe, **term)

        # Load rule base
        with open(os.path.join("triggering", "modes", intervention_mode, "rules.yml"), 'r') as f:
            rule_base = yaml.safe_load(f)

        # Parse rule base into skfuzzy
        fuzzy_rules = []
        for rule in rule_base:
            fuzzy_rules.append(
                parse_rule_to_skfuzzy(input_rule=rule,
                                      antecedents=antecedents,
                                      consequents=consequents)
            )

        # Create fuzzy system
        fuzzy_system = ctrl.ControlSystem(fuzzy_rules)
    
        return {'antecedents': antecedents, 'consequents': consequents, 'fuzzy_system': fuzzy_system}

    def compute(self,
                dialogue_state,
                mode: str = 'apt-base',
                return_sim: bool = False):
        fuzzy_sim = ctrl.ControlSystemSimulation(self.fuzzy_info[mode]['fuzzy_system'])
        # dialogue_variables = set() # todo
        for variable in dialogue_state.keys():
            if variable in self.fuzzy_info[mode]['antecedents']:
                try:
                    fuzzy_sim.input[variable] = dialogue_state[variable]
                except:
                    # Means the variable is not being used in the fuzzy rules, so it is not part of the system
                    pass 
        fuzzy_sim.compute()
        if return_sim:
            return fuzzy_sim
        return dict(fuzzy_sim.output)

    # Plot membership & save figures
    def plot_membership(self, mode: str):
        save_folder = os.path.join('triggering', 'modes', mode, 'inputs', 'plots', 'memberships')
        # Create dir to export membership information
        pathlib.Path(save_folder).mkdir(parents=True, exist_ok=True)
        for variable in self.fuzzy_info[mode]['antecedents']:
            plt.rcParams['axes.facecolor']='white'
            plt.rcParams['savefig.facecolor']='white'
            self.fuzzy_info[mode]['antecedents'][variable].view()
            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, f'{variable}.png'))
            plt.close()


def membership(universe: np.array, fn: str, **kwargs):
    kwargs.pop('name', None) # Not needed
    if fn == 'trimf':
        return skfuzzy.membership.trimf(universe, **kwargs)
    elif fn == 'gbellmf':
        return skfuzzy.membership.gbellmf(universe, **kwargs)
    elif fn == 'gaussmf':
        return skfuzzy.membership.gaussmf(universe, **kwargs)
    elif fn == 'smf':
        return skfuzzy.membership.smf(universe, **kwargs)
    elif fn == 'zmf':
        return skfuzzy.membership.zmf(universe, **kwargs)
    elif fn == 'trapmf':
        return skfuzzy.membership.trapmf(universe, **kwargs)
    else:
        raise Exception("`fn` is not supported.")


def parse_rule_to_skfuzzy(input_rule: Dict,
                          antecedents: Dict[str, ctrl.Antecedent],
                          consequents: Dict[str, ctrl.Consequent]):
    rule_antecedent = None
    subrule_L1 = None
    subrule_L2C = None
    subrule_negations = None
    for name, values in input_rule['antecedents'].items():
        try:
            if 'L1' in name:
                for v in values:
                    if v.split()[0]=='not':
                        v = v.split()[-1]
                        if subrule_negations is None:
                            subrule_negations = ~antecedents[name][v]
                        else:
                            subrule_negations &= ~antecedents[name][v]
                    elif subrule_L1 is None:
                        subrule_L1 = antecedents[name][v]
                    else:
                        subrule_L1 |= antecedents[name][v]
            elif 'L2' in name:
                for v in values:
                    if v.split()[0]=='not':
                        v = v.split()[-1]
                        if subrule_negations is None:
                            subrule_negations = ~antecedents[name][v]
                        else:
                            subrule_negations &= ~antecedents[name][v]
                    elif subrule_L2C is None:
                        subrule_L2C = antecedents[name][v]
                    else:
                        subrule_L2C |= antecedents[name][v]
            else:
                subrule = None
                for v in values:
                    if v.split()[0]=='not':
                        v = v.split()[-1]
                        if subrule_negations is None:
                            subrule_negations = ~antecedents[name][v]
                        else:
                            subrule_negations &= ~antecedents[name][v]
                    elif subrule is None:
                        subrule = antecedents[name][v]
                    else:
                        subrule |= antecedents[name][v]
                if rule_antecedent is None:
                    rule_antecedent = subrule
                else:
                    rule_antecedent &= subrule
        except Exception as e:
            print(f'{name, values} ~ {e}')
            raise e

    for subrule in [subrule_L1, subrule_L2C, subrule_negations]:
        if subrule is None:
            continue
        if rule_antecedent is None:
            rule_antecedent = subrule
        else:
            rule_antecedent &= subrule

    consequent_label = list(input_rule['consequent'].keys())[0]
    consequent_value = input_rule['consequent'][consequent_label]
    return ctrl.Rule(
        antecedent = rule_antecedent,
        consequent = consequents[consequent_label][consequent_value],
        label = input_rule['name']
    )


if __name__=="__main__":
    fire.Fire(TriggeringMechanism)
