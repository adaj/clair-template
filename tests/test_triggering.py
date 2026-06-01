"""
Script: test_triggering.py

These tests cover rule parsing, triggering outputs based on given dialogue states, and manager scenarios
for selecting appropriate interventions during dialogues.

To run the tests:
    python tests/test_triggering.py
"""

import sys

sys.path.append('.')

import unittest
import json
import pandas as pd
from collections import deque

from triggering.fuzzy import TriggeringMechanism, parse_rule_to_skfuzzy
from triggering.manager import AgentManager, GroupCache, RotatingQueue


class TestTriggeringMechanism(unittest.TestCase):

    def setUp(self):
        print("> Initializing TestTriggeringMechanism...")
        self.triggering = TriggeringMechanism()

    def tearDown(self):
        print("> TestTriggeringMechanism finished.")

    def test_rule_parsing(self):
        mode = 'apt-base'
        rule = {
            'name': 'build_on_prior_knowledge__rule0',
            'antecedents': {
                'L1_DOM': ['high'],
                'L2C_AR': ['high', 'medium', 'not low'],
                'TSIM': ['high', 'medium', 'not low'],
                'TACC': ['low'], 'TIME': ['high', 'medium']
            },
            'consequent': {
                'build_on_prior_knowledge': 'active'
            }
        }
        
        antecedents = self.triggering.fuzzy_info[mode]['antecedents']
        consequents = self.triggering.fuzzy_info[mode]['consequents']
        parsed_rule = parse_rule_to_skfuzzy(rule, antecedents, consequents)

        print(f"> Rule:\n{json.dumps(rule, indent=4)}")
        print(f"> Parsed rule:\n{parsed_rule}")

    def test_triggering(self):
        mode = 'apt-base'
        dialogue_state = {
            'L1_DOM': 1,
            'L1_COO': 0,
            'L1_OFF': 0,
            'L2C_IN': 0,
            'L2C_AR': 1,
            'L2C_AI': 0,
            'L2C_AM': 0,
            'L2C_NOS': 0,
            'TSIM': 0.3,
            'TACC': 0.3,
            'PACE': 5,
            'TIME': 20*60  # 20 minutes
        }
        
        fuzzy_output = self.triggering.compute(dialogue_state, mode)

        print(f"> Dialogue state:\n{json.dumps(dialogue_state, indent=4)}")
        print(f"> Fuzzy output:\n{json.dumps(fuzzy_output, indent=4)}")

    # def test_proactive_triggering(self):
    #     mode = 'apt-goals'
    #     dialogue_state = {
    #         'L1_COO': 0.00017142732,
    #         'L1_DOM': 0.003246353,
    #         'L1_OFF': 0.99981815,
    #         'L2C_AI': 2.6138277e-05,
    #         'L2C_AM': 0.0036670126,
    #         'L2C_AR': 0.00052710966,
    #         'L2C_IN': 0.0006189267,
    #         'L2C_NOS': 0.0,
    #         'L3_CU': 0.0025896817,
    #         'L3_IP': 0.0026961204,
    #         'L3_NV': 0.00032607056,
    #         'L4_SS': 0.00017910489,
    #         'PACE': 7.894736842105264,
    #         'TACC': 1.0,
    #         'TIME': 100,
    #         'TSIM': 0.0,
    #         'TSLM': 500,
    #         'TSLQ': 289,
    #         'FCQ': 0,
    #         'FCU': 0,
    #         'FIP': 0,
    #         'FNV': 0
    #     }
        
    #     fuzzy_output = self.triggering.compute(dialogue_state, mode)

    #     print(f"> Dialogue state:\n{json.dumps(dialogue_state, indent=4)}")
    #     print(f"> Fuzzy output:\n{json.dumps(fuzzy_output, indent=4)}")


def serialize_instance(obj):
    if isinstance(obj, RotatingQueue):
        return obj.to_dict()
    if isinstance(obj, deque):
        return list(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")




if __name__ == '__main__':
    unittest.main()
