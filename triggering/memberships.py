"""
This script is used to setup the fuzzy system.

It generates the antecedents and membership functions
for the fuzzy logic system, based on the dialogue states extracted
from the training data. The dialogue states are used to compute the
membership centroids for each variable. The membership centroids are used
to generate the antecedents and membership functions for the fuzzy logic
system.

The script can be called by the following command:
    python triggering/memberships.py inputs --mode="apt-base"
    python triggering/memberships.py outputs --mode="apt-base"

When you run the script for inputs, it will generate the following files:
    - triggering/<your_mode>/inputs/memberships.yml
    - triggering/<your_mode>/inputs/plots/clusters/<variable>.png (plots)

When you run the script for outputs, it will generate the following files:
    - triggering/<your_mode>/outputs/memberships.yml
"""
import sys

sys.path.append('.')

import os
import numpy as np
import pandas as pd
import yaml
import fire

from triggering.utils import extract_dialogue_states, compute_membership_centroids
from triggering.utils import generate_antecedents_memberships, generate_consequents_memberships


class Handler:

    def __init__(self, mode: str = 'apt-base'):
        mode = f"triggering/modes/{mode}"
        with open(os.path.join(mode, "config.yml"), 'r') as f:
            self.config = yaml.safe_load(f)
        self.mode = mode

    def inputs(self, n_groups: int = -1):
        print("Step 1 - Extracting dialogue states of prior datasets...")
        data_folders = self.config["data_folders"]
        if isinstance(data_folders, str):
            data_folders = [self.config["data_folders"]]
        include_intents = any(["L3" in i for i in self.config["inputs"]])
        dialogue_states = []
        for data_i in data_folders:
            print(f"Extracting dialogue states from {data_i}...")
            outputs = extract_dialogue_states(data_folder=data_i, 
                                              include_intents=include_intents,
                                              n_groups=n_groups)
            dialogue_states.append(outputs)
        dialogue_states = pd.concat(dialogue_states, axis=0)
        # TODO: Save dialogue states into an excel file

        print("Step 2 - Computing centroids based on the data...")
        centroids = compute_membership_centroids(dialogue_states,
                                                 self.config["inputs"],
                                                 save_plots=os.path.join(self.mode, "inputs/plots/clusters"))
        
        print("Step 3 - Parsing fuzzy inputs (dialogue variables) and their membership fns...")
        inputs = generate_antecedents_memberships(centroids)
        output_file_path = os.path.join(self.mode, 'inputs', 'memberships.yml')
        with open(output_file_path, 'w') as yaml_file:
            yaml.dump(inputs, yaml_file,
                      sort_keys=False, default_flow_style=False)
            
    def outputs(self):
        print("Step 4 - Parsing fuzzy outputs (talk moves) and their membership fns...")
        outputs = generate_consequents_memberships(self.config['outputs'])
        output_file_path = os.path.join(self.mode, 'outputs', 'memberships.yml')
        with open(output_file_path, 'w') as yaml_file:
            yaml.dump(outputs, yaml_file,
                    sort_keys=False, default_flow_style=False)
        print("Step 5 - Now prepare your rules.yml following the template available in the `triggering/apt-base` folder.")



if __name__=="__main__":
    fire.Fire(Handler)
