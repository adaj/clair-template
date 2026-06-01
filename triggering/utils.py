"""
This module contains the utility functions used by the triggering module.
The functions in this module are not meant to be used directly by the user.
"""
import sys

sys.path.append('.')

import os
from typing import List, Dict
import pathlib
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

import tensorflow_text
import tensorflow_hub as hub
from tqdm import tqdm

from nlu.consent.consent import ConSent
from nlu.topics import TopicEmbeddings, create_encoder
from nlu.dialogue import DialogueTracker
from nlu.intents.intent_classifier import IntentClassifier


def get_value(list_of_dicts: List[Dict], id: str):
    """
    Returns the value of a dictionary in a list of dictionaries

    Args:
        list_of_dicts (List[Dict]): List of dictionaries
        id (str): id of the dictionary

    Returns:
        _type_: value of the dictionary
    """
    for item in list_of_dicts:
        if item['id'] == id:
            return item['value']
    return None


def read_data(data_file_path: str):
    """
    Reads the data file and returns a pandas dataframe.

    Args:
        data_file_path (str): _description_

    Returns:
        _type_: _description_
    """
    try:
        data_df = pd.read_csv(data_file_path, index_col=0)
    except:
        data_df = pd.read_csv(data_file_path, index_col=0, sep='|')
    data_df = data_df.drop(columns=['L1', 'L2C'], errors='ignore')
    data_df = data_df.rename(columns={
        'message':'text', 'dialog_id': 'group', 'user': 'username'
    }, errors='ignore')
    data_df['text'] = data_df['text'].fillna(' ')
    return data_df


def find_clusters_kmeans(df: pd.DataFrame,
                         variable: str,
                         n_clusters: int):
    df_samples = df.reset_index(drop=True)
    kmeans = KMeans(n_clusters).fit(df_samples[variable].values.reshape(-1,1))
    df_samples[f'{variable}_cluster'] = kmeans.labels_
    return df_samples, kmeans


def plot_clusters_dist(output_file_path: str,
                       df_samples: pd.DataFrame,
                       variable: str,
                       centroids: np.array,
                       title: str,
                       bins: int,
                       logscale: bool):
    plt.rcParams.update({'font.size': 14, "font.family": "Times"})
    fig, ax = plt.subplots(figsize=(6,5))
    sns.histplot(data=df_samples, x=variable, hue=f'{variable}_cluster',
                 bins=bins, element="step",
                 palette='Set1', ax=ax)
    if logscale:
        ax.set_yscale('log')
    for c in np.sort(centroids):
        ax.axvline(x=c, color='cyan')
    fig.suptitle(title)
    fig.tight_layout()
    # Create dir to export plots
    pathlib.Path(os.path.dirname(output_file_path)).mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file_path, dpi=300)
    return fig, ax


def extract_dialogue_states(data_folder, include_intents=False, n_groups=-1):
    print(f'> Extracting dialogue states from {data_folder}...')
    # Step 1 - instantiate TopicEmbeddings, DialogueStateTracker
    try:
        with open(os.path.join(data_folder, 'topics.txt'), 'r', encoding='utf-8') as f:
            keywords = f.read().split('\n')
    except:
        # keywords = ["enzymes", "digestive system", "nutrients"]
        keywords = ["energy", "potential", "kinetic", "acceleration", "mass"]
        print(f'No topics file found in {data_folder}. Using hard-coded keywords: {keywords}', flush=True)
        
    
    language = data_folder.split('Chats-')[1][:2].upper()
    if language not in ['EN', 'PT', 'NL']:
        language = 'EN'
    
    # NLU
    L1_consent = ConSent(load='nlu/consent/code_L1__v2')
    L2C_consent = ConSent(load='nlu/consent/code_L2C__v2')
    sentence_encoder = create_encoder()
    topic_embeddings = TopicEmbeddings(
        encoder=sentence_encoder,
        keywords=keywords,
        stop_words_file=f"nlu/stop_words/{language}.txt"
    )
    if include_intents:
        intent_classifiers = {
            'L3P': IntentClassifier(load_model='nlu/intents/models/code_L3P_v1'),
            'L3C': IntentClassifier(load_model='nlu/intents/models/code_L3C_v1'),
            'L3V': IntentClassifier(load_model='nlu/intents/models/code_L3V_v1'),
            'L3E': IntentClassifier(load_model='nlu/intents/models/code_L3E_v1'),
            'L4': IntentClassifier(load_model='nlu/intents/models/code_L4_Enzymes_v1')\
                    .load_stop_words(stop_words_file=f"nlu/stop_words/{language}.txt"),
        }
    else:
        intent_classifiers = None

    # Step 2 - read data folders
    try:
        data_df = read_data(os.path.join(data_folder, 'preprocessed/chats.csv'))
    except:
        data_df = read_data(os.path.join(data_folder, 'preprocessed/chats_train.csv'))
    data_df = data_df[data_df['username'] != 'Clair']

    if isinstance(n_groups, str):
        # Pick one group
        data_df = data_df[data_df['group'] == n_groups]
    else:
        if 0 < n_groups < len(data_df['group'].unique()):
            groups = np.random.choice(data_df['group'].unique(), n_groups, replace=False)
            data_df = data_df[data_df['group'].isin(groups)]
        else:
            print(f"Using all groups ({len(data_df['group'].unique())})")

    # Step 3 - iterate through each group, logging state in a csv file
    try:
        group_data = data_df.groupby('group')
    except:
        group_data = data_df.groupby('dialog_id')
    dialogue_variables_df = []
    for gi, g in tqdm(group_data, total=len(group_data)):
        dst = DialogueTracker(L1_consent=L1_consent, 
                              L2C_consent=L2C_consent, 
                              topic_embeddings=topic_embeddings,
                              intent_classifiers=intent_classifiers)
        states = []
        for mi, message in g.iterrows():
            if message['username'] != 'Clair':
                dst.update(**message.to_dict())
                dialogue_state = copy.deepcopy(dst.variables)
                states.append({**message.to_dict(), **dialogue_state})
            else:
                states.append(message.to_dict())
        dialogue_variables_df.append(pd.DataFrame(states))
    dialogue_variables_df = pd.concat(dialogue_variables_df)
    return dialogue_variables_df


def compute_membership_centroids(dialogue_states, dialogue_variables, save_plots):
    # For each variable, find the membership centroids and plot histograms
    membership_centroids = []
    for variable in dialogue_variables:
        # Compure k-mean clusters with 3 clusters (high, medium, low)
        df_samples, kmeans = find_clusters_kmeans(df=dialogue_states,
                                                  variable=variable,
                                                  n_clusters=3)
        centroids = np.sort(kmeans.cluster_centers_.round(4).ravel())
        membership_centroids.append({
            "variable": variable,
            "min_value": df_samples[variable].min(),
            "max_value": df_samples[variable].max(),
            "C_low": centroids[0],
            "C_medium": centroids[1],
            "C_high": centroids[2],
        })
        # Generate histograms, plot and save figures
        if save_plots:
            plot_clusters_dist(
                output_file_path=os.path.join(save_plots, f'{variable}.png'),
                df_samples=df_samples, variable=variable, centroids=centroids,
                bins=20, logscale=False,
                title=f"variable:{variable}\ncluster centroids={centroids}"
            )
    return pd.DataFrame(membership_centroids)


def generate_antecedents_memberships(centroids):
    antecedents = []
    for _, c in centroids.iterrows():
        min_value = int(np.floor(c['min_value']))
        max_value = int(np.ceil(c['max_value']))
        # Intersection in the average
        avg_low_medium = (c['C_low'] + c['C_medium']) / 2
        avg_medium_high = (c['C_medium'] + c['C_high']) / 2
        # Definition of LOW
        low = {
            'name': 'low',
            'fn': 'trapmf',
            'abcd': [min_value, min_value, avg_low_medium, c['C_medium']]
        }
        # Definition of MEDIUM
        medium = {
            'name': 'medium',
            'fn': 'trapmf',
            'abcd': [c['C_low'], avg_low_medium, avg_medium_high, c['C_high']]
        }
        # Definition of HIGH
        high = {
            'name': 'high',
            'fn': 'trapmf',
            'abcd': [c['C_medium'], avg_medium_high, max_value, max_value]
        }
        antecedents.append({
            'name': c['variable'],
            'range': [min_value, max_value],
            'terms': [low, medium, high]
        })
    return antecedents


def generate_consequents_memberships(talk_moves):
    consequents = []
    for talk_move in talk_moves:
        min_value = 0
        max_value = 1
        # Definition of ACTIVE
        active = {
            'name': 'active',
            'fn': 'trimf',
            'abc': [min_value, max_value, max_value]
        }
        not_active = {
            'name': 'not_active',
            'fn': 'trimf',
            'abc': [min_value, min_value, max_value]
        }
        consequents.append({
            'name': talk_move,
            'range': [min_value, max_value],
            'terms': [active, not_active]
        })
    return consequents

