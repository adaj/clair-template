"""
Script: clair_api_tester.py

This script is designed to interface with the Clair API to test data interactions.
Given a specified CSV dataset, it will send data to the Clair API and capture 
the responses in an Excel spreadsheet.

The data required to run this script is a CSV file with the following columns:
    - group: the group of the message
    - username: the username of the message
    - timestamp: the timestamp of the message
    - text: the text of the message

Usage:
    set (or export, on Linux) CLAIR_URL=<YOUR_CLAIR_URL>
    python clair_api_tester.py --data_file_path=<path_to_data_file> --n_groups=<number_of_groups_to_test>
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import hashlib
import fire
from pprint import pprint
from tqdm import tqdm

sys.path.append('.')


def read_data(data_file_path: str):
    try:
        data_df = pd.read_csv(data_file_path, index_col=0)
    except:
        data_df = pd.read_csv(data_file_path, index_col=0, sep='|')
    data_df = data_df.drop(columns=['L1', 'L2C'], errors='ignore')
    data_df = data_df.rename(columns={
        'message':'text', 'dialog_id': 'group'
    }, errors='ignore')
    data_df['text'] = data_df['text'].fillna(' ')
    return data_df

def main(data_file_path: str, dataset_lang: str = "EN", topics_file: str = None, mode: str = "apt-base", n_groups: int = -1):

    # Ensure the CLAIR_URL environment variable is set.
    CLAIR_URL = os.environ.get('CLAIR_URL')
    assert CLAIR_URL and isinstance(CLAIR_URL, str), "Please set the env variable CLAIR_URL with a running clair http app."
    CLAIR_URL = CLAIR_URL.rstrip('/')
    print(CLAIR_URL)

    # Load topics
    if topics_file is None:
        topics_file = os.path.join(Path(data_file_path).parents[1], 'topics.txt')
    try:
        with open(topics_file, 'r', encoding='utf-8') as f:
            keywords = f.read().splitlines()
    except:
        # keywords = ["enzymes", "digestive system", "nutrients"]
        keywords = ["energy", "potential", "kinetic", "acceleration", "mass"]

    agent_configuration = {
        "learning_space": f"test/clair_api_{mode}",
        "is_active": True,
        "mode": mode,
        "language": dataset_lang,
        "keywords": keywords
    }

    pprint(agent_configuration)

    # Activate agent under this configuration
    req = requests.post(f"{CLAIR_URL}/configuration", data=json.dumps(agent_configuration), 
                        headers={'Content-Type': 'application/json', 'access_token': os.environ.get('CLAIR_TOKEN')},
                        timeout=20)
    print(req, req.text, req.status_code, req.reason)

    # Read data and, if specified, subsample groups
    data_df = read_data(data_file_path)

    if isinstance(n_groups, str):
        # Pick one group
        # 6ba0dc91-5b7e-4ab4-fa7b-3765cd1c2eee
        data_df = data_df[data_df['group'] == n_groups]
    else:
        if 0 < n_groups < len(data_df['group'].unique()):
            groups = np.random.choice(data_df['group'].unique(), n_groups, replace=False)
            data_df = data_df[data_df['group'].isin(groups)]
        else:
            print(f"Using all groups ({len(data_df['group'].unique())})")

    # Ensure data is temporally sorted
    data_df = data_df.sort_values(by='timestamp')

    print("Starting...")
    rows = []
    run_id = hashlib.md5(pd.Timestamp.now().strftime('%Y%m%d%H%M%S').encode()).hexdigest()[:4]
    for _, message in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        message_dict = message.fillna("").to_dict()

        message_dict['group'] = run_id + "_" + message['group']

        if message['username'] != "Clair" and message['username'] != "MAI":
            chat_msg = {
                'learning_space': agent_configuration["learning_space"],
                'group': message_dict['group'],
                'username': message['username'],
                'text': message['text'][:1000],
                'timestamp': message['timestamp']
            }
            try:
                req = requests.post(f"{CLAIR_URL}/message?retrieve_details=true&save=false",
                                    data=json.dumps(chat_msg), 
                                    headers={'Content-Type': 'application/json', 'access_token': os.environ.get('CLAIR_TOKEN')},
                                    timeout=30)
                response_data = req.json()
                
                # Check for errors
                if 'detail' in response_data:
                    raise ValueError(f"Input format error. Details: {response_data['detail']}")
                if 'dialogue_state' not in response_data:
                    raise ValueError("Expected 'dialogue_state' in the API response but it was missing.")
                
                # Add response to the data
                message_dict.update(response_data)
            except Exception as excep_msg:
                excep_msg.args = (f"Some failure happens at message: {message_dict}", *excep_msg.args)
                raise

        rows.append(message_dict)

    # Process and save data
    rows_df = pd.DataFrame(rows).drop(columns=['dialog_id'], errors='ignore')
    rows_df['timestamp'] = pd.to_datetime(rows_df['timestamp'])
    for col in ['dialogue_state', 'fuzzy_output']:
        inner_dict = pd.json_normalize(rows_df[col])
        rows_df = rows_df.join(inner_dict, rsuffix=f"_{col}")
    rows_df.drop(columns=['id', 'learning_space', 'last_update', 'dialogue_state', 'fuzzy_output'], 
                 errors='ignore', inplace=True)
    rows_df = rows_df.sort_values(by=['group', 'timestamp'])
    rows_df['timestamp'] = rows_df['timestamp'].astype(str)

    writer = pd.ExcelWriter(data_file_path.replace('.csv', f'__clair_api_{mode}.xlsx'), engine='xlsxwriter')
    for i, group in enumerate(rows_df['group'].unique()):
        rows_df[rows_df['group'] == group].to_excel(writer, encoding="utf-8", sheet_name=f"Chat-{i+1}")
    writer.save()

if __name__=="__main__":
    fire.Fire(main)
