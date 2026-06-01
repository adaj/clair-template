"""
Script: test_app.py

This script contains unittests to validate the behavior of the app module in the CLAIR system.
It interacts with the specified CLAIR API endpoints and checks the responses.

To run the tests, ensure the CLAIR_URL environment variable is correctly set.

Usage:
    python tests/test_app.py
"""

import sys
import os
import unittest
import requests
import time
import json
import pandas as pd
from pprint import pprint

sys.path.append('.')


class test_app(unittest.TestCase):
    """
    Test class for validating the CLAIR API routes.
    """

    def setUp(self):
        """
        Set up the testing environment, verifying the CLAIR_URL and initializing shared properties.
        """
        self.app_url = os.environ.get('CLAIR_URL', '').rstrip('/')
        assert self.app_url, "Please set the env variable CLAIR_URL with a running clair http app."
        print('CLAIR_URL:', self.app_url)

    def test_configuration(self):
        """
        Test the configuration endpoint of the CLAIR API.
        Ensure that the status code of the response is 200 (OK).
        """
        config = {
            "learning_space": "test_app",
            "is_active": True,
            "mode": "apt-base",
            "language": "EN",
            "keywords": [
                "Electric circuits",
                "Parallel and serial circuits",
                "Resistance and capacitance",
                "Voltage and current"
            ],
            "blacklist": [
                "blacklisted_group1",
                "blacklisted_group2"
            ],
        }

        req = requests.post(f"{self.app_url}/configuration", data=json.dumps(config))
        print(f"req-{req}\nstatus_code-{req.status_code}")
        print(f"text-{req.text}\nreason-{req.reason}")
        self.assertEqual(req.status_code, 200)

    def test_messages_from_blacklisted_group(self):
        print("\nMessage from blacklisted group test starting...\n")
        chat_msg = {
            'learning_space': 'test_app',
            'group': "blacklisted_group1",
            'username': "Milhouse",
            'text': 'hi!',
            'timestamp': str(pd.Timestamp.now().asm8)[:26]
        }
        req = requests.post(f"{self.app_url}/message?retrieve_details=true&save=false", data=json.dumps(chat_msg))
        req_json = req.json()
        chat_msg = {
            'learning_space': 'test_app',
            'group': "blacklisted_group1",
            'username': "Bart",
            'text': 'hi!',
            'timestamp': str(pd.Timestamp.now().asm8)[:26]
        }
        req = requests.post(f"{self.app_url}/message?retrieve_details=true&save=false", data=json.dumps(chat_msg))
        req_json = req.json()

        print("\nMessage")
        pprint(chat_msg)
        print("\nResponse")
        pprint(req_json)

        self.assertEqual(req_json['agent_intervention'], None)


    def test_message(self):
        """
        Test the message endpoint of the CLAIR API.
        Ensure that the status code of the response is 201 (Created).
        """
        chat_msg = {
            'learning_space': 'test_app',
            'group': "test",
            'username': "Milhouse",
            'text': 'hi!',
            'timestamp': str(pd.Timestamp.now().asm8)[:26]
        }
        req = requests.post(f"{self.app_url}/message?retrieve_details=true&save=false", data=json.dumps(chat_msg))
        res_json = req.json()

        print("\nMessage")
        pprint(chat_msg)
        print("\nDialogue state")
        pprint(res_json.get('dialogue_state'))
        print("\nFuzzy outputs")
        pprint(res_json.get('fuzzy_output'))

        self.assertEqual(req.status_code, 201)

    # def test_proactive(self):
    #     print("\nProactive message test starting...\n")
    #     chat_msg = {
    #         'learning_space': 'test_app',
    #         'group': "test",
    #         'username': "Student1",
    #         'text': 'hi!',
    #         'timestamp': int(pd.Timestamp.now(tz='utc').value / 10**6)
    #     }
    #     req = requests.post(f"{self.app_url}/message?retrieve_details=true", data=json.dumps(chat_msg))

    #     print('Msg 1:', req.status_code)

    #     ########################################
        
    #     chat_msg = {
    #         'learning_space': 'test_app',
    #         'group': "test",
    #         'username': "Student2",
    #         'text': 'hi student1!',
    #         'timestamp': int(pd.Timestamp.now(tz='utc').value / 10**6)
    #     }
    #     req = requests.post(f"{self.app_url}/message?retrieve_details=true", data=json.dumps(chat_msg))

    #     print('Msg 2:', req.status_code)

    #     # Pull the proactive message immediately after a message
    #     req = requests.get(f"{self.app_url}/pull/{chat_msg['learning_space']}/{chat_msg['group']}")

    #     print('Pull 0:', req.status_code, )
    #     self.assertEqual(req.json()['agent_intervention'], None)
        
    #     ######################################## Wait for five minutes
    #     print("Waiting for 300 seconds...")
    #     time.sleep(300)
        
    #     # Pull the proactive message
    #     req = requests.get(f"{self.app_url}/pull/{chat_msg['learning_space']}/{chat_msg['group']}")

    #     print('Pull 1:', req.status_code, req.text)
    #     self.assertEqual(type(req.json()['agent_intervention']), str)

    #     # Pull the proactive message again
    #     req = requests.get(f"{self.app_url}/pull/{chat_msg['learning_space']}/{chat_msg['group']}")

    #     print('Pull  2:', req.status_code, req.text)
    #     self.assertEqual(req.json()['agent_intervention'], None)

    #     # Pull the proactive message once more
    #     req = requests.get(f"{self.app_url}/pull/{chat_msg['learning_space']}/{chat_msg['group']}")

    #     print('Pull  3:', req.status_code, req.text)
    #     self.assertEqual(req.json()['agent_intervention'], None)

    #     # Send a message
    #     chat_msg = {
    #         'learning_space': 'test_app',
    #         'group': "test",
    #         'username': "Student2",
    #         'text': 'hi student1!',
    #         'timestamp': int(pd.Timestamp.now(tz='utc').value / 10**6)
    #     }
    #     req = requests.post(f"{self.app_url}/message?retrieve_details=true", data=json.dumps(chat_msg))

    #     print('Msg 3:', req.status_code)
    #     self.assertEqual(req.json()['agent_intervention'], None)

    #     # Pull again
    #     req = requests.get(f"{self.app_url}/pull/{chat_msg['learning_space']}/{chat_msg['group']}")

    #     print('Pull 4:', req.status_code, req.text)
    #     self.assertEqual(req.json()['agent_intervention'], None)

    #     # Pull once again
    #     req = requests.get(f"{self.app_url}/pull/{chat_msg['learning_space']}/{chat_msg['group']}")

    #     print('Pull 5:', req.status_code, req.text)

    #     ######################################## Wait for 

    #     print("Waiting for 70 seconds...")
    #     time.sleep(300)

    #     # Pull once more
    #     req = requests.get(f"{self.app_url}/pull/{chat_msg['learning_space']}/{chat_msg['group']}")

    #     print('Pull 6:', req.status_code, req.text)
    #     self.assertEqual(type(req.json()['agent_intervention']), str)

    #     # ########################################


if __name__ == '__main__':
    unittest.main()
