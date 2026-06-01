"""
Script: test_nlu.py

This script contains unittests for the natural language understanding (NLU) components of the system.
It covers functionality like ConSent, topic embeddings, and dialogue state tracking.

To run the tests:
    python tests/test_nlu.py
"""

import sys
import os
import unittest
import time
from pprint import pprint
import pandas as pd
from datetime import timedelta

# Configure environment for TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

sys.path.append('.')

from nlu.consent.consent import ConSent
from nlu.topics import TopicEmbeddings, create_encoder
from nlu.dialogue import DialogueTracker, TimeWindowFrequencyCounter
from nlu.intents.intent_classifier import IntentClassifier


class TestTimeWindowFrequencyCounter(unittest.TestCase):
    """
    Test class for validating TimeWindowFrequencyCounter with pretty-printed information.
    """

    def setUp(self):
        # 10-second time window for testing
        self.freq_time_window = 10
        self.counter = TimeWindowFrequencyCounter(self.freq_time_window)
        print("\n============================")
        print("Initializing TestTimeWindowFrequencyCounter")
        print("============================")

    def tearDown(self):
        print("============================")
        print("TestTimeWindowFrequencyCounter Completed")
        print("============================")

    def test_count_drop_to_zero_after_window(self):
        # 120-second time window for testing
        self.freq_time_window = 120
        self.counter = TimeWindowFrequencyCounter(self.freq_time_window)
        """Test that the frequency drops to zero 125 seconds after a single count, without using time.sleep."""
        print("Testing frequency drop to zero 125 seconds after a single count, without time.sleep...")
        
        # Initial timestamp
        initial_timestamp = pd.Timestamp.now(tz='utc')
        self.counter.append(initial_timestamp)  # Count at the initial timestamp

        # Simulate 125 seconds later by creating a future timestamp
        future_timestamp = initial_timestamp + timedelta(seconds=125)

        # Count using the future timestamp, which should cause the deque to be evaluated and old times removed
        freq_after_wait = self.counter.get_current_freq(future_timestamp)

        self.assertEqual(freq_after_wait, 0, "Expected frequency to drop to 0 after 125 seconds.")
        print("Test passed: Frequency correctly dropped to 0 after 125 seconds, using timestamps.\n")


class TestConSent(unittest.TestCase):
    """
    Test class for validating ConSent.
    """

    def setUp(self):
        print("> RUNNING TestConSent...")
        self.consent_L1 = ConSent(load='nlu/consent/code_L1__v2')
        self.consent_L2C = ConSent(load='nlu/consent/code_L2C__v2')

    def tearDown(self):
        print("> TestConSent COMPLETED")

    def test_training_and_inference_L1(self):
        """Test inference for consent_L1."""
        preds = self.consent_L1.predict_sequence(self._get_test_sequence())
        print('L1')
        pprint(preds)

    def test_training_and_inference_L2C(self):
        """Test inference for consent_L2C."""
        preds = self.consent_L2C.predict_sequence(self._get_test_sequence())
        print('L2C')
        pprint(preds)

    @staticmethod
    def _get_test_sequence():
        """Provide a sample sequence for testing."""
        return [
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'hoi'},
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'what we have to do?'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'I think we need to wait'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'or study the first question'},
            {'dialog_id': '4935ab', 'username': 'Bart', 'text': 'yes what is the frequency?'},
            {'dialog_id': '4935ab', 'username': 'Milhouse', 'text': 'I think 0.5'}
        ]


class TestTopicEmbeddings(unittest.TestCase):
    """
    Test class for validating TopicEmbeddings.
    """

    def setUp(self):
        print("> RUNNING TestTopicEmbeddings...")
        self.encoder = create_encoder()

    def tearDown(self):
        print("> TestTopicEmbeddings COMPLETED")

    def test_topic_similarity(self):
        """Test topic similarity generation."""
        keywords = ["elektrische circuits", "parallelle of serial circuits", "weerstand capaciteit", "spanning stroom"]

        topic_embeddings = TopicEmbeddings(self.encoder, 
                                           keywords=keywords,
                                           stop_words_file="nlu/stop_words/NL.txt")
        self.assertTrue(topic_embeddings.get_topic_similarity(text='hoi') < 0.1)
        self.assertTrue(topic_embeddings.get_topic_similarity(text='circuits') > 0.1)


class TestDialogueState(unittest.TestCase):
    """
    Test class for validating DialogueTracker.
    """

    def setUp(self):
        print("> RUNNING TestDialogueState...")
        L1_consent = ConSent(load='nlu/consent/code_L1__v2')
        L2C_consent = ConSent(load='nlu/consent/code_L2C__v2')
        # intent_classifiers = {
        #     'L3P': IntentClassifier(load_model='nlu/intents/models/code_L3P_L1'),
        #     'L3C': IntentClassifier(load_model='nlu/intents/models/code_L3C_L1'),
        #     'L3V': IntentClassifier(load_model='nlu/intents/models/code_L3V_L1'),
        #     'L4': IntentClassifier(load_model='nlu/intents/models/code_L4_Enzymes_L1'),
        # }

        encoder = create_encoder()
        topic_embeddings = TopicEmbeddings(encoder, keywords=[
            "elektrische circuits",
            "parallelle of serial circuits",
            "weerstand capaciteit",
            "spanning stroom"
        ])
        
        self.dialogue = DialogueTracker(L1_consent=L1_consent, 
                                        L2C_consent=L2C_consent, 
                                        # intent_classifiers=intent_classifiers,
                                        topic_embeddings=topic_embeddings)

    def tearDown(self):
        print("> TestDialogueState COMPLETED")

    def test_message(self):
        """Test dialogue state update based on a message."""
        message = {
            'learning_space': 'test',
            'group': 'PilotTesting_test_group',
            'username': 'Milhouse',
            'text': 'I think we need to wait',
            'timestamp': '2020-12-01 10:00:00'
        }
        self.dialogue.update(**message)
        message['dialogue_state'] = self.dialogue.variables
        pprint(message)
        message = {
            'learning_space': 'test',
            'group': 'PilotTesting_test_group',
            'username': 'Milhouse',
            'text': 'I think we need to wait (again)',
            'timestamp': '2020-12-02 10:00:00'
        }
        self.dialogue.update(**message)
        message['dialogue_state'] = self.dialogue.variables
        pprint(message)

    def test_time_since(self):
        """Test dialogue state update based on a message."""
        message = {
            'learning_space': 'test',
            'group': 'PilotTesting_test_group',
            'username': 'Milhouse',
            'text': 'I think we need to wait',
            'timestamp': pd.Timestamp.now(tz='utc')
        }
        self.dialogue.update(**message)
        print('Variables step 1:')
        pprint(self.dialogue.variables)
        time.sleep(10)
        self.dialogue.update_time_since()
        print('Variables step 2:')
        pprint(self.dialogue.variables)
        



if __name__ == '__main__':
    unittest.main()
