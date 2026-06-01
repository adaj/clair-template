from collections import deque
from typing import List, Dict
import numpy as np
import pandas as pd
import time
import pytz
from pprint import pprint

from nlu.consent.consent import ConSent
from nlu.topics import TopicEmbeddings
from nlu.intents.intent_classifier import IntentClassifier

# Time to truncate time_spent to TIME_CEIL
TIME_CEIL = 5 * (60*60) # 5 hours

# Time to add to time_spent when users were idle
TIME_IDLE = 10 * (60)    # 10 minutes


class TimeWindowFrequencyCounter:
    def __init__(self, freq_time_window: int):
        self.deque = deque()
        self.freq_time_window = freq_time_window
        self.freq = 0

    def append(self, timestamp: pd.Timestamp = None):
        if timestamp is None:
            current_time = pd.Timestamp.now(tz='utc')
        else:
            current_time = timestamp
        self.deque.append(current_time)
        self.freq = len(self.deque)
        return self

    def get_current_freq(self, current_time: pd.Timestamp = None):
        if current_time is None:
            current_time = pd.Timestamp.now(tz='utc')
        else:
            current_time = current_time
        while self.deque and (current_time - self.deque[0]).total_seconds() > self.freq_time_window:
            self.deque.popleft()
        self.freq = len(self.deque)
        return self.freq
    

def gini(data):
    """
    Calculates the Gini coefficient, a measure of inequality. Basic a assumptions of the Gini coefficient:
    - The number of times each student talks is non-negative.
    - The distribution of talk turns is not heavily skewed, meaning that one or a few students do not dominate the conversation significantly more than others.

    Args:
      data: A list or pandas Series representing the data for which to calculate the Gini coefficient.

    Returns:
      The Gini coefficient, a value between 0 and 1.
    """
    data = np.array(data)
    if len(data) == 1:	
        return 0 # Perfect equality
    elif len(data) == 2:
        if data[0] == data[1]:
            return 0 # Perfect equality
        else:
            return 1 # Perfect inequality
    sorted_data = np.sort(data)
    n = len(data)
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * sorted_data)) / (n * np.sum(data))
    return gini
    

class DialogueTracker:
    """
    DialogueTracker class for generating dialogue variables.
    """
    def __init__(self,
                 L1_consent: ConSent,
                 L2C_consent: ConSent,
                 topic_embeddings: TopicEmbeddings,
                 intent_classifiers: Dict[str, IntentClassifier] = None,
                 tsim_dom_threshold: float = 0.2,
                 freq_time_window: int = 5*60, # seconds
                 sqrt_transformation: List[str] = ["L1_DOM", "L2C_AR", "L2C_AM"]):
        self._L1_consent = L1_consent
        self._L2C_consent = L2C_consent
        self._intent_classifiers = intent_classifiers
        self._topic_embeddings = topic_embeddings
        self._tsim_dom_threshold = tsim_dom_threshold
        self._freq_time_window = freq_time_window
        self._sqrt_transformation = sqrt_transformation

        # Initialize
        self._dialogue_variables = {}
        self._time_start = None
        self._last_users = []
        self._timewindow_freqs = {
            "FIP": TimeWindowFrequencyCounter(freq_time_window),
            "FCU": TimeWindowFrequencyCounter(freq_time_window),
            "FCD": TimeWindowFrequencyCounter(freq_time_window),
            "FCC": TimeWindowFrequencyCounter(freq_time_window),
            "FNV": TimeWindowFrequencyCounter(freq_time_window),
            "FNE": TimeWindowFrequencyCounter(freq_time_window),
            "FCQ": TimeWindowFrequencyCounter(freq_time_window)
        }
        self._state = {}
        self._state['codes'] = {'L1': {}, 'L2C': {}}
        self._state['tsim'] = 0.                          
        self._state['tacc'] = pd.Series(dtype=np.float32) 
        self._state['nmsg'] = pd.Series(dtype=np.int64)   
        self._state['time_spent'] = 1                     # in seconds
        self._state['pace'] = 0.                          # messages per minute #TODO the implementation is wrong, but it's not used anywhere
        self._state['time_last_message'] = None           # pd.Timestamp
        self._state['time_last_question'] = None          # pd.Timestamp
        self._state['time_since_last_message'] = 0        # in seconds
        self._state['time_since_last_question'] = 0       # in seconds
        self._state['consecutive_off_counter'] = 0        # int

    @property
    def variables(self):
        return self._dialogue_variables
    
    def update_time_since(self, timestamp: pd.Timestamp = None):
        # This function is called every time a message is received
        # But also, it can be called manually to update the time_since variables.
        # This is useful when the dialogue tracker is not receiving messages (timestamp=None)
        if self._time_start is not None:
            if timestamp is None:
                timestamp = pd.Timestamp.now(tz='utc')
            self._state['time_since_last_message'] = \
                (timestamp - self._state['time_last_message']).seconds
            self._state['time_since_last_question'] = \
                (timestamp - self._state['time_last_question']).seconds
            # Update dialogue variables
            self._dialogue_variables['TSLM'] = min(self._state['time_since_last_message'], TIME_IDLE)
            self._dialogue_variables['TSLQ'] = min(self._state['time_since_last_question'], TIME_IDLE)
        return self

    def update(self, group, username, timestamp, text, **kwargs):
        """
        Update dialogue variables based on a message.
        """
        if username == 'Clair':
            # Send warning so that the user of this code can avoid sending Clair messages to the tracker
            print("[DialogueTracker] Warning: Clair messages should not be sent to DialogueTracker.")
            return self
        
        # Parse times as pd.Timestamp
        # If the timestamp is a UNIX timestamp in miliseconds, transform it to pd.Timestamp
        if isinstance(timestamp, (int,float)):
            # Transform UNIX timestamp in miliseconds to pd.Timestamp
            timestamp = pd.to_datetime(timestamp, unit='ms', utc=True)
        elif isinstance(timestamp, str):
            # Transform string timestamp to pd.Timestamp
            # Sometimes pd.to_datetime returns a numpy.str_
            timestamp = pd.Timestamp(str(timestamp), tz='utc')
        else:
           raise ValueError(f"timestamp must be UNIX timestamp in miliseconds (int or float). Received: {timestamp}.")
        if not isinstance(timestamp, pd.Timestamp):
            raise ValueError(f"timestamp must be pd.Timestamp. Received: {timestamp}, type={type(timestamp)}.")   

        # Update time_spent
        if self._time_start is None:
            self._time_start = timestamp 
            # Initialize time_last variables
            self._state['time_last_message'] = timestamp
            self._state['time_last_question'] = timestamp 
        else:
            time_delta = (timestamp - self._state['time_last_message'])
            if time_delta.seconds > TIME_CEIL:
                self._state['time_spent'] = 1 # reset, it's a new session
            elif time_delta.seconds > TIME_IDLE:
                self._state['time_spent'] += TIME_IDLE/2
            elif time_delta.seconds >= 0:
                self._state['time_spent'] += time_delta.seconds # accumulate
            else:
                raise ValueError(f"timestamp must be greater than time_last_message, not {timestamp} < {self._state['time_last_message']}")

        # Update time_since variables
        # If a question comes, update time_since_last_question
        if "?" in text or self._state['codes']['L2C'].get('AI', 0) > 0.3:
            self._state['time_last_question'] = timestamp # upd every question
        # Update time_since_last_message
        self._state['time_last_message'] = timestamp # upd every msg
        self.update_time_since(timestamp)

        # Count messages per user
        if username not in self._last_users and username != 'Clair':
            self._last_users.append(username)
            self._state['nmsg'][username] = 0
            self._state['tacc'][username] = 0.
        self._state['nmsg'][username] += 1

        # Compute pace (number of messages per minute)
        if self._state['time_spent'] > 0:
            self._state['pace'] = (self._state['nmsg'].sum() \
                                / (self._state['time_spent'] / 60))

        # Update last_users, with current username in the front (position 0)
        current_position = self._last_users.index(username)
        self._last_users.insert(0, self._last_users.pop(current_position))

        # Compute ConSent probabilities of each code
        _, L1_pred = self._L1_consent.predict_proba(group, username, text)
        self._state['codes']['L1'] = \
            {code: L1_pred[0][i]  for i, code in enumerate(self._L1_consent.config.codes)}
        _, L2C_pred = self._L2C_consent.predict_proba(group, username, text)
        self._state['codes']['L2C'] = \
            {code: L2C_pred[0][i] for i, code in enumerate(self._L2C_consent.config.codes)}

        # Compute tsim 
        self._state['tsim'] = self._topic_embeddings.get_topic_similarity(text)
        
        # If tsim_dom_threshold is set, update L1 and L2C codes accordingly
        if  self._tsim_dom_threshold is not None and \
            self._state['tsim'] > self._tsim_dom_threshold:
                # L1 (if TSIM is high, L1 should not be OFF)
                self._state['codes']['L1']['OFF'] = 0
                max_code = max(self._state['codes']['L1'], key=self._state['codes']['L1'].get)
                # Update ConSent cache of previous codes so that it knows about it too
                self._L1_consent.cache[group]['previous_codes'].popleft()
                self._L1_consent.cache[group]['previous_codes'].appendleft(max_code)
                # L2C (if TSIM is high, L2C should not be NOS or AM)
                self._state['codes']['L2C']['NOS'] = 0
                self._state['codes']['L2C']['AM'] = 0
                max_code = max(self._state['codes']['L2C'], key=self._state['codes']['L2C'].get)
                self._L2C_consent.cache[group]['previous_codes'].popleft()
                self._L2C_consent.cache[group]['previous_codes'].appendleft(max_code)

        # Compute intents and variables derived from them
        if self._intent_classifiers:
            for code, intent_clf in self._intent_classifiers.items():
                _, self._state['codes'][code] = intent_clf.predict_one(text, get_certainty='all')

            # Compute SSIM (standard statement with highest probability)
            if 'L4' in self._state['codes']:
                self._state['codes']['L4'].pop('none of the specified', None)
                statement, prob = max(self._state['codes']['L4'].items(), 
                                      key=lambda x: x[1])
                self._state['content_statement'] = statement
                self._state['L4_SS'] = prob

            # Compute frequency-based variables
            # FIP Individual Perspective - Frequency of L3P_"individual perspective" within the time window
            if max(self._state['codes']['L3P'], key=self._state['codes']['L3P'].get)=='individual perspective':
                self._timewindow_freqs['FIP'].append(timestamp=timestamp)

            # FCO COnfusion - Frequency of (L3C_confusion) within the time window
            if max(self._state['codes']['L3C'], key=self._state['codes']['L3C'].get)=='confusion':
                    self._timewindow_freqs['FCU'].append(timestamp=timestamp)

            # FCD Confusion on the domain - Frequency of L3C==confusion and L1_DOM==high
            if max(self._state['codes']['L3C'], key=self._state['codes']['L3C'].get)=='confusion' \
                and max(self._state['codes']['L1'], key=self._state['codes']['L1'].get)=='DOM':
                    self._timewindow_freqs['FCD'].append(timestamp=timestamp)

            # FCC Confusion on coordination - Frequency of L3C==confusion and L1_COO==high
            if max(self._state['codes']['L3C'], key=self._state['codes']['L3C'].get)=='confusion' \
                and max(self._state['codes']['L1'], key=self._state['codes']['L1'].get)=='COO':
                    self._timewindow_freqs['FCC'].append(timestamp=timestamp)

            # FNV Negative Valence - Frequency of L3V_"negative valence" within the time window
            if max(self._state['codes']['L3V'], key=self._state['codes']['L3V'].get)=='negative valence':
                self._timewindow_freqs['FNV'].append(timestamp=timestamp)

            # FNE Negative Emotion - Frequency of L3E_"negative emotion" within the time window
            if max(self._state['codes']['L3E'], key=self._state['codes']['L3E'].get)=='negative_emotion':
                self._timewindow_freqs['FNE'].append(timestamp=timestamp)

            # FCQ Content Questions - Frequency of (L1_DOM and L2C_AI) within the time window
            if max(self._state['codes']['L1'], key=self._state['codes']['L1'].get)=='DOM' \
                and max(self._state['codes']['L2C'], key=self._state['codes']['L2C'].get)=='AI' \
                and self._state['tsim'] > 0.05:
                    self._timewindow_freqs['FCQ'].append(timestamp=timestamp)

        # Compute tacc
        self._state['tacc'][username] += self._state['tsim']

        # Compute consecutive number of messages with L1 as OFF
        if max(self._state['codes']['L1'], key=self._state['codes']['L1'].get)=='OFF':
            self._state['consecutive_off_counter'] += 1
        else:
            self._state['consecutive_off_counter'] = 0

        # Serialize everything into a dict of dialogue variables (str:float)
        self._dialogue_variables = {
            # ConSent variables
            "L1_DOM": self._state['codes']['L1']['DOM'],
            "L1_COO": self._state['codes']['L1']['COO'],
            "L1_OFF": self._state['codes']['L1']['OFF'],
            "L2C_IN": self._state['codes']['L2C']['IN'],
            "L2C_AR": self._state['codes']['L2C']['AR'],
            "L2C_AI": self._state['codes']['L2C']['AI'],
            "L2C_AM": self._state['codes']['L2C']['AM'],
            "L2C_NOS": self._state['codes']['L2C']['NOS'],
            # Topic embedding variables
            "TSIM": self._state['tsim'],
            "TACC": relative_ratio(self._state['tacc'], self._last_users[0]),
            # Time-based variables
            "TIME": min(self._state['time_spent'], TIME_CEIL),
            "PACE": self._state['pace'],
            "TSLM": min(self._state['time_since_last_message'], TIME_IDLE),
            "TSLQ": min(self._state['time_since_last_question'], TIME_IDLE),
            # Intent classifier variables
            "L3_IP": self._state['codes']['L3P']['individual perspective'] if 'L3P' in self._state['codes'] else 0.0,
            "L3_CU": self._state['codes']['L3C']['confusion'] if 'L3C' in self._state['codes'] else 0.0,
            "L3_NV": self._state['codes']['L3V']['negative valence'] if 'L3V' in self._state['codes'] else 0.0,
            "L3_NE": self._state['codes']['L3E']['negative_emotion'] if 'L3E' in self._state['codes'] else 0.0,
            "L4_SS": self._state['L4_SS'] if 'L4_SS' in self._state else 0.0,
            # Frequency-based variables
            "COFF": self._state['consecutive_off_counter'],
            "GINI": gini(self._state['nmsg']),
            "FIP": self._timewindow_freqs['FIP'].get_current_freq(current_time=timestamp),
            "FCU": self._timewindow_freqs['FCU'].get_current_freq(current_time=timestamp),
            "FCD": self._timewindow_freqs['FCD'].get_current_freq(current_time=timestamp),
            "FCC": self._timewindow_freqs['FCC'].get_current_freq(current_time=timestamp),
            "FNV": self._timewindow_freqs['FNV'].get_current_freq(current_time=timestamp),
            "FNE": self._timewindow_freqs['FNE'].get_current_freq(current_time=timestamp),
            "FCQ": self._timewindow_freqs['FCQ'].get_current_freq(current_time=timestamp),
        }

        # Apply sqrt transformation to the variables indicated
        if self._sqrt_transformation:
            for var in self._sqrt_transformation:
                self._dialogue_variables[var] = np.sqrt(self._dialogue_variables[var])

        return self


def relative_ratio(data_df, user):
    x = data_df[user]
    y = data_df[list(set(data_df.index) - set([user]))].sum().mean()
    if y is np.nan:
        y = 0
    epsilon = np.finfo(float).eps
    return (x + epsilon) / (x + y + epsilon)

