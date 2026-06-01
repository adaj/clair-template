import os
import json
from collections import deque
from typing import List, Dict
import yaml
import pandas as pd
import random
import string
from pprint import pprint

# Initialize CONTENT as the content within the file triggering/content


# RotatingQueue for talk moves and followups, so that we can select the next unused variation
class RotatingQueue:
    def __init__(self, iterable):
        self.data = deque(iterable)
    
    def query(self):
        element = self.data.popleft()  # Get the first element
        self.data.append(element)      # Move it to the last spot
        return element

    def __str__(self):
        return json.dumps(list(self.data))
    
    def __repr__(self):
        return str(self)[:15]+'...'


class GroupCache:

    def __init__(self, talk_moves: List[Dict], followups: List[Dict] = None, repetition_window: int = 3):
        # Parse talk moves into adequate data structure for queries
        self.talk_moves = {}
        for item in talk_moves:
            variations = item['value']
            if isinstance(variations, str):
                variations = [variations]
            self.talk_moves[item['id']] = RotatingQueue(variations)
        # Parse followups (if any) into adequate data structure for queries
        self.followups = {}
        if followups:
            for item in followups:
                # Check if followup key matches to a talk move key
                assert item['id'] in self.talk_moves.keys(), f"Follow-up {item} does not have a corresponding talk move to be followed up."
                variations = item['value']
                if isinstance(variations, str):
                    variations = [variations]
                self.followups[item['id']] = RotatingQueue(variations)
        else:
            self.followups = None
        # Initialize talk move data tracking
        self.tm_freqs = {tm: 0 for tm in self.talk_moves}
        self.last_tms = deque(maxlen=repetition_window)
        self.last_tm_addressed_user = None
        self.words_addressed_user = 0
        self.followup_delivered = True 
        self.last_smalltalk = deque(maxlen=5)
        self.msgs_since_last_tm = 0
        self.time_last_tm = pd.to_datetime(0, unit='ms', utc=True) # pd.Timestamp.now(tz='utc')

    def update(self, talk_move, timestamp, followup=None, addressed_user=None, trigger_only_once=None, username=None, text=None):
        # Update cache if valid talk move is given
        if talk_move in self.talk_moves:
            # Check if the talk move is not in the trigger_only_once before updating talk moves state
            # Three conditions:
            # 1 - No trigger_only_once list provided
            # 2 - Talk move not in trigger_only_once list
            # 3 - Talk move in trigger_only_once list, but it's the first time it's being used
            if trigger_only_once is None or \
                    (trigger_only_once is not None and talk_move not in trigger_only_once) or \
                    (trigger_only_once is not None and talk_move in trigger_only_once and self.tm_freqs[talk_move]==0):
                # If this talk move is not restricted by trigger_only_once, update its frequency etc
                self.tm_freqs[talk_move] += 1
                self.last_tms.append(talk_move)
                self.last_tm_addressed_user = addressed_user
                self.words_addressed_user = 0
                self.followup_delivered = False
                self.msgs_since_last_tm = 0

            # check if timestamp is pandas timestamp and it has tz utc
            if isinstance(timestamp, pd.Timestamp) and timestamp.tzinfo.zone == 'UTC':
                self.time_last_tm = timestamp
            else:
                self.time_last_tm = pd.Timestamp.now(tz='utc')
                raise ValueError(f"Timestamp {timestamp} is not a pandas timestamp with UTC timezone.")
            self.time_last_tm = timestamp
        else:
            self.msgs_since_last_tm += 1
            # If comes from addressed user, accumulate the number of words to check for followup 
            # (TODO: check if this is a good idea)
            if username is not None and username == self.last_tm_addressed_user:
                self.words_addressed_user += len(text.split())
        # If valid followup is given, update cache
        if self.followups and followup in self.followups:
            self.followup_delivered = True
            self.msgs_since_last_tm = 0  


class AgentManager:

    AGENT_NAME = "Clair"
    INTERVENTION_MODES = [f for f in os.listdir(os.path.join("triggering", "modes")) if '.' not in f]

    def __init__(self):
        self.blacklist_groups = {}
        self.blacklist_users = set()
        self.group_cache = {}
        self.interventions = {} 
        for mode in self.INTERVENTION_MODES:
            self.interventions[mode] = {}
            for item in ['config', 'talk_moves', 'small_talk', 'followups', 'what_ifs']:
                file_path = os.path.join("triggering", "modes", mode, f'{item}.yml')
                if os.path.exists(file_path):  # Check if the file exists
                    print(f"[AgentManager] - Loading {item} for mode {mode}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        item_data = yaml.safe_load(f)
                    self.interventions[mode][item] = item_data
                # Optional files
                elif item == "followups" or item == "what_ifs": 
                    pass
                else:
                    raise FileNotFoundError(f"File {file_path} for item '{item}' not found!")
        
    def add_blacklist_groups(self, learning_space: str, group: str):
        if learning_space not in self.blacklist_groups:
            self.blacklist_groups[learning_space] = set()
        self.blacklist_groups[learning_space].add(group)

    def rm_blacklist_groups(self, learning_space: str, group: str):
        if learning_space in self.blacklist_groups:
            self.blacklist_groups[learning_space].remove(group)

    def add_blacklist_users(self, username: str):
        self.blacklist_users.add(username)

    def rm_blacklist_users(self, username: str):
        self.blacklist_users.remove(username)
    
    def generate_intervention(self, 
                              learning_space: str,
                              group: str, 
                              username: str, 
                              timestamp: pd.Timestamp,
                              text: str, 
                              intent: str,
                              last_users: List[str],
                              tacc: dict,
                              content_statement: str, 
                              fuzzy_outputs: Dict, 
                              mode: str, 
                              language: str, 
                              verbose=False,):
        """
        This function is how the manager orchestrates the intervention process as a whole.
        """
        assert mode in self.INTERVENTION_MODES, f"Mode {mode} not supported. Supported modes: {self.INTERVENTION_MODES}"
        talk_move, agent_response = None, None # talk_move is the key of the talk move, agent_response is the value
        addressed_user = None
        followup = None
        trigger_only_once = None

        # If talk_moves for the language are not available, use EN instead
        if language not in self.interventions[mode]['talk_moves']:
            print(f"Talk moves for language {language} not available. Using EN instead.")
            language = 'EN'

        # General conditions for not intervening at all
        is_not_group = last_users is not None and len(last_users) < 2 #and "Speaker" not in username # speaker not in username is a quick fix for the voice interface (that use Speaker0,1,2 as username)
        is_from_agent = username == self.AGENT_NAME
        is_user_blacklisted = username in self.blacklist_users
        is_group_blacklisted = group in self.blacklist_groups.get(learning_space, set())
        
        # if verbose is True and username is milhouse, means that we are testing the manager, so skip the silence window
        if verbose and username.lower() in ["milhouse", "bart"]:
            is_silence_window = False
        
        if is_not_group or is_from_agent or is_user_blacklisted or is_group_blacklisted:
            # Clair can't help, nothing else is checked
            return {'selected_move': None, 'text': None}        
        
        # Manager rules for intervention
        print("\n\n[AgentManager] - New message received!") if verbose else None
        # Check if it's the first message of the group
        if group not in self.group_cache:
            print("[AgentManager]  - New group detected") if verbose else None
            # Create cache for this group
            self.group_cache[group] = GroupCache(talk_moves=self.interventions[mode]['talk_moves'][language], 
                                                 followups=self.interventions[mode].get('followups', {}).get(language.upper()[:2]),
                                                 repetition_window=self.interventions[mode]['config']['repetition_window'])
            # # Agent respond with greeting
            agent_response = self.interventions[mode]['small_talk'][language]['greetings']

        # Check if message is a question to Clair
        elif text is not None and intent is not None and ('clair' in text.lower()[:10]):
            # Pick a response if there is any, as defined in the small_talk file
            agent_response = self.interventions[mode]['small_talk'][language].get(intent, None)
            
            if agent_response is not None:
                print("[AgentManager]  - Question to Clair detected") if verbose else None
                # Caching small talk usage, so that it's not repeated too often
                self.group_cache[group].last_smalltalk.append(intent)
                # Check if clair_intent appears more than 3 times in last_smalltalk deque
                if self.group_cache[group].last_smalltalk.count(intent) >= 3:
                    # If so, avoid using it again as agent_response
                    agent_response = None

        # Check if message can be followed up
        elif text is not None and self.is_for_followup(group, username, timestamp, text, mode, verbose) \
            and not self.group_cache[group].followup_delivered:
                last_talk_move = self.group_cache[group].last_tms[-1]
                print(f"[AgentManager]  - Follow-up ({last_talk_move}) opportunity detected") if verbose else None
                followups = self.group_cache[group].followups.get(last_talk_move, None)
                if followups is not None:
                    agent_response = followups.query()
                    followup = last_talk_move
                    
        # Check if message triggers a talk move
        else:
            print(f"[AgentManager]  - Talk move opportunity detected. Checking...") if verbose else None
            text_len = len(text.split(' ')) if text is not None else 0
            is_too_short = username is not None and text_len < self.interventions[mode]['config']['min_words_in_message']
            is_too_long = username is not None and text_len > self.interventions[mode]['config']['max_words_in_message']
            is_silence_window = self.group_cache[group].msgs_since_last_tm < self.interventions[mode]['config']['silence_window']
            ends_with_special_char = text is not None and not (text[-1].isalnum() or text[-1] in string.punctuation)
        
            if not is_too_short and not is_too_long and not is_silence_window and not ends_with_special_char:
                talk_move = self.select_talk_move(group, fuzzy_outputs, mode, verbose)
                if talk_move is not None:
                    trigger_only_once = self.interventions[mode]['config'].get('trigger_output_only_once', None)
                    if trigger_only_once and (talk_move in trigger_only_once) and (self.group_cache[group].tm_freqs[talk_move]>0):
                        pass
                    else: # No restrictions of triggering once
                        # Query group's RotatingQueue to get unused talk move variation
                        agent_response = self.group_cache[group].talk_moves[talk_move].query()

        # Filling the <>s to generate final agent response (TODO: this could be a done in a function)
        if agent_response is not None:
            # Detect who is the speaker and discussant
            speaker = last_users[0]
            if len(last_users) > 1:
                if isinstance(tacc, dict):
                    discussant = [k for k, v in tacc.items() if v == min(tacc.values()) and k != speaker][0]
                elif isinstance(tacc, pd.Series):
                    discussant = tacc.drop(speaker).idxmin()
            else:
                discussant = speaker
            # Cache the addressed username
            if agent_response[:9] == '<speaker>':
                addressed_user = speaker
            elif agent_response[:12] == '<discussant>':
                addressed_user = discussant
            # Post-process usernames
            #   1 - Remove integer numbers that may appear in usernames when splitted by space
            speaker = ' '.join(word for word in speaker.split() if not word.isdigit())
            discussant = ' '.join(word for word in discussant.split() if not word.isdigit())
            #   2 - Remove any trailing spaces
            speaker = speaker.strip()
            discussant = discussant.strip()
            #   3 - Capitalize first letter
            speaker = '-'.join(substring.capitalize() for substring in speaker.split('-'))
            discussant = '-'.join(substring.capitalize() for substring in discussant.split('-'))
            # Finally replace <>s with actual usernames (capitalize first letter)
            if '<speaker>' in agent_response:
                assert speaker is not None, f"Speaker is None, but agent response '{agent_response}' requires a speaker."
                agent_response = agent_response.replace('<speaker>', speaker.capitalize())
            if '<discussant>' in agent_response:
                assert discussant is not None, f"Discussant is None, but agent response '{agent_response}' requires a discussant."
                agent_response = agent_response.replace('<discussant>', discussant.capitalize())
            if '<what_if>' in agent_response:
                agent_response = agent_response.replace('<what_if>', self.interventions[mode]['what_ifs'][language][content_statement]) 

        # Update group cache (which talk move was used, who was addressed, etc)
        self.group_cache[group].update(talk_move, timestamp, followup, addressed_user, trigger_only_once, username, text)

        if verbose:
            print("[AgentManager]  - Group cache updated")
            pprint(self.group_cache[group].__dict__)
            print("[AgentManager]  - Output generated:")
            print({'selected_move': talk_move, 'text': agent_response})

        # Retrieve the selected talk move and agent response
        return {'selected_move': talk_move, 'text': agent_response}
    
    def is_for_followup(self, 
                        group: str, 
                        username: str, 
                        timestamp: pd.Timestamp,
                        text: str, 
                        mode: str, 
                        verbose: bool = False):
        """
        This function is how the manager decides whether a message is a good opportunity for a follow-up.
        """
        # If there is no followup for this mode and language, return False
        if self.interventions[mode].get('followups') is None:
            return False
        # If last intervention is within window of 5 to 180 seconds
        # check if timestamp is pandas timestamp and it has tz utc
        if isinstance(timestamp, pd.Timestamp) and timestamp.tzinfo.zone == 'UTC':
            time_since_last_tm =  timestamp - self.group_cache[group].time_last_tm
        else:
            raise ValueError(f"Timestamp {timestamp} is not a pandas timestamp with UTC timezone.")
        is_within_followup_time_window = 1 < time_since_last_tm.seconds < 120
        # If the current message comes from the addressed user
        is_from_addressed_user = username == self.group_cache[group].last_tm_addressed_user
        # If it's not a question
        is_not_question = '?' not in text
        # If the current message is not too short
        n_words = len(text.split(' '))
        is_not_too_short = n_words > self.interventions[mode]['config']['min_words_in_message']
        # If the last consecutive messages from the user sums up to more than N words
        if is_from_addressed_user:
            words_addressed_user = self.group_cache[group].words_addressed_user + n_words
        else:
            words_addressed_user = 0
        is_more_than_few_words = words_addressed_user > 5 # if the user has said more than 5 words (across multiple messages) - TODO: this should be a config parameter
        # Print followup conditions
        print(f"""
    [AgentManager]   - Conditions for `is_for_followup`? {is_within_followup_time_window and is_from_addressed_user and is_not_question and is_not_too_short and is_more_than_few_words}
        is_within_followup_time_window={is_within_followup_time_window}
        is_from_addressed_user={is_from_addressed_user}
        is_not_question={is_not_question}
        is_not_too_short={is_not_too_short}
        is_more_than_few_words={is_more_than_few_words}
            
        """) if verbose else None
        if is_within_followup_time_window and is_from_addressed_user and is_not_question and is_not_too_short and is_more_than_few_words:
            return True
        return False

    def select_talk_move(self, 
                         group: str,
                         fuzzy_outputs: Dict, 
                         mode: str, 
                         verbose: bool = False):
        """
        This function is how the manager picks one single talk move from the fuzzy outputs.
        """
        selected = None
        candidates = [tm for tm in fuzzy_outputs if fuzzy_outputs[tm] != 0]
        # Check if there are initial candidates and whether any crossed the threshold
        max_val = max(fuzzy_outputs.values())
        print("[AgentManager]   - Talk move all candidates", candidates, max_val, self.interventions[mode]['config']['sensitivity_threshold']) if verbose else None
        if candidates and (max_val >= self.interventions[mode]['config']['sensitivity_threshold']):
            # Check candidates with max_val that were not used before
            max_candidates = [k for k, v in fuzzy_outputs.items() \
                              if (v == max_val) and (k not in self.group_cache[group].last_tms)]
            print("[AgentManager]   - Talk move max candidates", max_candidates) if verbose else None
            if not max_candidates:
                # Means that the moment for triggering is good, but all candidates were used before
                # Get the next highest candidates if any
                filtered_fuzzy_outputs = {k: v for k, v in fuzzy_outputs.items() \
                                          if v < max_val and (k not in self.group_cache[group].last_tms)}
                if filtered_fuzzy_outputs:
                    max_val = max(filtered_fuzzy_outputs.values())
                    max_candidates = [k for k, v in filtered_fuzzy_outputs.items() if (v == max_val) and (v>0.5)]
                print("[AgentManager]   - Talk move under-max candidates", max_candidates) if verbose else None
            if len(max_candidates) >= 1:
                # If there are more than one candidates tied, select the one that was used least
                filtered_freqs = {k: v for k, v in self.group_cache[group].tm_freqs.items() \
                                  if k in max_candidates}
                min_freq = min(filtered_freqs.values())
                selected = [tm for tm in max_candidates if self.group_cache[group].tm_freqs[tm] == min_freq]
                # If they are tied in frequency, select one randomly
                selected = random.choice(selected)
        return selected
        

