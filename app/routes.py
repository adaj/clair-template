import os
import asyncio
from typing import List
import pandas as pd
import copy

from fastapi import APIRouter, Body, status, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi_utils.tasks import repeat_every
from fastapi.security.api_key import APIKey

import app.db.engine as database
from app.db.utils import remove_none_keys, type_encoding
from app.db.models import AgentConfiguration, ChatMessage

from nlu.consent.consent import ConSent
from nlu.topics import TopicEmbeddings, create_encoder
from nlu.dialogue import DialogueTracker
from nlu.intents.intent_classifier import IntentClassifier
from triggering.fuzzy import TriggeringMechanism
from triggering.manager import AgentManager


router = APIRouter()
print("====> Clair API and models are loading. Please wait...")

# Read folder "nlu/stop_words" to get the languages available
STOP_WORDS_LANGUAGES = [file.split('.')[0] for file in os.listdir("nlu/stop_words") if file.endswith('.txt')]
print(f"====> Available languages for stop words: {STOP_WORDS_LANGUAGES}")


"""
Instantiate API
"""
@router.on_event("startup")
async def initialize_global_variables():
    global active_configurations, \
            L1_consent, L2C_consent, sentence_encoder, topic_embeddings, dialogues, \
            clair_intent_clf, intent_classifiers, triggering, manager, STOP_WORDS_LANGUAGES

    active_configurations = {} # key: learning_space, values: Dict

    # Triggering
    triggering = TriggeringMechanism()
    manager = AgentManager()
    # NLU
    L1_consent = ConSent(load='nlu/consent/code_L1__v2')
    L2C_consent = ConSent(load='nlu/consent/code_L2C__v2')
    sentence_encoder = create_encoder()
    topic_embeddings = {} # key: learning_space, values: TopicEmbeddings
    dialogues = {} # key: learning_space, values: Dict[group, DialogueTracker]
    print("====> Intent models being loaded...")
    clair_intent_clf = IntentClassifier(load_model='nlu/intents/models/clair_intents_v1', 
                                        handle_punctuation=True)
    intent_classifiers = {
        'L3P': IntentClassifier(load_model='nlu/intents/models/code_L3P_v1'),
        'L3C': IntentClassifier(load_model='nlu/intents/models/code_L3C_v1'),
        'L3V': IntentClassifier(load_model='nlu/intents/models/code_L3V_v1'),
        'L3E': IntentClassifier(load_model='nlu/intents/models/code_L3E_v1'),
        # TODO: Load L4 intent classifier during agent configuration and think about implications
        # 'L4': IntentClassifier(load_model='nlu/intents/models/code_L4_Enzymes_v1')\
        #         .load_stop_words(stop_words_file="nlu/stop_words/EN.txt"),
        'L4': IntentClassifier(load_model='nlu/intents/models/code_L4_Genetics_v1')\
                .load_stop_words(stop_words_file="nlu/stop_words/EN.txt")
    }
    print("====> Intent models loaded.")

    """
    Load active configurations
    """
    # Load all previous configs from the database
    configs = await database.retrieve_all(
        database.collections['agent_configurations'],
        where={"is_active": True}
    )
    for config in configs:
        # TODO: 'topics' will be deprecated, but for now, we are parsing it to 'keywords'
        if 'topics' in config.keys():
            if 'keywords' in config['topics'].keys():
                config['keywords'] = config['topics']['keywords']
            else:
                config['keywords'] = []
            config.pop('topics', None)
        # TODO: Old configuration version without mode means that the mode is 'apt-base'
        if 'mode' not in config.keys():
            config['mode'] = 'apt-base'

        language = config['language'][:2].upper()
        if language not in STOP_WORDS_LANGUAGES:
            continue # Skip this configuration, the language is not supported

        # Setup active_configurations, dialogues, and topic_embeddings
        active_configurations[config['learning_space']] = config
        dialogues[config['learning_space']] = {}
        
        topic_embeddings[config['learning_space']] = TopicEmbeddings(
            encoder=sentence_encoder,
            keywords=config['keywords'],
            stop_words_file=f"nlu/stop_words/{language}.txt"
        )
        intent_classifiers['L4'].load_stop_words(stop_words_file=f"nlu/stop_words/{language}.txt")

        # Check the blacklisting of groups on each active configuration and update the manager
        if 'blacklist' in config.keys():
            manager.blacklist_groups[config['learning_space']] = set(config['blacklist'])

    await asyncio.sleep(5)
    print(f"""
 _____ _       _              _____    ___ 
/  __ \ |     (_)            |  _  |  /   |
| /  \/ | __ _ _ _ __  __   _| |/' | / /| |
| |   | |/ _` | | '__| \ \ / /  /| |/ /_| |
| \__/\ | (_| | | |     \ V /\ |_/ /\___  |
 \____/_|\__,_|_|_|      \_/  \___(_)   |_/

Started at {pd.Timestamp.now(tz='utc').strftime('%d-%m-%Y %H:%M:%S')}""", flush=True)


"""
Clean session data every 12 hours
"""
@repeat_every(seconds=12 * 60 * 60)  # every 12 hours
async def clean_sessions():
    global active_configurations, L1_consent, L2C_consent, \
            topic_embeddings, dialogues, manager, intervention_pulled

    for learning_space in list(active_configurations.keys()):  # Use list to avoid runtime modification issues
        try:
            last_message = await database.retrieve_last(
                database.collections['chat_messages'],
                where={"learning_space": learning_space},
                field="last_update"
            )
            
            if not last_message:
                continue

            time_since_last_message = \
                pd.Timestamp.now(tz='utc') - pd.Timestamp(last_message['last_update'])
                
            if time_since_last_message.total_seconds() > 3 * 60 * 60:
                # Clean ConSent cache
                for dialog_id in dialogues[learning_space].keys():
                    L1_consent.cache.pop(dialog_id, None)
                    L2C_consent.cache.pop(dialog_id, None)
                    # Also clean intervention_pulled cache
                    intervention_pulled.pop(dialog_id, None)
                
                # Drop data of dialogues states and manager's cache
                dialogues[learning_space] = {}
                manager.group_cache.clear()
                
                # Substitute TopicEmbeddings object to save memory
                topic_embeddings[learning_space] = \
                    active_configurations[learning_space]['keywords']
                
                print(f"====> Cleaned cache for learning space: {learning_space}", flush=True)
                
        except Exception as e:
            print(f"====> Error cleaning cache for {learning_space}: {str(e)}", flush=True)
            continue

    print(f"====> Completed session cleaning at {pd.Timestamp.now(tz='utc')}", flush=True) 

"""
Agent configuration
"""
# Create/Update AgentConfiguration
@router.post("/configuration")
async def add_agent_configuration(
        config: AgentConfiguration = Body(...),
        api_key: APIKey = Depends(database.get_api_key)
    ):
    global lookups, L1_consent, L2C_consent, sentence_encoder, topic_embeddings, \
            dialogues, manager, STOP_WORDS_LANGUAGES

    # Parse configuration
    config = config.dict()
    config = jsonable_encoder(config)
    config = remove_none_keys(config)

    if config['is_active'] == False:
        # The request is to deactivate the agent
        # Change topic_embeddins to the keywords to save some memory
        topic_embeddings[config['learning_space']] = config['keywords'] 
        # Clean ConSent cache
        for dialog_id in dialogues.get(config['learning_space'], []):
            L1_consent.cache.pop(dialog_id, None)
            L2C_consent.cache.pop(dialog_id, None)
        # Flush dialogues states and manager cache
        dialogues.pop(config['learning_space'], None)
        manager.group_cache = {}
    else: # is_active == True
        # The request is to activate the agent
        active_configurations[config['learning_space']] = config

        # Instantiate dialogues as an empty dict for new learning_space
        if config['learning_space'] not in dialogues.keys():
            dialogues[config['learning_space']] = {}

        if 'keywords' in config.keys():
            # Check if incoming language attribute is in STOP_WORDS_LANGUAGES else give it EN
            stop_words_language = config['language'][:2].upper()
            if stop_words_language not in STOP_WORDS_LANGUAGES:
                print(f"====> Language {stop_words_language} not supported. Using EN.")
                stop_words_language = 'EN'
            # Setup TopicEmbeddings
            topic_embeddings[config['learning_space']] = \
                TopicEmbeddings(
                    encoder=sentence_encoder,
                    keywords=config['keywords'],
                    stop_words_file=f"nlu/stop_words/{stop_words_language}.txt"
                )
            intent_classifiers['L4'].load_stop_words(stop_words_file=f"nlu/stop_words/{stop_words_language}.txt")
            print(f"====> Topic changed. {topic_embeddings[config['learning_space']].keywords}")

    # Blacklist groups
    if 'blacklist' in config.keys():
        manager.blacklist_groups[config['learning_space']] = set(config['blacklist'])
        # Flush manager cache of those groups
        for group in config['blacklist']:
            manager.group_cache.pop(group, None)

    # Create/Update the configuration received
    await database.create_or_update_one(
        collection=database.collections['agent_configurations'],
        upd_filter={'learning_space': config['learning_space']},
        data=config)
    
    return JSONResponse(status_code=200)


# Retrieve AgentConfiguration
@router.get("/configuration/{learning_space}")
async def get_agent_configuration(learning_space, 
                                  api_key: APIKey = Depends(database.get_api_key)):
    configuration = await database.retrieve_one(
        database.collections['agent_configurations'],
        where={'learning_space': learning_space})
    if configuration:
        return JSONResponse(status_code=200,
                            content=configuration)
    else:
        raise HTTPException(status_code=404,
                            detail=f"{learning_space} not found")


"""
Blacklisting groups and users
"""
# Groups
# Post update in the agent manager to blacklist a group
@router.post("/blacklist/{learning_space}/{group}")
async def blacklist(learning_space: str, group: str, 
                    api_key: APIKey = Depends(database.get_api_key)):
    global manager
    manager.add_blacklist_groups(learning_space, group)
    return JSONResponse(status_code=200)

# Post to remove a group from the blacklist
@router.post("/blacklist/{learning_space}/{group}/remove")
async def remove_blacklist(learning_space: str, group: str, 
                           api_key: APIKey = Depends(database.get_api_key)):
    global manager
    manager.rm_blacklist_groups(learning_space, group)
    return JSONResponse(status_code=200)

# Get the current blacklist of groups
@router.get("/blacklist")
async def get_blacklist_groups(api_key: APIKey = Depends(database.get_api_key)):
    global manager
    blacklist = {id: list(users) for id, users in manager.blacklist_groups.items()}
    return JSONResponse(status_code=200,
                        content=blacklist) 

# Users
# Post update in the agent manager to blacklist a user
@router.post("/blacklist-user/{username}")
async def blacklist_user(username: str,
                         api_key: APIKey = Depends(database.get_api_key)):
    global manager
    manager.add_blacklist_users(username)
    return JSONResponse(status_code=200)

# Post to remove a user from the blacklist
@router.post("/blacklist-user/{username}/remove")
async def remove_blacklist_user(username: str,
                                api_key: APIKey = Depends(database.get_api_key)):
    global manager
    manager.rm_blacklist_users(username)
    return JSONResponse(status_code=200)

# Get the current blacklist of users
@router.get("/blacklist-user")
async def get_blacklist_user(api_key: APIKey = Depends(database.get_api_key)):
    global manager
    return JSONResponse(status_code=200,
                        content=list(manager.blacklist_users)) 



"""
Modes, Languages, and Talk moves
"""
# Get the modes available for the agent
@router.get("/modes")
async def get_modes(api_key: APIKey = Depends(database.get_api_key)):
    global manager
    modes = list(manager.interventions.keys())
    return JSONResponse(status_code=200,
                        content=modes) 

# Get the languages available for a given mode
@router.get("/languages/{mode}")
async def get_languages(mode, api_key: APIKey = Depends(database.get_api_key)):
    global manager
    languages = list(manager.interventions.get(mode, {})\
                                          .get('talk_moves', {})\
                                          .keys())
    return JSONResponse(status_code=200,
                        content=languages) 

# Get the talkmoves available for a given mode and language
@router.get("/talk_moves/{mode}/{language}")
async def get_talk_moves(mode, language, api_key: APIKey = Depends(database.get_api_key)):
    global manager
    talk_moves = manager.interventions.get(mode, {})\
                                      .get('talk_moves', {})\
                                      .get(language, [])
    return JSONResponse(status_code=200,
                        content=talk_moves) 


"""
Agent interventions
"""

"""
Type 1 - Reactive interventions
"""
# Create/Add ChatMessage
@router.post("/message")
async def add_chat_message(message: ChatMessage = Body(...),
                           retrieve_details: bool = True,
                           save: bool = True,
                           api_key: APIKey = Depends(database.get_api_key)):
    global active_configurations, L1_consent, L2C_consent, \
            sentence_encoder, topic_embeddings, dialogues, \
            clair_intent_clf, intent_classifiers, \
            triggering, manager, intervention_pulled

    # Parse message
    message = jsonable_encoder(message)
    # Filter text that is too long
    message['text'] = message['text'][:1000]
    # Make sure the dhialog_id is available (TODO fix this, there shouldn't be a need for dialog_id)
    message['dialog_id'] = message['group'] 

    # If the timestamp received is a string, convert it to a unix timestamp in miliseconds
    if isinstance(message['timestamp'], str):
        print(f"====> Timestamp `str` received: {message['timestamp']}, {type(message['timestamp'])}")
        pandas_timestamp = pd.to_datetime(message['timestamp'], utc=True)
        message['timestamp'] = int(pandas_timestamp.timestamp() * 1000) # unix timestamp in miliseconds
    # If a timestamp received is in unix format, then use the time of the server instead to avoid issues
    else:
        print(f"====> Timestamp unix received: {message['timestamp']}, {type(message['timestamp'])}")
        message['timestamp'] = int(pd.Timestamp.now(tz='utc').timestamp() * 1000) # unix timestamp in miliseconds

    # Make sure DialogTracker is instantiated
    # Check if the config of this learning_space is active
    config = active_configurations.get(message['learning_space'], None)
    if config is not None: # Is active
        # Check if the group has a DialogueTracker
        if message.get('group') not in dialogues.get(message['learning_space'], {}).keys():
            # Create new DialogueTracker for this group
            dialogues[message['learning_space']][message['group']] = \
                DialogueTracker(L1_consent=L1_consent, 
                                L2C_consent=L2C_consent, 
                                topic_embeddings=topic_embeddings[message['learning_space']],
                                intent_classifiers=intent_classifiers)
    else: # Is not active, retrieve the config from the database
        config = await database.retrieve_one(
            database.collections['agent_configurations'],
            where={'learning_space': message['learning_space']}
        )
        # But should it be active?
        if config and config['is_active']:
            # If yes, create new topic_embeddings
            topic_embeddings[message['learning_space']] = TopicEmbeddings(
                encoder=sentence_encoder,
                keywords=config['keywords'],
                stop_words_file=os.path.join("nlu/stop_words",config['language'][:2].upper())
            )
            # Create new DialogueTracker for the learning_space, starting with this group
            dialogues[message['learning_space']] = {
                message['group']: \
                   DialogueTracker(L1_consent=L1_consent, 
                                   L2C_consent=L2C_consent, 
                                   intent_classifiers=intent_classifiers,
                                   topic_embeddings=topic_embeddings[message['learning_space']])
            }
        else: # Means that someone is trying to send a message to an inactive agent
            # Raise an error and stop the process
            raise HTTPException(status_code=404,
                                detail=f"There is no active agent for this "+\
                                        "`learning_space`. Please activate "+\
                                        "using the /configuration interface.")

    # Update dialogue state
    message['dialogue_state'] = \
        dialogues[message['learning_space']][message['group']]\
          .update(**message)\
          .variables
    
    # Temporary bug fix: Clear dialogue variables that are not needed for reactive interventions
    for key in message['dialogue_state']:
        if key in ['TSLM', 'TSLQ']:
            message['dialogue_state'][key] = 0

    # Compute fuzzy output
    message['fuzzy_output'] = \
        triggering.compute(message['dialogue_state'], mode=config['mode'])

    # Compute user intent
    message['intent'] = clair_intent_clf.predict_one(message['text'])

    # Generate intervention with the manager
    message['agent_intervention'] = \
        manager.generate_intervention(
            # Settings
            mode=config['mode'], 
            language=config['language'][:2].upper(),
            # Message variables
            learning_space=message['learning_space'],
            group=message['group'], 
            username=message['username'], 
            timestamp=dialogues[message['learning_space']][message['group']]._state['time_last_message'],
            text=message['text'], 
            intent=message['intent'],
            # Dialogue state variables
            last_users=dialogues[message['learning_space']][message['group']]._last_users,
            tacc=dialogues[message['learning_space']][message['group']]._state['tacc'],
            content_statement=dialogues[message['learning_space']][message['group']]._state['content_statement'],
            # Fuzzy output
            fuzzy_outputs=message['fuzzy_output']
        )
    
    # When a new message is received, intervention_pulled is set to False to allow for new proactive interventions
    intervention_pulled[message['group']] = False

    message = type_encoding(message)
    if save:
        created_message = await database.create_or_update_one(
            database.collections['chat_messages'],
            data=message)
        # If save is set to True, it means Clair is operating with students
        # Then add a short sleep time to make Clair sound more spontaneous/natural
        await asyncio.sleep(2)
    else:
        created_message = None
    if retrieve_details:
        output = {
            'id': str(created_message),
            'selected_move': message['agent_intervention']['selected_move'],
            'agent_intervention': message['agent_intervention']['text'],
            'dialogue_state': message['dialogue_state'],
            'fuzzy_output': message['fuzzy_output'],
            'intent': message['intent']
        }
    else:
        output = {
            'id': str(created_message),
            'agent_intervention': message['agent_intervention']['text'],
        }
    return JSONResponse(status_code=status.HTTP_201_CREATED,
                        content=output)


"""
Type 2 -Proactive interventions
"""

intervention_pulled = {} # key: group, value: bool

@router.get("/pull/{learning_space}/{group}")
async def pull_intervention(learning_space: str, group: str, timestamp: int = None,
                            api_key: APIKey = Depends(database.get_api_key)):
    global active_configurations, dialogues, triggering, manager, intervention_pulled
    output = {
        'id': None,
        'selected_move': None,
        'agent_intervention': None,
        'variables': None
    }
    if timestamp is None:
        timestamp = pd.Timestamp.now(tz='utc')
    elif not isinstance(timestamp, (int, float)):
        raise HTTPException(status_code=400,
                            detail=f"Timestamp must be an integer or float (UNIX timestamp in miliseconds). Received: {timestamp}.")
    # Check if the config of this learning_space is active and the group has a DialogueTracker
    config = active_configurations.get(learning_space, None)
    if config is not None and config['is_active']:
        if group in dialogues.get(learning_space, {}).keys():
            # Update time-based variables in the dialogue state 
            dialogues[learning_space][group].update_time_since(timestamp)
            # Retrieve the variables
            variables_copy = copy.deepcopy(dialogues[learning_space][group].variables)
            
            # Clear dialogue variables that are not needed for proactive interventions
            for key in variables_copy:
                if key not in ['TSLM', 'TSLQ']:
                    variables_copy[key] = 0

            # Temporaryr bug fix: if TSLM is maxed out, set it to zero again
            if variables_copy['TSLM'] > 500:
                variables_copy['TSLM'] = 0

            output['variables'] = variables_copy
            # If an intervention was not recently pulled, check if there is one
            if not intervention_pulled[group]:
                # Compute fuzzy output and check if there is an intervention
                fuzzy_output = triggering.compute(variables_copy, mode=config['mode'])
                agent_intervention = manager.generate_intervention(
                    verbose=False, 
                    learning_space=learning_space,
                    group=group, 
                    username=None, 
                    timestamp=timestamp,
                    text=None, 
                    intent=None,
                    last_users=dialogues[learning_space][group]._last_users,
                    tacc=dialogues[learning_space][group]._state['tacc'],
                    content_statement=dialogues[learning_space][group]._state['content_statement'], 
                    fuzzy_outputs=fuzzy_output, 
                    mode=config['mode'], 
                    language=config['language'][:2].upper())
                if agent_intervention['text'] is not None:
                    output['selected_move'] = agent_intervention['selected_move'] # None or str
                    output['agent_intervention'] = agent_intervention['text'] # None or str
                    intervention_pulled[group] = True
                    # Extra guard to avoid consecutive proactive intervetions:
                    dialogues[learning_space][group]._state['time_last_message'] = timestamp #pd.Timestamp.now(tz='utc')
                    dialogues[learning_space][group]._state['time_last_question'] = timestamp #pd.Timestamp.now(tz='utc')
                    dialogues[learning_space][group].update_time_since(timestamp)
                    print(f"\n\n====> Proactive intervention ({learning_space} @ {group}): \n{output}\n\n")

    # Prevent caching of the response by the client
    response = JSONResponse(status_code=status.HTTP_200_OK,
                            content=output)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response
