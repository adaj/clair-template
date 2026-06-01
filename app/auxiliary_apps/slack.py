import os
from dotenv import load_dotenv
from pathlib import Path
# from fastapi import FastAPI, Depends, Form
from flask import Flask, request, Response, jsonify
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slackeventsapi import SlackEventAdapter
import requests
import json
from pprint import pprint


env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
# Ensure the CLAIR_URL environment variable is set.
CLAIR_URL = os.environ.get('CLAIR_URL')
assert CLAIR_URL and isinstance(CLAIR_URL, str), "Please set the env variable CLAIR_URL with a running clair http app."
CLAIR_URL = CLAIR_URL.rstrip('/')
print("CLAIR_URL: ", CLAIR_URL)


# app = FastAPI(title="Clair Slack API",
#               version="0.1")
app = Flask(__name__)
client = WebClient(token=os.environ.get('SLACK_TOKEN'))
BOT_ID = client.api_call("auth.test")['user_id']
slack_event_adapter = SlackEventAdapter(os.environ.get('SLACK_SIGNING_SECRET'), "/slack/events", app)


@app.route("/slack/commands", methods=["POST"])
def commands():
    payload = request.form
    command = payload.get("command")
    channel = payload.get('channel_id')

    if command == "/reset-session":
        config = requests.get(f"{CLAIR_URL}/configuration/{channel}", timeout=30).json()
        # Deactivate agent under this configuration
        config['is_active'] = False
        req = requests.post(f"{CLAIR_URL}/configuration", data=json.dumps(config), timeout=30)  
        # Activate agent under this configuration
        config['is_active'] = True
        req = requests.post(f"{CLAIR_URL}/configuration", data=json.dumps(config), timeout=30)  
        response = "Session reset successfully."

    elif command == "/set-clair":
        dataset_lang, keywords = payload.get('text').split(maxsplit=1)
        keywords = keywords.replace(',',' ').split()
        agent_configuration = {
            "learning_space": channel,
            "is_active": True,
            "language": dataset_lang,
            "topics": {"keywords": keywords}
        }
        pprint(agent_configuration)

        # Activate agent under this configuration
        req = requests.post(f"{CLAIR_URL}/configuration", data=json.dumps(agent_configuration), timeout=10)
        response = f"Set language to `{dataset_lang}`, and topics to `{' '.join(keywords).strip()}`."

    elif command == "/help-commands":
        response = "Available commands: `/reset-session`, `/set-clair`."

    try:
        # client.chat_postMessage(channel=payload.get('channel_id'), 
        #                         text=response)
        return jsonify({'text': response}), 200
    except SlackApiError as e:
        print(f"Error sending message: {e.response['error']}")
        return jsonify({'error': 'Failed to send help message'}), 500


@slack_event_adapter.on("message")
def slack_events(payload):
    event = payload.get("event", {})
    event_type = payload.get("type")

    if event.get('user') == BOT_ID:
        return {}

    if event_type == "url_verification":
        return {"challenge": payload.get("challenge")}

    if event_type == "event_callback":
        response = None
        if "text" in event and event.get('user') != BOT_ID and event.get('text')[0] != '/':
            chat_msg = {
                    'learning_space': event.get('channel'),
                    'group': event.get('channel'),
                    'username': event.get('user'),
                    'text': event.get('text'),
                    'timestamp': event.get('ts')
                }
            req = requests.post(f"{CLAIR_URL}/message?retrieve_details=true&save=false",
                                data=json.dumps(chat_msg), 
                                timeout=10)
            
            # Here you'd parse the response to decide on what message to send back to slack
            if req.status_code == 201:
                print(1111, req.json())
                print(2222, event)
                response = req.json().get('agent_intervention')['text']
                if response:
                    client.chat_postMessage(channel=event.get('channel'), 
                                            text=response)



if __name__ == "__main__":
    app.run(debug=True, port=8001, host='0.0.0.0')