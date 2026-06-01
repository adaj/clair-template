# app

To use the API, you need to follow the steps below. The API is divided into two
main parts: configuration and messaging. The configuration part is used to
setup the agent, and the messaging part is used to send messages and receive
responses.

If you need further support or have any questions, join the Clair community on
[Slack](https://join.slack.com/t/clair-zfn7400/shared_invite/zt-2duu6tcja-HotOBcUIlEZYwaM4BheMBg).


## Step 1: Configuration

> In this step you will focus on using the routes `/configuration` and `/lookups`.

The configuration API is used to setup the agent. The agent will be active only
if the configuration is valid and the `is_active` flag is set to `true`. 

The `configuration/` route expects a POST request with a JSON object containing the following fields:

```
{
  "learning_space": "5ac59b",
  "is_active": true,
  "keywords": [
      "Elektrische circuits",
      "Parallelle of seriële circuits",
      "Weerstand of capaciteit",
      "Spanning of stroom"
  ],
  "mode": "apt-base",
  "language": "NL",
  "blacklist": ["group1", "group2"]
}
```

To make sure you send a valid configuration, check the modes and languages available.
This can be made with GET requests on the following routes:

```
GET /modes
```

```
GET /languages
```


## Step 2: Messaging

> In this step you will focus on using the routes `/message` and `/pull`.

Only after you configured an active agent, you can send POST requests with messages
and wait for a response. 

The `message/` route expects a POST request with a JSON object containing the following fields:

```
{
  "learning_space": '5ac59b',
  "group": "b28qw3",
  "username": "Bart Simpson",
  "text": "Hi Milhouse",
  "timestamp": 1705915781803,
}
```

While the `message/` route is for POST, it returns data, either none or a JSON object with the following fields:

```
{
  "learning_space": '5ac59b',
  "group": "b28qw3",
  "username": "Bart Simpson",
  "text": "Hi Milhouse",
  "timestamp": 1705915781803,
  "dialogue_state": {
      "L1_DOM": 0.5,
      "L1_COO": 0.6,
      "L1_OFF": 0.7,
      ...
  },
  "fuzzy_output": {
      "talk_move_1": 0.5,
      "talk_move_2": 1.0,
      ...
  },
  "agent_intervention": {
      "intervention_type": "greeting",
      "message": "Hi, I am Clair!"
  }
}
```

But in some modes (e.g., `apt-goals`), Clair may also intervene in a proactive way (as oppose to reacting to messages).
To allow for these interventions, the `pull/` route is used. 

The `pull/` route expects a GET request to be made periodically (e.g., twice per minute), indicating the learning_space and group in the URL. For example:
  
```	
GET /pull/5ac59b/b28qw3
```

