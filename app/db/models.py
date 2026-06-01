from datetime import datetime
from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field, constr, conint
from bson import ObjectId


# mongo collection: agent_configuration
class AgentConfiguration(BaseModel):
    learning_space: constr(max_length=50) # hash
    is_active: bool
    keywords: Optional[List[constr(max_length=200)]]
    mode: Optional[str]
    language: Optional[constr(max_length=5)]
    blacklist: Optional[List[constr(max_length=50)]]
    last_update: datetime = Field(default_factory=datetime.now)

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "learning_space": "5ac59b",
                "is_active": True,
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
        }


# mongo collection: chat_message
class ChatMessage(BaseModel):
    learning_space: constr(max_length=50) # hash
    group: constr(max_length=50) # hash
    username: constr(max_length=50)
    text: str
    timestamp: Union[int, str]
    dialogue_state: Optional[Dict]
    fuzzy_output: Optional[Dict]
    agent_intervention: Optional[Dict]
    last_update: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "learning_space": '5ac59b',
                "group": "b28qw3",
                "username": "Bart Simpson",
                "text": "Hi",
                "timestamp": 1705915781803,
                "dialogue_state": {
                    "L1_DOM": 0.5,
                    "L1_COO": 0.6,
                    "L1_OFF": 0.7,
                    "L2C_IN": 0.5,
                    "L2C_AR": 0.6,
                    "L2C_AI": 0.7,
                    "L2C_AM": 0.5,
                    "L2C_NOS": 0.6,
                    "TSIM": 0.7,
                    "TACC": 0.5,
                    "TIME": 0.6,
                    "PACE": 0.7,
                },
                "fuzzy_output": {
                    "talk_move_1": 0.5,
                    "talk_move_2": 1.0,
                    "talk_move_3": 0.2,
                    "talk_move_4": 0.0,
                },
                "agent_intervention": {
                    "intervention_type": "greeting",
                    "message": "Hi, I am Clair!"
                }
            }
        }