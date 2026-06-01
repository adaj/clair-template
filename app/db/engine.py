import os
from typing import Union
import motor.motor_asyncio
from fastapi import HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from datetime import datetime


"""
DB settings
"""

MONGO_URI = os.environ['MONGO_URI']
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

# Enter mongodb database name (client.database_name)
database = client.clair

collections = {k: database.get_collection(k) \
                for k in ['chat_messages', 'agent_configurations', 'tokens']}

# index configurations by learning_space
collections['agent_configurations'].create_index('learning_space',
                                                 unique=True)

"""
DB operations
"""

async def create_or_update_one(collection: str,
                               data: dict,
                               upd_filter: Union[dict, bool] = False):
    # Return false if an empty request body is sent.
    if len(data) < 1:
        return False
    to_update = await collection.find_one(upd_filter)
    if to_update:
        updated = await collection.update_one(
            upd_filter, {"$set": data}
        )
        return str(to_update['_id'])
    else:
        to_add = await collection.insert_one(data)
        return str(to_add.inserted_id)


async def retrieve_one(collection: str,
                       where: dict,
                       projection: dict = {}) -> dict:
    if len(projection)==0:
        projection = {'_id': 0}
    to_retrieve = await collection.find_one(where, projection)
    if to_retrieve:
        return to_retrieve


async def retrieve_last(collection: str,
                        where: dict = {},
                        field: str = 'last_update') -> dict:
    to_retrieve = await collection.find(where).sort(field, -1).limit(1)
    return list(to_retrieve)[0]


async def retrieve_all(collection: str,
                       where: dict,
                       projection: dict = {}) -> list:
    if len(projection)==0:
        projection = {'_id': 0}
    cursor = collection.find(where, projection)
    to_retrieve = []
    async for item in cursor:
        to_retrieve.append(item)
    return to_retrieve


# async def retrieve_distinct(collection: str,
#                             field: str,
#                             where: dict) -> list:
#     to_retrieve = await collection.distinct(field, where)
#     return to_retrieve


async def delete(collection:str, where: dict):
    to_delete = await collection.find_one(where)
    if to_delete:
        await collection.delete_one(where)
        return True


# API Key header definition
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header), request: Request = None):
    token_data = await database.tokens.find_one({"token": api_key_header})
    if token_data:
        if datetime.utcnow() > token_data["expires"]:
            raise HTTPException(
                status_code=403, detail="Token expired"
            ) 
        # Extract the route name from the request URL path
        route_name = request.url.path.replace('/chatBot', '')
        route_name = route_name.split('/')[1] if route_name else 'root'
        # Update the route usage count in the database
        await database.tokens.update_one(
            {"token": api_key_header},
            {"$inc": {route_name: 1}}
        )  
        return api_key_header
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )

