"""
To generate a valid token, enter the docker container (docker exec -it <container> /bin/bash) and run the following command:
python app/db/generate_token.py <partner> <months_to_expire>
"""


import sys
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import secrets
import os

# MongoDB client setup
MONGO_DETAILS = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_DETAILS)
db = client.clair
tokens_collection = db.tokens

async def generate_token(partner: str, months_to_expire: int):
    # Generate a secure token
    token = secrets.token_urlsafe(8)
    expires = datetime.utcnow() + timedelta(days=30 * months_to_expire)
    
    token_data = {
        "partner": partner,
        "token": token,
        "expires": expires
    }
    
    await tokens_collection.insert_one(token_data)
    print(f"Token for {partner}: {token}")
    print(f"Expires on: {expires}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_token.py <partner> <months_to_expire>")
        sys.exit(1)
    
    partner = sys.argv[1]
    months_to_expire = int(sys.argv[2])
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(generate_token(partner, months_to_expire))
