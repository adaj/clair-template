import os
import json

def load_lookups(path: str = None):
    if path is None:
        path = os.path.join("app", "lookups")
    lookups = {}
    for li in os.listdir(path):
        try:
            file_path = os.path.join(path, li)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            key = li.split('/')[-1].replace(".json", "")
            lookups[key] = data
        except Exception as e:
            e.args = (f"Some failure happens at lookup: {li}", *e.args)
            raise
    return lookups


def parse_lookup(config_talk_move):
    new_talk_moves_config = []
    for item in config_talk_move:
        new_talk_moves_config.append({
            'id': item['id'],
            'intervention': \
                get_value(config_talk_move,
                          id=item['id'])
        })
    return new_talk_moves_config


def topics_is_correct(topics):
    if ('keywords' in topics) or \
       ('wiki_page_name' in topics):
       return True
    return False


def get_value(list_of_dicts, id):
    for item in list_of_dicts:
        if item['id'] == id:
            return item['value']
    return None



async def get_api_key(api_key_header: str = Security(api_key_header)):
    token_data = await tokens_collection.find_one({"token": api_key_header})
    if token_data:
        if datetime.utcnow() > token_data["expires"]:
            raise HTTPException(
                status_code=403, detail="Token expired"
            )
        return api_key_header
    raise HTTPException(
        status_code=403, detail="Could not validate credentials"
    )