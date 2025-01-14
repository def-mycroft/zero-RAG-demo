"""Functions for handling openai completions"""

import openai

from .imports import * 
from gpt_general.config import load_config

CONFIG = load_config()
with open(CONFIG['api_key_path'], 'r') as f:
    openai.api_key = f.read().strip()


def init_convo(message='you are a helpful assistant'):
    conversation = [
        {'role': 'system', 'content': message}
    ]
    return conversation


def new_message(message, conversation, model='', ):
    if not model:
        raise Exception('must have model')
    conversation.append({'role': 'user', 'content': message})
    response = openai.ChatCompletion.create(
        model=model,
        messages=conversation
    )
    reply = response['choices'][0]['message']['content']
    conversation.append({'role': 'assistant', 'content': reply})
    d = {
        'convo':str(conversation),
        'message':message,
        'reply':reply,
    }
    
    return conversation, reply

