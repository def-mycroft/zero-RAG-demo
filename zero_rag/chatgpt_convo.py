"""Functions for handling openai completions"""

import openai

from gpt_general.config import load_config

CONFIG = load_config()
with open(CONFIG['api_key_path'], 'r') as f:
    openai.api_key = f.read().strip()


def init_convo(prompt='you are a helpful assistant'):
    """Return an initial conversation object

    The prompt param will be the first prompt received by ChatGPT. 

    """
    conversation = [
        {'role': 'system', 'content': prompt}
    ]
    return conversation


def new_prompt(prompt, conversation, model=''):
    """Given a conversation/prompt, return an updated conversation"""
    if not model:
        raise Exception('must have model')
    conversation.append({'role': 'user', 'content': prompt})
    response = openai.ChatCompletion.create(
        model=model,
        messages=conversation
    )
    reply = response['choices'][0]['message']['content']
    conversation.append({'role': 'assistant', 'content': reply})

    return conversation, reply
