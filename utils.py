import pandas as pd
import requests
from bs4 import BeautifulSoup
import openai
from typing import List, Dict
import heapq
import math
from openai import OpenAI
from tqdm import tqdm
import random
import json
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)  
import pickle
import glob
from dotenv import load_dotenv
import os
from pathlib import Path
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from typing import Iterable, Callable, Any, List, Tuple, Union, Optional
import time
import prompts

load_dotenv()

api_key_openai = os.getenv('OPENAI_API_KEY')
api_key_google = os.getenv('GOOGLE_API_KEY')
if not api_key_openai:
    api_key_openai = ""
if not api_key_google:
    api_key_google = ""



def run_command(user_prompt, system_prompt=None, model="gpt-4o", prev_response_id=None):
    client = OpenAI(api_key=api_key_google)
    if system_prompt is None:
        messages = [{"role": "user", "content": user_prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    try:
        text = response.choices[0].message.content
        reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
        cached_tokens = response.usage.input_tokens_details.cached_tokens
    except AttributeError:
        text = response.choices[0].message.content
        reasoning_tokens = None
        cached_tokens = None

    return {
        'text': text,
        'cached tokens': cached_tokens,
        'reasoning tokens': reasoning_tokens,
        'entire response': response
    }

def deepseek_run_command(user_prompt, system_prompt=None, model="deepseek-chat", prev_response_id=None):
    client = OpenAI(api_key="sk-690b4388d1a64dd9b8b37ef5c4ac2d81", base_url="https://api.deepseek.com")
    if system_prompt is None:
        messages = [{"role": "user", "content": user_prompt}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    time_to_wait = 5
    for i in range(10):
        try:
            text = response.choices[0].message.content
            reasoning_tokens = None
            cached_tokens = None
        except:
            time.sleep(time_to_wait)
            time_to_wait = time_to_wait*2
            
        else:
            break       


    return {
        'text': text,
        'cached tokens': cached_tokens,
        'reasoning tokens': reasoning_tokens,
        'entire response': response
    }


def gemini_run_command(user_prompt, system_prompt=None, model="gemma-3-27b-it", thinking_budget =None):
    client = genai.Client(api_key=api_key_google)
    time_to_wait = 5
    for i in range(10):
        try:
            if system_prompt is None:
                response = client.models.generate_content(
                    model=model, contents=user_prompt
            )
            elif model == "gemma-3-27b-it":
                user_prompt = system_prompt + user_prompt
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt   
                )
            elif thinking_budget != None:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=GenerateContentConfig(
            system_instruction=[system_prompt],
            thinking_config=genai.types.ThinkingConfig
      (thinking_budget=thinking_budget)
                )
                )                                    
            else:
                response = client.models.generate_content(
                    model=model,
                    contents=user_prompt,
                    config=GenerateContentConfig(
            system_instruction=[system_prompt]),
                )                
        except:
            print("here 2")            
            time.sleep(time_to_wait)
            time_to_wait = time_to_wait*2
        else:
            break
            
    return {'text': response.text,
            'output_tokens':response.usage_metadata.candidates_token_count,
            'response': response}


    
def extract_solution(text):
    """
    Extracts the word following 'solution:' or 'solution =' (case-insensitive).
    """
    pattern = r'solution\s*[:=]\s*([A-Za-z]+)'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None

def solution_verification(prover, problem):
        if(prover["answer"] == problem["solution"]):
            return True
        else:
            return False

from typing import Iterable, Callable, Any, List, Tuple, Union, Optional

Thresholds = Union[int, Tuple[int, int]]

def split_by_value(
    seq: Iterable[Any],
    thresholds: Thresholds,
    *,
    key: Optional[Callable[[Any], int]] = None,
) -> List[list]:
    """
    Split items into buckets based on a numeric measure.

    measure = key(item) if provided; otherwise:
      - len(item) if item supports len()
      - else the item itself (assumed int)

    thresholds = n        -> returns [<= n, > n]
    thresholds = (a, b)   -> returns [<= a, (a, b], > b]   (a and b are sorted)

    Examples:
      split_by_value(words, 6)          # length <= 6 vs > 6
      split_by_value(words, (3, 6))     # length <= 3, 4..6, > 6
      split_by_value(nums, 10, key=int) # numbers <= 10 vs > 10
    """
    # default key: try len(), else identity
    if key is None:
        def key(x):
            try:
                return len(x)  # type: ignore[arg-type]
            except TypeError:
                return x       # assumes x is numeric

    # normalize thresholds
    if isinstance(thresholds, int):
        a, b = thresholds, None
    elif isinstance(thresholds, (tuple, list)) and len(thresholds) == 2:
        a, b = sorted(int(t) for t in thresholds)
    else:
        raise ValueError("thresholds must be an int or a 2-tuple of ints")

    # bucketize
    bucket0, bucket1, bucket2 = [], [], []
    for item in seq:
        m = key(item)
        if b is None:
            (bucket0 if m <= a else bucket1).append(item)
        else:
            if m <= a:
                bucket0.append(item)
            elif m <= b:
                bucket1.append(item)
            else:
                bucket2.append(item)

    return [bucket0, bucket1] if b is None else [bucket0, bucket1, bucket2]

Thresholds = Union[int, Tuple[int, int]]

def split_by_value(
    seq: Iterable[Any],
    thresholds: Thresholds,
    *,
    key: Optional[Callable[[Any], int]] = None,
) -> List[list]:
    """
    Split items into buckets based on a numeric measure.

    measure = key(item) if provided; otherwise:
      - len(item) if item supports len()
      - else the item itself (assumed int)

    thresholds = n        -> returns [<= n, > n]
    thresholds = (a, b)   -> returns [<= a, (a, b], > b]   (a and b are sorted)

    Examples:
      split_by_value(words, 6)          # length <= 6 vs > 6
      split_by_value(words, (3, 6))     # length <= 3, 4..6, > 6
      split_by_value(nums, 10, key=int) # numbers <= 10 vs > 10
    """
    # default key: try len(), else identity
    if key is None:
        def key(x):
            try:
                return len(x)  # type: ignore[arg-type]
            except TypeError:
                return x       # assumes x is numeric

    # normalize thresholds
    if isinstance(thresholds, int):
        a, b = thresholds, None
    elif isinstance(thresholds, (tuple, list)) and len(thresholds) == 2:
        a, b = sorted(int(t) for t in thresholds)
    else:
        raise ValueError("thresholds must be an int or a 2-tuple of ints")

    # bucketize
    bucket0, bucket1, bucket2 = [], [], []
    for item in seq:
        m = key(item)
        if b is None:
            (bucket0 if m <= a else bucket1).append(item)
        else:
            if m <= a:
                bucket0.append(item)
            elif m <= b:
                bucket1.append(item)
            else:
                bucket2.append(item)

    return [bucket0, bucket1] if b is None else [bucket0, bucket1, bucket2]

def solution_verification(prover, problem):
    if(prover["answer"] == problem["solution"]):
        return True
    else:
        return False

def accuracy_mesurement(dataset, model):
    response = []
    response_solution = []
    for problem in dataset['Problems']:
        incorrect_dict = True
        while incorrect_dict:
            response = run_command(problem, prompts.prover_system_prompt, model)
            response_text = response['text'].replace('“', '"').replace('”', '"').replace("'", '"')
            response_dict = re.search(r'\{\s*"Gate.*?}\s*', response_text, re.DOTALL)
            if (response_dict != None):
                try:
                    # Try your code here
                    response_dict = json.loads(response_dict.group(0))
                    incorrect_dict = False
                except:
                    # Optionally handle the error or just pass
                    print("Error occurred, retrying...")
                    pass  
        response += [response_text]
        response_solution += [response_dict]
    correct = 0
    index_incorrect = []
    for index in range(len(dataset['Solutions'])):
        lower_solution = dataset['Solutions'][index].lower()
        response = response_solution[index]['Gate'].lower()
        if lower_solution == response:
            correct += 1
        else:
            index_incorrect += [index]
    
    return (correct/len(dataset))*100
