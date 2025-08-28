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

import utils
#import prompt

with open("qdatasets/mixed_difficulty_set.pkl", 'rb') as file:
    compiled_dataset = pickle.load(file)

print("Calculationg Accuracy on mixed difficulty set for GPT-4o:")

accuracy_gpt_4o = utils.accuracy_mesurement(compiled_dataset, "gpt-4o")

print("GPT-4o Accuracy:" + str(accuracy_gpt_4o) + "%")

print("Calculationg Accuracy on mixed difficulty set for o3-mini:")

accuracy_o3_mini = utils.accuracy_mesurement(compiled_dataset, "o3-mini")

print("o3-mini Accuracy:" + str(accuracy_o3_mini) + "%")

print("Calculationg Accuracy on mixed difficulty set for GPT-4o:")

accuracy_o3 = utils.accuracy_mesurement(compiled_dataset, "o3")

print("o3 Accuracy:" + str(accuracy_o3) + "%")