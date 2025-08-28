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
import utils
import prompts
import os
import sys


folder_path = "ICL-examples"

if os.path.isdir(folder_path):
    print(f"The folder '{folder_path}' exists. Delete it to start generation")
    sys.exit()
    

my_file = open("words.txt", "r")
data = my_file.read() 
data_into_list = data.replace('\n', ' ').split(" ")
my_file.close() 
sorted_words = sorted(data_into_list, key=len)
sorted_list_five_letters = sorted_words[9770:]

with open("domains.pkl", 'rb') as file:
    domains = pickle.load(file)


random_elements = random.sample(sorted_list_five_letters, 50)
random_domains = random.sample(domains, 20)

def challenge_generator_func(verifier_system_prompt, generation_history, generation_command, words, domains, previous_gate_number=0, model = "o3-mini"):
    if previous_gate_number == 0:
        current_generator_input = """Previous Rebuses - None 
                                new difficulty request - Generate first gate
                                previous gate number = 0
                                domains = """ + str(domains) + """
                                words = """ + str(words)
        
        current_generator_input = re.sub(r'\n +', r'\n', current_generator_input)
    else:
        current_generator_input = """ Previous Rebus -""" 
        for previous_generation in generation_history:
            current_generator_input += "\n" + str(previous_generation[0])
            current_generator_input += "\n" + "difficulty request  - " + str(previous_generation[1])
        current_generator_input += "\n\n" + "new difficulty request - " + str(generation_command)
        current_generator_input += "\n\n"+ "Previous Gate Number - " + str(previous_gate_number)
        current_generator_input += "\n" + "Domains = "+ str(domains)
        current_generator_input += "\n" + "Words = " + str(words)
    
    generator_response = utils.run_command(current_generator_input, verifier_system_prompt, model)
    generator_solution = re.search(r'(?<=Solution = )\w+', generator_response['text']).group(0)
    generator_question = generator_response['text'].replace(generator_solution, ".").lower()

    generator_output = {
        'question': generator_question,
        'solution': generator_solution,
        'generator response': generator_response
    }

    return generator_output

def response_generator_func(challenge, prover_system_promt, model="o3-mini"):
    response = utils.run_command(challenge, prover_system_promt, model)
    response_text = response['text'].replace('“', '"').replace('”', '"').replace("'", '"')
    response_dict = json.loads(re.search(r'\{\s*"Gate.*?}\s*', response_text, re.DOTALL).group(0))
    response_solution = response_dict[list(response_dict.keys())[-1]].lower()
    prover_output = {
        'answer': response_solution, 
        'response dict': response_dict,
        'prover response': response
    }

    return prover_output

current_verifier_command = prompts.generator_difficulty_commands["First"]
gpt4o_correct_responses = []
o3_mini_correct_responses = []
o3_correct_responses = []
o3_incorrect_responses = []
prev_gate_number = 0
verifier_outputs = []
verifier_history = []
prover_response_dict = {}
prover_response_dict["4o"] = []
prover_response_dict["o3-mini"] = []
prover_response_dict["o3"] = []
easy_problems = 0
medium_problems = 0
difficult_problems = 0
minimum_problem_per_difficulty = 1

while(easy_problems <= minimum_problem_per_difficulty and medium_problems <= minimum_problem_per_difficulty and difficult_problems <= minimum_problem_per_difficulty):

    random_elements = random.sample(sorted_list_five_letters, 50)
    random_domains = random.sample(domains, 20)

    if prev_gate_number == 0:
        problem = challenge_generator_func(prompts.icl_verifier_system_prompt, None, None, random_elements, random_domains, 0, "o3")
        verifier_outputs += [problem]
        verifier_history += [[problem["generator response"]['text'], current_verifier_command]]
    else:
        input_history = verifier_history if len(verifier_history) <= 5 else verifier_history[5:]
        problem = challenge_generator_func(prompts.icl_verifier_system_prompt, input_history, current_verifier_command, random_elements, random_domains, prev_gate_number)
        verifier_outputs += [problem]
        verifier_history += [[problem["generator response"]['text'], current_verifier_command]]


    prover_response_gpt4o = response_generator_func(problem["question"], prompts.icl_prover_system_prompt, "gpt-4o")
    prover_response_dict["4o"] += [prover_response_gpt4o]

    prover_response_o3_mini = response_generator_func(problem["question"], prompts.icl_prover_system_prompt, "o3-mini")
    prover_response_dict["o3-mini"] += [prover_response_o3_mini]

    prover_response_o3 = response_generator_func(problem["question"], prompts.icl_prover_system_prompt, "o3")
    prover_response_dict["o3"] += [prover_response_o3]

    current_verifier_command = prompts.generator_difficulty_commands["Maintain"]

    gpt4o_correct = False
    o3_mini_correct = False
    o3_correct = False

    if(utils.solution_verification(prover_response_gpt4o, problem)):
        gpt4o_correct_responses += prover_response_gpt4o["response dict"]
        gpt4o_correct = True
        if easy_problems >=  minimum_problem_per_difficulty:
            current_verifier_command = prompts.generator_difficulty_commands["Increase"]
    elif easy_problems < minimum_problem_per_difficulty:
        current_verifier_command = prompts.generator_difficulty_commands["Reduce"]

    if(utils.solution_verification(prover_response_o3_mini, problem)):
        o3_mini_correct_responses += prover_response_o3_mini["response dict"]
        o3_mini_correct = True
        if medium_problems >=  minimum_problem_per_difficulty:
            current_verifier_command = prompts.generator_difficulty_commands["Increase"]        
    elif medium_problems < minimum_problem_per_difficulty:
        current_verifier_command = prompts.generator_difficulty_commands["Reduce"]


    if(utils.solution_verification(prover_response_o3, problem)):
        o3_correct_responses += prover_response_o3["response dict"]
        o3_correct = True
    elif difficult_problems <minimum_problem_per_difficulty:
        current_verifier_command = prompts.generator_difficulty_commands["Reduce"]


    if gpt4o_correct and o3_mini_correct and o3_correct:
        easy_problems += 1
    elif o3_mini_correct and o3_correct:
        medium_problems += 1
    elif o3_correct:
        difficult_problems += 1

    prev_gate_number += 1

    if (prev_gate_number%5 == 0):
        print("Some progress has been been made")
        
        with open('ICL-examples/low_reasoning.pkl', 'wb')  as file:
            pickle.dump(gpt4o_correct_responses, file)
        with open('ICL-examples/medium_reasoning.pkl', 'wb')  as file:
            pickle.dump(o3_mini_correct_responses, file)
        with open('ICL-examples/high_reasoning.pkl', 'wb')  as file:
            pickle.dump(o3_correct_responses, file)
        with open('ICL-examples/verifier_outputs.pickle', 'wb')  as file:
            pickle.dump(verifier_outputs, file)
        with open('ICL-examples/verifier_history.pickle', 'wb')  as file:
            pickle.dump(verifier_history, file)  
        with open('ICL-examples/prover_response_dict.pickle', 'wb')  as file:
            pickle.dump(prover_response_dict, file)   

with open('ICL-examples/low_reasoning.pkl', 'wb')  as file:
    pickle.dump(gpt4o_correct_responses, file)
with open('ICL-examples/medium_reasoning.pkl', 'wb')  as file:
    pickle.dump(o3_mini_correct_responses, file)
with open('ICL-examples/high_reasoning.pkl', 'wb')  as file:
    pickle.dump(o3_correct_responses, file)
with open('ICL-examples/verifier_outputs.pickle', 'wb')  as file:
    pickle.dump(verifier_outputs, file)
with open('ICL-examples/verifier_history.pickle', 'wb')  as file:
    pickle.dump(verifier_history, file)  
with open('ICL-examples/prover_response_dict.pickle', 'wb')  as file:
    pickle.dump(prover_response_dict, file)   





    