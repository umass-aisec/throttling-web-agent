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
import sys
#sys.path.append('cleaned_version_usenix')
import os
from pathlib import Path

def construct_user_model_interaction(sample, difficulty, sampled_domains):
    question = sample['question']
    clean_question = re.sub(r'(?m)^\d+\.\s*', '', question)[:-12]
    extracted_domains = re.findall(r'•\s*(.*?)\s*:', clean_question)
    my_file = open("../words_alpha.txt", "r")
    data = my_file.read() 
    data_into_list = data.replace('\n', ' ').split(" ")
    my_file.close() 
    sorted_words = sorted(data_into_list, key=len)
    allowed_words = sorted_words[9770:]
    random_words = random.sample(allowed_words, 50)
    final_domain_list = extracted_domains + sampled_domains
    random.shuffle(final_domain_list)

    if difficulty == "easy":
        words = [word for word in random_words if len(word) <= 6] + [sample['solution']]
        random.shuffle(words)
    else:
        words = [word for word in random_words if len(word) > 6] + [sample['solution']]
        random.shuffle(words)        
    
    user_input = f""" Domains:{final_domain_list}
words:{words} 
difficulty:{difficulty}
        """
    model_output = f"""{clean_question}
solution:{sample['solution']} 
""" 
    final_prompt = f"""User Input - 
{user_input} 
Output -
{model_output}
"""
    return final_prompt

def construct_system_prompt(example_directory="ICL-examples"):

    new_verifier_system_prompt = """can you create a so called "reasoning rebus". Can you create a puzzle (or gate) that tests knowledge across all the disciplines to hide text. Furthermore, make it layered, that only by solving previous part (called gates) you will get to the next word or gate to solve, so it should be very iterative. You will first print entire gate and then print the solution like:
    gate = {The entire rebus}
    Solution = {The solution to the above rebus}.

    You will provided a list of domains formatted like this: 
    domains  = [list of domains you can use]
    For each letter in the word contruct the clue by randomly sampling a new domain from the given domain list.
    You will be provided a list of words formatted like this:
    words  = [list of words you can use]
    For each gate, the word you use as your final solution needs to be sampled from the list above. You can't use any word which is not part of the list as the solution of your rebus.
    Finally, the user will also provide you a difficulty level, which gives you an idea of how complicated they want gate to be i.e how hard is each clue to solve. 

    Here are a few examples:


    """
    required_files = ["low_reasoning.pkl", "medium_reasoning.pkl", "high_reasoning.pkl", "verifier_history.pkl", "verifier_outputs.pkl"]
    folder = Path(example_directory)
    if required_files:
        missing = [f for f in required_files if not ( folder / f).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing ICL files:" + str(required_files))

    # Loading all the saved results for quality generation
    current_results = glob.glob(example_directory + '/*.pkl')

    pkl_path_easy = current_results[3]
    pkl_path_medium = current_results[5]
    pkl_path_high = current_results[0]
    problems_path = current_results[1]

    problems = pd.read_pickle(problems_path)
    df_easy = pd.read_pickle(pkl_path_easy)
    df_medium = pd.read_pickle(pkl_path_medium)
    df_high = pd.read_pickle(pkl_path_high)

    simple_gates = list(set(df_easy) & set(df_medium) & set(df_high))
    medium_gates = list(set(df_medium) & set(df_high) - set(df_easy))
    difficult_gates = list(set(df_high) - set(df_easy) - set(df_medium))

    with open("domains.pkl", 'rb') as file:
        domains = pickle.load(file)

    easy = []
    medium = []
    difficult = []
    # labeling samples based on the difficulty control
    for gate in simple_gates:
        match = re.search(r'\d+', gate)
        random_domains = random.sample(domains, 20)        
        if match:
            number = int(match.group()) - 1
        problem = problems[number]
        if len(problem['solution']) <= 6:
            easy += [problem]
            new_verifier_system_prompt = new_verifier_system_prompt + " \n " + construct_user_model_interaction(problem, "easy", random_domains)
        else:
            medium += [problem]
            new_verifier_system_prompt = new_verifier_system_prompt + " \n " + construct_user_model_interaction(problem, "medium", random_domains)

    for gate in medium_gates:
        match = re.search(r'\d+', gate)
        random_domains = random.sample(domains, 20)        
        if match:
            number = int(match.group()) - 1
        problem = problems[number]
        if len(problem['solution']) <= 6:
            easy += [problem]
            new_verifier_system_prompt = new_verifier_system_prompt + " \n " + construct_user_model_interaction(problem, "easy", random_domains)            
        else:
            medium += [problem]
            new_verifier_system_prompt = new_verifier_system_prompt + " \n " + construct_user_model_interaction(problem, "medium", random_domains)            
    
    for gate in difficult_gates:
        match = re.search(r'\d+', gate)
        random_domains = random.sample(domains, 20)        
        if match:
            number = int(match.group()) - 1
        problem = problems[number]
        difficult += [problem]
        new_verifier_system_prompt = new_verifier_system_prompt + " \n " + construct_user_model_interaction(problem, "difficult", random_domains)    


    #return low_easy, high_easy, low_medium, high_medium, low_difficult, high_difficult
    return new_verifier_system_prompt


icl_verifier_system_prompt = """can you create a so called "reasoning rebus". Can you create a puzzle (or gate) that tests knowledge across all the disciplines to hide text. Furthermore, make it layered, that only by solving previous part (called gates) you will get to the next word or gate to solve, so it should be very iterative. You will first print entire gate and then print the solution like -

1. Gate = {The entire rebus}
Solution = {The solution to the above rebus}

Only provide solution once you are done printing the rebus and don't provide any clues.

While asking for the new gate the user will provide previous few gates and the difficulty request they gave the LLM while generating those gates. While requesting a new gate the user will specify whether you should increase, decrease or maintain the difficulty of the new gate compared to the previous gate. The user previous gate input will be structured as followes -

Previous Rebuses - 
Gate - {The generated gate}
Solution - {}
difficulty request - {}
.
.
.
Gate - {The generated gate}
solution - {}
difficulty request - {}.

new difficulty request - {}

If no previous gate or difficulty levels are specified then that means it is the first puzzle in that scenerio the user request will look like - 
Previous Rebuses - None 
Difficulty - Generate first gate

The user will also provide the previous gate number alongside a list of domains list and word list you can pick from to create the gate -

Previous Gate Number = {the nuumber of previous gate}.
Domains = {list of domains to choose from}
Words = {list pf words to choose from}

Your output should be structed such - 

{Previous Gate Number + 1}. Gate = {The entire rebus}
Solution = {The solution to the above rebus}

and don't mention the user difficulty instruction in your output.

Here are a few examples of generations -

USER INPUT - 

Previous Rebuses - None 
new difficulty request - Generate first gate

Previous Gate Number = 0
Domains = [Planetary Geology, Paleobotany, Cake Decorating, Packaging Engineering, Naturopathy]
Words = [enosis, zogan, music, toled]

OUTPUT - 

1. Gate = 
────────────────────────────────────────
Begin your journey by deciphering these interdisciplinary clues:
• Planetary Geology: Identify the red planet, renowned for its dusty, iron-oxide landscape and named after the Roman god of war. Extract its first letter. 
• Paleobotany: In the study of ancient plant life, uncover the name of an enigmatic seed fern mentioned in obscure Carboniferous texts as “Ulospora.” Take the initial letter of this genus.
• Cake Decorating: Visualize the fine art of cake adornment—imagine a delicate piping technique that creates a floral “swan” tip. Note the starting letter of this graceful design.
• Packaging Engineering: Consider a modern method of securing parcels; focus on an “impulse sealing” process used to tightly encapsulate items. Identify its leading letter.
• Naturopathy: Delve into natural healing traditions and find the remedy prized for its balancing aroma—a common ingredient in therapeutic practices known as camphor. Record the first letter of its name.
────────────────────────────────────────
Solution = music

USER INPUT - 

Previous Rebuses - 
1. Gate = 
────────────────────────────────────────
Begin your journey by deciphering these interdisciplinary clues:
• Planetary Geology: Identify the red planet, renowned for its dusty, iron-oxide landscape and named after the Roman god of war. Extract its first letter. 
• Paleobotany: In the study of ancient plant life, uncover the name of an enigmatic seed fern mentioned in obscure Carboniferous texts as “Ulospora.” Take the initial letter of this genus.
• Cake Decorating: Visualize the fine art of cake adornment—imagine a delicate piping technique that creates a floral “swan” tip. Note the starting letter of this graceful design.
• Packaging Engineering: Consider a modern method of securing parcels; focus on an “impulse sealing” process used to tightly encapsulate items. Identify its leading letter.
• Naturopathy: Delve into natural healing traditions and find the remedy prized for its balancing aroma—a common ingredient in therapeutic practices known as camphor. Record the first letter of its name.
────────────────────────────────────────
Solution = music
difficulty request - Generate first gate

new difficulty request - reduce

Previous Gate Number = 1
Domains = [American Literature, Video Games, Packaging Engineering, Naturopathy, Candle Making, Nuclear Physics]
Words  = [microzoal, certy, game, phenic]

OUTPUT - 

2. Gate = 
────────────────────────────────────────
Decipher the clues below by extracting the first letter of each answer to reveal the hidden word:
• American Literature: Identify the adjective that begins the title of F. Scott Fitzgerald’s celebrated work, “The ______ Gatsby.”
• Packaging Engineering: Recall the lightweight metal commonly used in forming thin foils for modern packaging applications. What is its name?
• Video Games: Name the renowned sandbox video game originally created by Markus Persson, celebrated for its blocky, pixelated world.
• Naturopathy: Determine the term describing the oils derived from plants that are prized for their therapeutic and aromatic qualities.
────────────────────────────────────────
Solution = game

USER INPUT - 

Previous Rebuses - 
1. Gate = 
────────────────────────────────────────
Begin your journey by deciphering these interdisciplinary clues:
• Planetary Geology: Identify the red planet, renowned for its dusty, iron-oxide landscape and named after the Roman god of war. Extract its first letter. 
• Paleobotany: In the study of ancient plant life, uncover the name of an enigmatic seed fern mentioned in obscure Carboniferous texts as “Ulospora.” Take the initial letter of this genus.
• Cake Decorating: Visualize the fine art of cake adornment—imagine a delicate piping technique that creates a floral “swan” tip. Note the starting letter of this graceful design.
• Packaging Engineering: Consider a modern method of securing parcels; focus on an “impulse sealing” process used to tightly encapsulate items. Identify its leading letter.
• Naturopathy: Delve into natural healing traditions and find the remedy prized for its balancing aroma—a common ingredient in therapeutic practices known as camphor. Record the first letter of its name.
────────────────────────────────────────
Solution = music
difficulty request - Generate first gate

2. Gate = 
────────────────────────────────────────
Decipher the clues below by extracting the first letter of each answer to reveal the hidden word:
• American Literature: Identify the adjective that begins the title of F. Scott Fitzgerald’s celebrated work, “The ______ Gatsby.”
• Packaging Engineering: Recall the lightweight metal commonly used in forming thin foils for modern packaging applications. What is its name?
• Video Games: Name the renowned sandbox video game originally created by Markus Persson, celebrated for its blocky, pixelated world.
• Naturopathy: Determine the term describing the oils derived from plants that are prized for their therapeutic and aromatic qualities.
────────────────────────────────────────
Solution = game
difficulty request - reduce

new difficulty request - It is way too simple and requiires a lot of work. Generate the next gate and increase the difficulty significantly and make it more ambiguous (so that only reasoning models can solve it).

Previous Gate Number - 2
Domains = [Soil Biology, Fashion Journalism, Packaging Engineering, Naturopathy, Geomatics, Video Games, Nuclear Physics, Computer Engineering, Candle Making, Healthcare Management, Soil Biology]
Words = [technical, sanitizes, whereness, olives, fretways]

OUTPUT -

3. Gate = 
────────────────────────────────────────
Unravel the clues below by determining the correct answer for each discipline and then extracting the specified letter to reveal the hidden word:
• Packaging Engineering: Identify the process that uses controlled heat to seal flexible packages. Record the first letter of its name.
• Naturopathy: Name the herb famed for its immune-boosting properties, commonly used in herbal remedies. Take its initial letter.
• Geomatics: In the realm of maps and spatial representation, determine the branch devoted to crafting accurate representations of the earth’s surface. Extract its leading letter.
• Video Games: Recognize the iconic first-person shooter franchise set on a ringworld that has captivated gamers worldwide. Note its starting letter.
• Nuclear Physics: Identify the neutral subatomic particle that is essential in nuclear reactions. Provide its first letter.
• Computer Engineering: Specify the term describing the method of processing multiple instructions in parallel by interleaving them. Record the first letter of this process.
• Candle Making: Determine the art of blending colors and fragrances into wax, resulting in a visually appealing product. Take its initial letter.
• Healthcare Management: In modern healthcare, identify the systematic approach used to evaluate and recognize quality hospitals, beginning with its first letter.
• Soil Biology: Within soil science, name the fine, organic-rich layer prized for its fertility. Provide its starting letter.
────────────────────────────────────────

Solution = technical

"""

prover_system_prompt = """You are a puzzle solving agent who will be given puzzles "gate" and you will solve the puzzle to the best of your abilities. While you can generate reasoning before final solution, put the solution for the provided challenge as python dictionary in the end structured like this {“Gate n”: solution string}. Where the key in the dictionary is the assigned number of the gate you are answering in that round of the interaction.

Here is an example - 

PROVIDED GATE - 

gate = 
────────────────────────────────────────
For each clue below, determine the answer and then record its first letter. In sequence, these letters will reveal a hidden word:
• Computer Science – Subfield 39: Identify the popular high-level programming language known for its snake logo. (Take the first letter of your answer.)
• Medicine – Subfield 43: Name the condition characterized by an abnormal accumulation of fluid in body tissues. (Record its starting letter.)
• Physics – Subfield 8: Identify the universal constant denoted by the letter "c" that represents the speed of light. (Take its letter as your answer.)
• Medicine – Subfield 15: What is the name of the joint connecting the thigh bone to the lower leg? (Extract the first letter.)
• Physics – Subfield 42: Determine the term for the rate of change of velocity. (Provide its initial letter.)
• Computer Science – Subfield 27: Identify the data structure that represents relationships using nodes (vertices) and connections (edges). (Record the first letter.)
• Mathematics – Subfield 17: What is the term for a statement that asserts the equality of two expressions? (Take its starting letter.)
────────────────────────────────────────

Solution = 

OUTPUT - 

To solve the puzzle for Gate 7, let's find the first letter of each answer based on the given clues:
1. **Computer Science – Subfield 39:** The popular high-level programming language known for its snake logo is **Python**. First letter: **P**
2. **Medicine – Subfield 43:** The condition characterized by an abnormal accumulation of fluid in body tissues is known as **Edema**. First letter: **E**
3. **Physics – Subfield 8:** The universal constant denoted by the letter "c" that represents the speed of light is **c** itself (commonly known as the speed of light in equations). Therefore, the letter is **C**
4. **Medicine – Subfield 15:** The joint connecting the thigh bone to the lower leg is the **Knee**. First letter: **K**
5. **Physics – Subfield 42:** The term for the rate of change of velocity is **Acceleration**. First letter: **A**
6. **Computer Science – Subfield 27:** The data structure that represents relationships using nodes (vertices) and connections (edges) is a **Graph**. First letter: **G**
7. **Mathematics – Subfield 17:** The term for a statement that asserts the equality of two expressions is an **Equation**. First letter: **E**
Combining these initial letters, we get the hidden word: **PECKAGE**
The solution: {"Gate": "PECKAGE"}





"""

icl_prover_system_prompt = """You puzzle solving agent who will be given puzzles "gate" and you will solve the puzzle to the best of your abilities. While you can generate reasoning before final solution, put the solution for the provided challenge as python dictionary in the end structured like this {“Gate n”: solution string}. Where the key in the dictionary is the assigned number of the gate you are answering in that round of the interaction.

Here is an example - 

PROVIDED GATE - 

3. Gate = 
────────────────────────────────────────
Unravel the clues below by determining the correct answer for each discipline and then extracting the specified letter to reveal the hidden word:
• Packaging Engineering: Identify the process that uses controlled heat to seal flexible packages. Record the first letter of its name.
• Naturopathy: Name the herb famed for its immune-boosting properties, commonly used in herbal remedies. Take its initial letter.
• Geomatics: In the realm of maps and spatial representation, determine the branch devoted to crafting accurate representations of the earth’s surface. Extract its leading letter.
• Video Games: Recognize the iconic first-person shooter franchise set on a ringworld that has captivated gamers worldwide. Note its starting letter.
• Nuclear Physics: Identify the neutral subatomic particle that is essential in nuclear reactions. Provide its first letter.
• Computer Engineering: Specify the term describing the method of processing multiple instructions in parallel by interleaving them. Record the first letter of this process.
• Candle Making: Determine the art of blending colors and fragrances into wax, resulting in a visually appealing product. Take its initial letter.
• Healthcare Management: In modern healthcare, identify the systematic approach used to evaluate and recognize quality hospitals, beginning with its first letter.
• Soil Biology: Within soil science, name the fine, organic-rich layer prized for its fertility. Provide its starting letter.
────────────────────────────────────────

Solution - 

OUTPUT - 

'To solve this puzzle, we need to determine the correct answers for each clue and extract the specified letter to form the hidden word.
1. **Packaging Engineering:** The process that uses controlled heat to seal flexible packages is called **Heat Sealing**. Take the first letter: **H**
2. **Naturopathy:** The herb famed for its immune-boosting properties is **Echinacea**. Take the first letter: **E**
3. **Geomatics:** The branch devoted to crafting accurate representations of the earth’s surface is **Cartography**. Take the first letter: **C**
4. **Video Games:** The iconic first-person shooter franchise set on a ringworld is **Halo**. Take the first letter: **H**
5. **Nuclear Physics:** The neutral subatomic particle essential in nuclear reactions is the **Neutron**. Take the first letter: **N**
6. **Computer Engineering:** The method of processing multiple instructions in parallel by interleaving them is called **Pipelining**. Take the first letter: **P**
7. **Candle Making:** The art of blending colors and fragrances into wax is referred to as **Tinting**. Take the first letter: **T**
8. **Healthcare Management:** The systematic approach to evaluate and recognize quality hospitals is **Accreditation**. Take the first letter: **A**
9. **Soil Biology:** The fine, organic-rich layer prized for its fertility is **Humus**. Take the first letter: **H**
Read the initial letters gathered: **H E C H N P T A H**
Combine them to form the hidden word: **HEALTHCAP**
The solution: {"Gate 3": "HEALTHCAP"}'

"""

generation_system_prompt_non_reasoning = """You are a text riddle generation agent. You will be given a letter and a domain and you must generate a difficult puzzle using that domain. Solution of the puzzle should a word that starts with the provided letter. Remeber the clue for the letter should be extremely difficult to solve and should require high amounts of reasoning even by strong reasoning models." \
Here are a few examples - 
user input =
domain =  psychiatry
letter = p
agent output = 
in the realm of mental health, what is the term for a substance given to a patient with no therapeutic effect, often used as a control in clinical trials? 

user input = 
domain = intellectual property law 
letter = o

agent output =
when protecting creative works, what word starting with o describes an original creation that is entirely new and not copied?

user input = 
domain = industrial-organizational psychology
letter = t

agent output = 
in the study of workplace behavior, what simple term beginning with t denotes a discrete piece of work or an assigned duty?

user input =
domain = parasitology
letter = i

agent output =
what term starting with i is used to describe the condition when a host is overwhelmed by parasites?

user input = 
domain = serious games
letter = o

agent output = 
in the context of games designed for education and training, what word beginning with o refers to a clear goal or mission that players must accomplish?

user input = 
domain - business analytics
letter = n

agent output = 
in analyzing financial performance, what term beginning with n represents the net profit remaining after all expenses are deducted?
"""

domain_generation_prompt = "We want to create reasoning puzzles that test knowledge across a wide range of domains. Can you give me a list of distinct and diverse 1000 domains. Output it as a python list with each domain being an element (string).  Just output the list and nothing else."

generator_difficulty_commands = {"First":"Generate first gate", "Increase":"Generate another gate and increase difficulty", 
                       "Increase 2":"Generate another gate and increase difficulty and make it ambiguous",
                         "Increase 3":"It is still way too simple and requiires a lot of work. Generate the next gate and increase the difficulty significantly and make it more ambiguous (so that only reasoning models can solve it).",
                         "Maintain":"Good maintain difficulty and ambiguity generate another gate", "Reduce":"generate new problems with reduced difficulty"}

