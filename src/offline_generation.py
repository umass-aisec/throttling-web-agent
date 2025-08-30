import pandas as pd
import random
from pathlib import Path
import prompts
import utils 

def generate_threshold(thresholds):
    if isinstance(thresholds, tuple) and len(thresholds) == 2 and thresholds[0] < thresholds[1]:
        easy_threshold = thresholds[0]
        medium_threshold = thresholds
        hard_threshold = thresholds[1]
    elif isinstance(thresholds, int):
        valid_threshold = thresholds
        easy_threshold = thresholds
        medium_threshold = thresholds
        hard_threshold = thresholds + 1        
    else:
        raise TypeError("Not a valid threshold. Must be an integer or a tuple (a, b) where b > a")
    return easy_threshold, medium_threshold, hard_threshold

def valid_challenge_len(difficulty, generator_solution, threshold):
    if difficulty == "easy":
        if len(generator_solution) <= threshold:
            return True
        else:
            return False
    if difficulty == "medium":
        if isinstance(threshold, tuple) and len(generator_solution) >= threshold[0] and len(generator_solution) <= threshold[1]:
            return True
        elif isinstance(threshold, int ) and len(generator_solution) <= threshold:
            return True
        else:
            return False
    if difficulty == "difficult":
        if len(generator_solution) >= threshold:
            return True
        else:
            return False
    return None

def gen_sample(difficulty, bank, domains, threshold, system_prompt, model, model_provider, thinking_budget):
    random_elements = random.sample(bank, 50)
    random_domains = random.sample(domains, 20)
    user_input = f"""Domain: {random_domains}
    words: {random_elements}
    difficulty: {difficulty}
    """
    length = False
    while not length:
        if model_provider == "Google":
          generator_response = utils.gemini_run_command(user_input, system_prompt, model, thinking_budget)
        else:  
            generator_response = utils.run_command(user_input, system_prompt, model)
        generator_response['text'] = generator_response['text'].lower()
        #print(generator_response['text'])
        generator_solution = utils.extract_solution(generator_response['text'])
        generator_question = generator_response['text'].replace(generator_solution, ".").lower()
        if valid_challenge_len(difficulty, generator_solution, threshold):
            length = True
    generator_output = {
        "generated_question": generator_question,
        "generated_solution": generator_solution,
        "generated_response": generator_response
    }
    return generator_output

    
def generate_bank(domains, words, system_prompt, thresholds, num_samples, model="o3-mini", model_provider="OpenAI", thinking_budget=None):

    easy_threshold, medium_threshold, difficult_threshold = generate_threshold(thresholds)

    threshold_bank = utils.split_by_value(words, thresholds)
    
    if isinstance(thresholds, tuple):
        easy_bank = threshold_bank[0]
        medium_bank = threshold_bank[1]
        difficult_bank = threshold_bank[2]
    else:
        easy_bank = threshold_bank[0]
        medium_bank = threshold_bank[0]
        difficult_bank = threshold_bank[1]


    problems = []
    solutions = []
    labels = []

    for i in range(num_samples):
        
        easy_problem = gen_sample("easy", easy_bank, domains, easy_threshold, system_prompt, model, model_provider, thinking_budget)
        problems += [easy_problem['generated_question']]
        solutions += [easy_problem['generated_solution']]
        labels += ["easy"]

    for i in range(num_samples):

        medium_problem = gen_sample("medium", medium_bank, domains, medium_threshold, system_prompt, model, model_provider, thinking_budget)  
        problems += [medium_problem['generated_question']]
        solutions += [medium_problem['generated_solution']]
        labels += ["medium"]

    for i in range(num_samples):

        difficult_problem = gen_sample("difficult", difficult_bank, domains, difficult_threshold, system_prompt, model, model_provider, thinking_budget)  
        problems += [difficult_problem['generated_question']]
        solutions += [difficult_problem['generated_solution']]
        labels += ["difficult"]
    
    data = {'Problems':problems, 'Solutions':solutions, 'Difficulty Level':labels}
    challenge_dataframe = pd.DataFrame(data)   

    return challenge_dataframe 

def generate_problem_non_reasoning(word, domains, model):
    generated_problem = "gate =\n────────────────────────────────────────\nembark on this quest by deciphering the interdisciplinary clues below. for each prompt, determine the correct answer and record its first letter. when arranged sequentially, these letters will reveal a concealed english word.\n\n"
    for letter in word:
        sampled_domain = random.sample(domains, 1)[0]
        user_input = "domain = " + sampled_domain + "\n letter = " + letter
        generated_riddle = utils.run_command(user_input, prompts.generation_system_prompt_non_reasoning, model)['text']
        generated_problem += "\n\n• " + sampled_domain + " - " + generated_riddle + " (record its first letter.)\n"

    generated_problem += "\n\nwhen the letters extracted from each correct answer are arranged in sequence, they will unveil the hidden word.\n\n────────────────────────────────────────\n\nsolution: ."
    
    return generated_problem

def generate_bank_non_reasoning(words, domains, num_samples):
    problems = []
    solutions = []
    for i in range(num_samples):
        word = random.sample(words, 1)[0]
        problems += [generate_problem_non_reasoning(word, domains, "gpt-4o")]
        solutions += [word]
    bank_df = pd.DataFrame({"Problems":problems, "Solutions":solutions})
    return bank_df

