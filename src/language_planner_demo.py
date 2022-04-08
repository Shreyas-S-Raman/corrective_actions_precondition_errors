#!/usr/bin/env python
# coding: utf-8

# ## Language Models as Zero-Shot Planners:<br>Extracting Actionable Knowledge for Embodied Agents
# 
# This is the official demo code for our [Language Models as Zero-Shot Planners](https://huangwl18.github.io/language-planner/) paper. The code demonstrates how Large Language Models, such as GPT-3 and Codex, can generate action plans for complex human activities (e.g. "make breakfast"), even without any further training. The code can be used with any available language models from [OpenAI API](https://openai.com/api/) and [Huggingface Transformers](https://huggingface.co/docs/transformers/index) with a common interface.
# 
# **Note:**
# - It is observed that best results can be obtained with larger language models. If you cannot run [Huggingface Transformers](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) models locally or on Google Colab due to memory constraint, it is recommended to register an [OpenAI API](https://openai.com/api/) account and use GPT-3 or Codex (As of 01/2022, $18 free credits are awarded to new accounts and Codex series are free after [admitted from the waitlist](https://share.hsforms.com/1GzaACuXwSsmLKPfmphF_1w4sk30?)).
# - Due to language models' high sensitivity to sampling hyperparameters, you may need to tune sampling hyperparameters for different models to obtain the best results.
# - The code uses the list of available actions supported in [VirtualHome 1.0](https://github.com/xavierpuigf/virtualhome/tree/v1.0.0)'s [Evolving Graph Simulator](https://github.com/xavierpuigf/virtualhome/tree/v1.0.0/simulation). The available actions are stored in [`available_actions.json`](https://github.com/huangwl18/language-planner/blob/master/src/available_actions.json). The actions should support a large variety of household tasks. However, you may modify or replace this file if you're interested in a different set of actions or a different domain of tasks (beyond household domain).
# - A subset of the [manually-annotated examples](http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip) originally collected by the [VirtualHome paper](https://arxiv.org/pdf/1806.07011.pdf) is used as available examples in the prompt. They are transformed to natural language format and stored in [`available_examples.json`](https://github.com/huangwl18/language-planner/blob/master/src/available_examples.json). Feel free to change this file for a different set of available examples.

# ### Setup
# If you're on Colab, do the following:
# 1. run the following cell to setup repo and install dependencies.
# 2. enable GPU support (Runtime -> Change Runtime Type -> Hardware Accelerator -> set to GPU).

# In[ ]:


# !git clone https://github.com/huangwl18/language-planner
# %cd language-planner
# !pip install -r requirements.txt
# %cd src


# Import packages and specify which GPU to be used

# In[ ]:


import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
import pdb

pdb.set_trace()
GPU = 0
if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
OPENAI_KEY = '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'  # replace this with your OpenAI API key, if you choose to use OpenAI API


# ### Define hyperparemeters for plan generation
# 
# Select the source to be used (**OpenAI API** or **Huggingface Transformers**) and the two LMs to be used (**Planning LM** for plan generation, **Translation LM** for action/example matching). Then define generation hyperparameters.
# 
# Available language models for **Planning LM** can be found from:
# - [Huggingface Transformers](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads)
# - [OpenAI API](https://beta.openai.com/docs/engines) (you would need to paste your OpenAI API key from your account to `openai.api_key` below)
# 
# Available language models for **Translation LM** can be found from:
# - [Sentence Transformers](https://huggingface.co/sentence-transformers)

# In[ ]:


source = 'huggingface'  # select from ['openai', 'huggingface']
planning_lm_id = 'gpt2-large'  # see comments above for all options
translation_lm_id = 'stsb-roberta-large'  # see comments above for all options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.8  # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples
if source == 'openai':
    openai.api_key = OPENAI_KEY
    sampling_params =             {
                "max_tokens": 10,
                "temperature": 0.6,
                "top_p": 0.9,
                "n": 10,
                "logprobs": 1,
                "presence_penalty": 0.5,
                "frequency_penalty": 0.3,
                "stop": '\n'
            }
elif source == 'huggingface':
    sampling_params =             {
                "max_tokens": 10,
                "temperature": 0.1,
                "top_p": 0.9,
                "num_return_sequences": 10,
                "repetition_penalty": 1.2,
                'use_cache': True,
                'output_scores': True,
                'return_dict_in_generate': True,
                'do_sample': True,
            }


# ### Planning LM Initialization
# Initialize **Planning LM** from either **OpenAI API** or **Huggingface Transformers**. Abstract away the underlying source by creating a generator function with a common interface.

# In[ ]:


def lm_engine(source, planning_lm_id, device):
    if source == 'huggingface':
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(planning_lm_id)
        model = AutoModelForCausalLM.from_pretrained(planning_lm_id, pad_token_id=tokenizer.eos_token_id).to(device)

    def _generate(prompt, sampling_params):
        if source == 'openai':
            response = openai.Completion.create(engine=planning_lm_id, prompt=prompt, **sampling_params)
            generated_samples = [response['choices'][i]['text'] for i in range(sampling_params['n'])]
            # calculate mean log prob across tokens
            mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in range(sampling_params['n'])]
        elif source == 'huggingface':
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_len = input_ids.shape[-1]
            output_dict = model.generate(input_ids, max_length=prompt_len + sampling_params['max_tokens'], **sampling_params)
            # discard the prompt (only take the generated text)
            generated_samples = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
            # calculate per-token logprob
            vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)  # [n, length, vocab_size]
            token_log_probs = torch.gather(vocab_log_probs, 2, output_dict.sequences[:, prompt_len:, None]).squeeze(-1).tolist()  # [n, length]
            # truncate each sample if it contains '\n' (the current step is finished)
            # e.g. 'open fridge\n<|endoftext|>' -> 'open fridge'
            for i, sample in enumerate(generated_samples):
                stop_idx = sample.index('\n') if '\n' in sample else None
                generated_samples[i] = sample[:stop_idx]
                token_log_probs[i] = token_log_probs[i][:stop_idx]
            # calculate mean log prob across tokens
            mean_log_probs = [np.mean(token_log_probs[i]) for i in range(sampling_params['num_return_sequences'])]
        generated_samples = [sample.strip().lower() for sample in generated_samples]
        return generated_samples, mean_log_probs

    return _generate

generator = lm_engine(source, planning_lm_id, device)


# ### Translation LM Initialization
# Initialize **Translation LM** and create embeddings for all available actions (for action translation) and task names of all available examples (for finding relevant example)

# In[ ]:


# initialize Translation LM
translation_lm = SentenceTransformer(translation_lm_id).to(device)

# create action embeddings using Translated LM
with open('available_actions.json', 'r') as f:
    action_list = json.load(f)
action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

# create example task embeddings using Translated LM
with open('available_examples.json', 'r') as f:
    available_examples = json.load(f)
example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
example_task_embedding = translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

# helper function for finding similar sentence in a corpus given a query
def find_most_similar(query_str, corpus_embedding):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return most_similar_idx, matching_score


# ### Autoregressive Plan Generation
# Generate action plan autoregressively by mapping each action step to the closest available action.
# 
# The algorithm is summarized as follows:
# 1. Extract the most relevant human-annotated example by matching the task names (`Make breakfast` <-> `Make toast`)
# 2. Prompt the **Planning LM** with `[One Example Plan] + [Query Task] (+ [Previously Generated Actions])`
# 3. Sample **Planning LM** `n` times for single-step text output.
# 4. Rank the samples by `[Matching Score] + BETA * [Mean Log Prob]`, where
#     1. `[Matching Score]` is the cosine similarity to the closest allowed action as determined by **Translation LM**
#     2. `[Mean Log Prob]` is the mean log probabilities across generated tokens given by **Planning LM**
# 5. Append the closest allowed action of the highest-ranked sample to the prompt in Step 3
# 6. Return to Step 3 and repeat

# In[ ]:


# define query task
task = 'Make breakfast'
# find most relevant example
example_idx, _ = find_most_similar(task, example_task_embedding)
example = available_examples[example_idx]
# construct initial prompt
curr_prompt = f'{example}\n\nTask: {task}'
# print example and query task
print('-'*10 + ' GIVEN EXAMPLE ' + '-'*10)
print(example)
print('-'*10 + ' EXAMPLE END ' + '-'*10)
print(f'\nTask: {task}')
for step in range(1, MAX_STEPS + 1):
    best_overall_score = -np.inf
    # query Planning LM for single-step action candidates
    samples, log_probs = generator(curr_prompt + f'\nStep {step}:', sampling_params)
    for sample, log_prob in zip(samples, log_probs):
        most_similar_idx, matching_score = find_most_similar(sample, action_list_embedding)
        # rank each sample by its similarity score and likelihood score
        overall_score = matching_score + BETA * log_prob
        translated_action = action_list[most_similar_idx]
        # heuristic for penalizing generating the same action as the last action
        if step > 1 and translated_action == previous_action:
            overall_score -= 0.5
        # find the translated action with highest overall score
        if overall_score > best_overall_score:
            best_overall_score = overall_score
            best_action = translated_action

    # terminate early when either the following is true:
    # 1. top P*100% of samples are all 0-length (ranked by log prob)
    # 2. overall score is below CUTOFF_THRESHOLD
    # else: autoregressive generation based on previously translated action
    top_samples_ids = np.argsort(log_probs)[-int(P * len(samples)):]
    are_zero_length = all([len(samples[i]) == 0 for i in top_samples_ids])
    below_threshold = best_overall_score < CUTOFF_THRESHOLD
    if are_zero_length:
        print(f'\n[Terminating early because top {P*100}% of samples are all 0-length]')
        break
    elif below_threshold:
        print(f'\n[Terminating early because best overall score is lower than CUTOFF_THRESHOLD ({best_overall_score} < {CUTOFF_THRESHOLD})]')
        break
    else:
        previous_action = best_action
        formatted_action = (best_action[0].upper() + best_action[1:]).replace('_', ' ') # 'open_fridge' -> 'Open fridge'
        curr_prompt += f'\nStep {step}: {formatted_action}'
        print(f'Step {step}: {formatted_action}')


# In[ ]:




