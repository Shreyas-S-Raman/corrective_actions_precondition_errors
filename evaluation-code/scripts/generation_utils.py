import json
import copy
import os
from enum import Enum
import os
import openai
from sentence_transformers import util as st_utils
import sys
sys.path.append('../dataset_utils')
from add_preconds import *
import numpy as np
from vh_configs import *
from collections import OrderedDict
import parse
import torch
import time
from scene_gym import SceneGym
from prompt_generator import PromptGenerator
import pdb

np.random.seed(123)

API_KEYS = ['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']
#API_KEYS = ['ENTER API KEYS']

init_key_idx = np.random.randint(0, len(API_KEYS))
print(f'using key {init_key_idx}')
openai.api_key = API_KEYS[init_key_idx]

def normalize(v, min_v, max_v):
    return (v - min_v) / (max_v - min_v)

def swap_key():
    curr_idx = API_KEYS.index(str(openai.api_key))
    openai.api_key = API_KEYS[(curr_idx + 1) % len(API_KEYS)]

def save_txt(save_path, txt):
    if save_path[-4:] != '.txt':
        save_path += '.txt'
    with open(save_path, 'w') as f:
        f.write(txt)
    return save_path

def load_txt(load_path):
    if load_path[-4:] != '.txt':
        load_path += '.txt'
    with open(load_path, 'r') as f:
        return f.read()

def save_dict(save_path, _dict):
    if save_path[-5:] != '.json':
        save_path += '.json'
    with open(save_path, 'w') as f:
        json.dump(_dict, f)
    return save_path

def load_dict(load_path):
    if load_path[-5:] != '.json':
        load_path += '.json'
    with open(load_path, 'r') as f:
        return json.load(f)


def parseStrBlock(block_str):
    """ Given a str block [Rinse] <CLEANING SOLUTION> (1)
        parses the block, returning Action, List Obj, List Instance
    """
    action = block_str[1:block_str.find(']')]
    block_str = block_str[block_str.find(']')+3:-1]
    block_split = block_str.split(') <') # each element is name_obj> (num
    obj_names = [block[0:block.find('>')] for block in block_split]
    inst_nums = [block[block.find('(')+1:] for block in block_split]
    action = action.strip()
    obj_names_corr = []
    inst_nums_corr = []
    for i in range(len(obj_names)):
        if len(obj_names[i].strip()) > 0 and len(inst_nums[i].strip()) > 0:
            obj_names_corr.append(obj_names[i])
            inst_nums_corr.append(inst_nums[i])
    return action, obj_names_corr, inst_nums_corr

def process_example_arg(arg):
    # don't use any underscore in args
    arg = arg.replace('_', ' ')
    if 'coffee' not in arg and 'coffe' in arg:
        arg = arg.replace('coffe', 'coffee')
    return arg

def program_lines2program_english(program_lines):
    program_english = ''
    for l, line in enumerate(program_lines):
        script_action, script_args, _ = parseStrBlock(line)
        # don't use any underscore in args
        script_args = [process_example_arg(arg) for arg in script_args]
        action_num_args = None
        action_template = None
        for _action in EvolveGraphAction:
            if _action.name.upper() == script_action.upper():
                action_num_args = _action.value[1]
                action_template = _action.value[2]
                break
        assert action_num_args is not None and action_num_args == len(script_args)
        action_str = action_template.format(*script_args)
        # make the first letter capitalized
        action_str = action_str[0].upper() + action_str[1:]
        program_english += 'Step {}: {}\n'.format(l + 1, action_str)
    return program_english.strip()

def construct_example(example_path, add_desc=False):
    example = load_txt(example_path)
    full_program_lines = example.strip().split('\n')
    title = full_program_lines[0]
    program_lines = [line for line in full_program_lines if '[' in line]
    program_english = program_lines2program_english(program_lines)
    if add_desc:
        description = full_program_lines[1]
        return 'Task: {}\nDescription: {}\n{}\n\n'.format(title, description, program_english)
    else:
        return 'Task: {}\n{}\n\n'.format(title, program_english)

def select_most_similar_example_idx(sentence_model, query_title, title_embedding, device):
    """get the path to the most similar example from vh dataset"""
    if ':' in query_title:
        query_title = query_title[query_title.index(':') + 1:].strip()
    most_similar_idx, _ = top_k_similar(sentence_model, query_title, title_embedding, device, top_k=1)
    most_similar_idx = most_similar_idx[0]
    return most_similar_idx

def top_k_similar(model, query_str, corpus_embedding, device, top_k=1):
    """
    translate orignal_action to the closest action in action_list using semantic similarity
    adapted from: https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6
    """
    # encode sentence to get sentence embeddings
    query_embedding = model.encode(query_str, convert_to_tensor=True, device=device)
    # compute similarity scores of the sentence with the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0]
    cos_scores = cos_scores.detach().cpu().numpy()
    # Sort the results in decreasing order and get the first top_k
    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
    return top_results[:top_k], cos_scores[top_results[:top_k]]

def api_retry_if_failed(params, engine='davinci-codex', max_iters=1000):
    curr_iter = 0
    response = None
    while curr_iter < max_iters:
        try:
            # different usage for zero-shot api and finetuned api
            if ':ft-' in engine:
                response = openai.Completion.create(model=engine, **params)
            else:
                response = openai.Completion.create(engine=engine, **params)
            break
        except (openai.error.APIError, openai.error.RateLimitError, openai.error.APIConnectionError) as err:
            curr_iter += 1
            print(f'*** [{curr_iter}/{max_iters}] API returns {err.__class__.__name__}: {err}')
            if 'RateLimitError' == str(err.__class__.__name__) or 'Rate limit reached for tokens' in str(err):
                swap_key()
                sleep_time = np.random.randint(low=10, high=30)
            else:
                sleep_time = 1
            print(f'*** sleep for {sleep_time} second and retrying...')
            time.sleep(sleep_time)
            continue

    return response

def one_shot_api_request(example, task_prompt, api_params, sentence_model, action_list_embedding, device, action_list, max_iters=1000, beta=0.5, raw_lm=False, engine='davinci-codex', step_by_step = False):

    def _get_score(matching_score, log_prob):
        return matching_score + beta * log_prob

    def _format_api_output(output):
        # trim generated output
        if 'Task:' in output[10:]:
            output = output[:10 + output[10:].index('Task:')]
        # ignore everything after """
        if '"""' in output:
            output = output[:output.index('"""')]
        if "'''" in output:
            output = output[:output.index("'''")]
        return output.strip().replace('\n\n', '\n')

    default_params = copy.deepcopy(api_params)
    default_params['prompt'] = example + task_prompt + '\nStep 1:'

    default_params['max_tokens'] = max(default_params['max_tokens'], 100)
    if isinstance(engine, str):
        if 'codex' in engine:
            default_params['stop'] = '"""'
        else:
            default_params['stop'] = '\n\n'
    else:
        default_params['stop'] = 'Task:'
    if isinstance(engine, str):
        response = api_retry_if_failed(default_params, engine=engine, max_iters=max_iters)
    else:
        response = engine(default_params)
    best_score = -float('inf')
    best_generated = None
    best_translated_actions = None
    for i in range(default_params['n']):
        generated_text = response['choices'][i]['text']
        generated_text = _format_api_output(task_prompt + '\nStep 1:' + generated_text)
        logprob = np.mean(response['choices'][i]['logprobs']['token_logprobs'])
        if raw_lm:
            _, parse_info = str2program_list(generated_text.split('\n')[1:])
            curr_score = _get_score(0, logprob)
            all_translated_actions = None
        else:
            program_lines = generated_text[generated_text.index('Step 1:'):].split('\n')
            all_translated_actions = []
            program_matching_score = 0
            for line in program_lines:
                try:
                    processed = line[line.index(':') + 1:].strip().lower()
                except ValueError:
                    processed = line.strip().lower()
                most_similar_idx, matching_score = top_k_similar(sentence_model, processed, action_list_embedding, device, top_k=1)
                most_similar_idx, matching_score = most_similar_idx[0], matching_score[0]
                program_matching_score += matching_score
                translated_action = action_list[most_similar_idx]
                all_translated_actions.append(translated_action)
            curr_score = _get_score(np.mean(program_matching_score), logprob)
        if curr_score > best_score:
            best_score = curr_score
            best_generated = generated_text
            best_translated_actions = all_translated_actions
    return best_generated, best_translated_actions, None

def iterative_api_request(example, task_prompt, api_params, sentence_model, action_list_embedding, device, action_list, max_iters=1000, max_steps=20, verbose=False, cutoff_threshold=-100, beta=0.5, percent_terminate=0.6, engine='davinci-codex', translated_condition=False, step_by_step = False):

    def _get_score(matching_score, log_prob):
        return matching_score + beta * log_prob

    def _format_api_output(output):
        # exclude examples
        if '\n\n' in output:
            output = output[output.index('\n\n') + 2:]
        return output.strip()

    default_params = copy.deepcopy(api_params)
    # stop when seeing a new line since we are generating one action per iter
    default_params['stop'] = '\n'
    full_text = example + task_prompt + '\nStep 1:' if not step_by_step else example + task_prompt + '\nLet\'s think step by step.' + '\nStep 1:'
    #pdb.set_trace()
    all_translated_actions = []
    curr_step = 0
    while curr_step < max_steps:

        '''tracks all options for generated text + translated actions for the current step'''
        curr_generated = []
        curr_matching = []
        curr_logprobs = []
        curr_translated = []
        curr_overall = []

        # query api ===================================
        #add full text (prompt and query task) as 'prompt' for LLM prediction
        default_params['prompt'] = full_text
        if isinstance(engine, str):
            response = api_retry_if_failed(default_params, max_iters=max_iters, engine=engine)
        else:
            response = engine(default_params)

        '''response format: {'choices': [{'text': '<s><s><s>.....', 'logprobs': {'token_logprobs': array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}}]}

            iterates all responses + chooses best for next step?
        '''
        for i in range(default_params['n']):
            generated_text = response['choices'][i]['text']
            logprob = np.mean(response['choices'][i]['logprobs']['token_logprobs'])

            # calculate score for current step
            '''removes white space from generated text and convert to lower case
               assumes that after curr_step 0 ==> model will auto generate 'Step 2:', 'Step 3:' prefixes to generated_text
            '''
            if curr_step == 0:
                processed = generated_text.strip().lower()
            else:
                try:
                    processed = generated_text[generated_text.index(':') + 1:].strip().lower()
                except ValueError as e:
                    curr_generated.append('PARSING ERROR')
                    curr_matching.append(-200)
                    curr_logprobs.append(-200)
                    curr_translated.append('PARSING ERROR')
                    curr_overall.append(-200)
                    continue
            most_similar_idx, matching_score = top_k_similar(sentence_model, processed, action_list_embedding, device, top_k=1)
            most_similar_idx, matching_score = most_similar_idx[0], matching_score[0]
            overall_score = _get_score(matching_score, logprob)
            '''matching_score + beta * log_prob'''

            '''indexes action list with most similar index for action'''
            translated_action = action_list[most_similar_idx]

            if verbose:
                print(f'** {generated_text} ({translated_action}; matching_score={matching_score:.2f}; mean_logprob={logprob:.2f}); overall={overall_score:.2f}')

            # record metrics for each output
            '''generated_text contains raw LLM output
               translated_action contains matched/converted action from action_list (list of defined actions in VH)
               curr_overal: tracks overall score/performance of plan
            '''
            curr_matching.append(matching_score)
            curr_translated.append(translated_action)
            curr_logprobs.append(logprob)
            curr_generated.append(generated_text)
            # penalize seen actions
            if translated_action in all_translated_actions:
                if verbose:
                    print('=' * 40 + f'\n== {translated_action} has been seen, assigning score 0...\n' + '=' * 40)
                curr_overall.append(-100)
            else:
                curr_overall.append(overall_score)

        # stop when model thinks it's finished or format is wrong (very unlikely in practice)
        num_to_look_at = int(percent_terminate * default_params['n'])
        '''sort the log probabilities of choices in increasing order + pick most 'k' likely as final choice for next step'''
        highest_ids = np.argsort(curr_logprobs)[-num_to_look_at:]
        terminate = True
        for idx in highest_ids:
            if len(curr_generated[idx]) > 0 and (curr_step == 0 or curr_generated[idx][:4] == 'Step'):
                terminate = False
        if terminate:
            if verbose:
                print(f'** model thinks it should terminate {generated_text}')
            break

        # calculate most likely step ===================================
        '''compare best score with cutoff threshold: cutoff threshold not implemented by default
           best_idx: proposed action/step with best overall score
        '''
        highest_score = np.max(curr_overall)
        best_idx = np.argsort(curr_overall)[-1]
        if cutoff_threshold != -100 and highest_score < cutoff_threshold:
            if verbose:
                print(f'## STOP GENERATION because best score after {default_params["n"]} attempts was {curr_generated[best_idx]} ({highest_score} < {cutoff_threshold})')
            break
        # select the previously generated output whose score is the highest
        '''uses args.translated_condition option: takes best from translated actions (rather than generated text) '''
        if translated_condition:
            best_curr = curr_translated[best_idx]
            best_curr = best_curr[0].upper() + best_curr[1:]
            best_curr = best_curr.replace('_', ' ')
            if curr_step == 0:
                best_curr = f' {best_curr}'
            else:
                best_curr = f'Step {curr_step + 1}: {best_curr}'
        else:
            best_curr = curr_generated[best_idx]
        if verbose:
            print(f'## selecting best-score output "{best_curr}" (score: {highest_score}; raw: {curr_generated[best_idx]}; translated: {curr_translated[best_idx]})\n')

        # accumulate output and continue
        '''appends best_curr (either generated text or translated actions) to full_text
           appends the best/chosen translated action (with highest overall score) to all_translated_actions
        '''
        full_text += f'{best_curr}\n'
        all_translated_actions.append(curr_translated[best_idx])
        curr_step += 1

    '''_format_api_output() reindexes and removes spacing in full_text output
        all_translated_actions is list of all translated actions
    '''
    return _format_api_output(full_text.strip()), all_translated_actions, curr_step

'''
Note: get good video output from VH for intermediate steps and final best plan
'''
def online_api_request(example, task_prompt, api_params, sentence_model, action_list_embedding, device, action_list, raw_lm, scene_path, scene_num, prompt_args, max_iters=1000, max_steps=20, verbose=False, cutoff_threshold=-100, beta=0.5, percent_terminate=0.6, engine='davinci-codex', translated_condition=False, step_by_step = False ):

    def _get_score(matching_score, log_prob):
        return matching_score + beta * log_prob

    def _format_api_output(output):
        # exclude examples
        if '\n\n' in output:
            output = output[output.index('\n\n') + 2:]
        return output.strip()

    def _generate_action(full_text, default_params):
        '''tracks all options for generated text + translated actions for the current step'''
        curr_generated = []
        curr_matching = []
        curr_logprobs = []
        curr_translated = []
        curr_overall = []

        # query api ===================================
        #add full text (prompt and query task) as 'prompt' for LLM prediction
        default_params['prompt'] = full_text
        if isinstance(engine, str):
            response = api_retry_if_failed(default_params, max_iters=max_iters, engine=engine)
        else:
            response = engine(default_params)

        '''response format: {'choices': [{'text': '<s><s><s>.....', 'logprobs': {'token_logprobs': array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}}]}

            iterates all responses + chooses best for next step?
        '''
        for i in range(default_params['n']):
            generated_text = response['choices'][i]['text']
            logprob = np.mean(response['choices'][i]['logprobs']['token_logprobs'])

            # calculate score for current step
            '''removes white space from generated text and convert to lower case
               assumes that after curr_step 0 ==> model will auto generate 'Step 2:', 'Step 3:' prefixes to generated_text
            '''
            if curr_step == 0:
                processed = generated_text.strip().lower()
            else:
                try:
                    processed = generated_text[generated_text.index(':') + 1:].strip().lower()
                except ValueError as e:
                    curr_generated.append('PARSING ERROR')
                    curr_matching.append(-200)
                    curr_logprobs.append(-200)
                    curr_translated.append('PARSING ERROR')
                    curr_overall.append(-200)
                    continue
            most_similar_idx, matching_score = top_k_similar(sentence_model, processed, action_list_embedding, device, top_k=1)
            most_similar_idx, matching_score = most_similar_idx[0], matching_score[0]
            overall_score = _get_score(matching_score, logprob)
            '''matching_score + beta * log_prob'''

            '''indexes action list with most similar index for action'''
            translated_action = action_list[most_similar_idx]

            if verbose:
                print(f'** {generated_text} ({translated_action}; matching_score={matching_score:.2f}; mean_logprob={logprob:.2f}); overall={overall_score:.2f}')

            # record metrics for each output
            '''generated_text contains raw LLM output
               translated_action contains matched/converted action from action_list (list of defined actions in VH)
               curr_overal: tracks overall score/performance of plan
            '''
            curr_matching.append(matching_score)
            curr_translated.append(translated_action)
            curr_logprobs.append(logprob)
            curr_generated.append(generated_text)
            # penalize seen actions
            if translated_action in all_translated_actions:
                if verbose:
                    print('=' * 40 + f'\n== {translated_action} has been seen, assigning score 0...\n' + '=' * 40)
                curr_overall.append(-100)
            else:
                curr_overall.append(overall_score)

        # stop when model thinks it's finished or format is wrong (very unlikely in practice)
        num_to_look_at = int(percent_terminate * default_params['n'])
        '''sort the log probabilities of choices in increasing order + pick most 'k' likely as final choice for next step'''
        highest_ids = np.argsort(curr_logprobs)[-num_to_look_at:]

        nogen_terminate = True; score_terminate = False; error_message = None

        for idx in highest_ids:
            if len(curr_generated[idx]) > 0 and (curr_step == 0 or curr_generated[idx][:4] == 'Step'):
                nogen_terminate = False

        if nogen_terminate:
            if verbose:
                print(f'** model thinks it should terminate {generated_text}')

            error_message = f'No plan generated: model thinks it should terminate'

            return None, None, nogen_terminate, score_terminate, error_message

        # calculate most likely step ===================================
        '''compare best score with cutoff threshold: cutoff threshold not implemented by default
           best_idx: proposed action/step with best overall score
        '''
        highest_score = np.max(curr_overall)
        best_idx = np.argsort(curr_overall)[-1]
        if cutoff_threshold != -100 and highest_score < cutoff_threshold:
            score_terminate = True
            if verbose:
                print(f'## STOP GENERATION because best score after {default_params["n"]} attempts was {curr_generated[best_idx]} ({highest_score} < {cutoff_threshold})')

            error_message = f'STOP GENERATION because best score after {default_params["n"]} attempts was {curr_generated[best_idx]} ({highest_score} < {cutoff_threshold})'

            return None, curr_translated[best_idx], nogen_terminate, score_terminate, error_message

        # select the previously generated output whose score is the highest
        '''uses args.translated_condition option: takes best from translated actions (rather than generated text) '''
        if translated_condition:
            best_curr = curr_translated[best_idx]
            best_curr = best_curr[0].upper() + best_curr[1:]
            best_curr = best_curr.replace('_', ' ')
            if curr_step == 0:
                best_curr = f' {best_curr}'
            else:
                best_curr = f'Step {curr_step + 1}: {best_curr}'
        else:
            best_curr = curr_generated[best_idx]
        if verbose:
            print(f'## selecting best-score output "{best_curr}" (score: {highest_score}; raw: {curr_generated[best_idx]}; translated: {curr_translated[best_idx]})\n')

        return best_curr, curr_translated[best_idx], nogen_terminate, score_terminate, error_message



    scene_environment = SceneGym(scene_path, scene_num, task_prompt)
    prompt_generator = PromptGenerator(prompt_args)

    default_params = copy.deepcopy(api_params)
    # stop when seeing a new line since we are generating one action per iter
    default_params['stop'] = '\n'

    full_text = example + task_prompt  if not step_by_step else example + task_prompt + '\nLet\'s think step by step.'
    final_text = example + task_prompt

    all_translated_actions = []
    final_translated_actions = []

    all_errors = []

    curr_step = 0; total_steps = 0



    #track errors until escape step

    while curr_step < max_steps and total_steps < max_steps*2:
        #pdb.set_trace()
        no_gen_error = None; score_error = None; parsing_error = None; empty_program_error = None; precond_error = None; check_script_error = None
        executed = True

        # accumulate output and continue
        best_curr, translated_action, nogen_terminate, score_terminate, error_message = _generate_action(full_text, default_params)


        #failure check 1: no_gen_terminate
        if nogen_terminate:
            executed = True
            no_gen_error = error_message
            break

        #failure check 2: score terminate
        if score_terminate:
            executed = True
            score_error = error_message
            break


        #add best step to plan + continue
        full_text += f'\nStep 1: {best_curr}' if curr_step==0 else f'{best_curr}\n'
        all_translated_actions.append(translated_action)
        total_steps +=1


        #check for execution/precondition errors
        formatted_text = _format_api_output(full_text.strip())

        if raw_lm:
            program_lines, parse_info = str2program_list(full_text.split('\n')[1:])
            program_text = '\n'.join(program_lines).strip()

        else:
            matched_program_text = '\n'.join(all_translated_actions).strip()
            program_lines, parse_info = str2program_list(all_translated_actions)
            program_lines = remove_same_consecutive(program_lines)
            program_text = '\n'.join(program_lines).strip()

        parsed_program_lines = arg2abstract(program_lines)

        #failure check 3: parsing error
        if parse_info['parsibility']==0:
            executed = False
            parsing_error = parse_info['parsing_error']

            all_errors.append(parsing_error)

            error_prompt = prompt_generator.generate_prompt('parsibility', parsing_error, total_steps, best_curr, translated_action, parsed_program_lines[-1])

            full_text += '{}\n'.format(error_prompt)
            continue


        #failure check 4: empty program error
        if len(parsed_program_lines) == 0:
            executed = False
            empty_program_error = 'Script Fail: empty program'

            all_errors.append(empty_program_error)

            error_prompt = prompt_generator.generate_prompt('empty_program', empty_program_error, total_steps, best_curr, translated_action, parsed_program_lines[-1])

            full_text += '{}\n'.format(error_prompt)
            continue



        #failure check 5: precondition error on the last action taken
        try:

            preconditions = get_preconds_script([parsed_program_lines[-1]], verbose=verbose).printCondsJSON()
        except ScriptFail as e:
            executed = False
            precond_error = 'ScriptFail: {}'.format(e.message)

            all_errors.append(precond_error)

            error_prompt = prompt_generator.generate_prompt('precond', precond_error, total_steps, best_curr, translated_action, parsed_program_lines[-1])

            full_text += '{}\n'.format(error_prompt)
            continue


        #take a single step/action in the VH scene
        try:
            message, message_params, graph_dict, ____, prev_graph_dict, modified_script = scene_environment.step([parsed_program_lines[-1]], preconditions)

        except Exception as e:
            message = "{}: {}".format(e.__class__.__name__, e)

        #failure check 6: executability error
        if not 'is executable' in message:
            executed = False
            check_script_error = message

            all_errors.append(check_script_error)

            error_prompt = prompt_generator.generate_prompt('check_script', check_script_error, total_steps, best_curr, translated_action, message_params)

            full_text += '{}\n'.format(error_prompt)
            continue


        #if all failure checks pass: then increment step
        curr_step += 1
        #add best step to plan + continue
        final_text += f'\n{best_curr}' if curr_step > 1 else f'\nStep 1:{best_curr}'
        final_translated_actions.append(translated_action)

    info = { 'parsed_program': '\n'.join(program_lines).strip(), 'executed': executed, 'scene_path': scene_path,
    'init_graph_dict': scene_environment.initial_graph_dict, 'modified_program': modified_script.to_string(),
    'execution_error': check_script_error, 'precond_error': precond_error, 'parsing_error':parsing_error,
    'empty_program_error':empty_program_error, 'total_steps': total_steps, 'final_steps': curr_step, 'no_gen_error':no_gen_error, 'score_error':score_error,  'all_errors': '\n'.join(all_errors)}


    return _format_api_output(final_text.strip()), final_translated_actions, _format_api_output(full_text.strip()), all_translated_actions, info

def online_api_request_one_error(example, task_prompt, api_params, sentence_model, action_list_embedding, device, action_list, raw_lm, scene_path, scene_num, prompt_args, max_iters=1000, max_steps=20, verbose=False, cutoff_threshold=-100, beta=0.5, percent_terminate=0.6, engine='davinci-codex', translated_condition=False, step_by_step = False ):

    def _get_score(matching_score, log_prob):
        return matching_score + beta * log_prob

    def _format_api_output(output):
        # exclude examples
        if '\n\n' in output:
            output = output[output.index('\n\n') + 2:]
        return output.strip()
    

    def _generate_reasons(full_text, default_params):
        
        pass

    def _generate_action(full_text, default_params):
        '''tracks all options for generated text + translated actions for the current step'''
        curr_generated = []
        curr_matching = []
        curr_logprobs = []
        curr_translated = []
        curr_overall = []

        # query api ===================================
        #add full text (prompt and query task) as 'prompt' for LLM prediction
        default_params['prompt'] = full_text
        if isinstance(engine, str):
            response = api_retry_if_failed(default_params, max_iters=max_iters, engine=engine)
        else:
            response = engine(default_params)

        '''response format: {'choices': [{'text': '<s><s><s>.....', 'logprobs': {'token_logprobs': array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}}]}

            iterates all responses + chooses best for next step?
        '''
        for i in range(default_params['n']):
            generated_text = response['choices'][i]['text']
            logprob = np.mean(response['choices'][i]['logprobs']['token_logprobs'])

            # calculate score for current step
            '''removes white space from generated text and convert to lower case
               assumes that after curr_step 0 ==> model will auto generate 'Step 2:', 'Step 3:' prefixes to generated_text
            '''
            if curr_step == 0:
                processed = generated_text.strip().lower()
            else:
                try:
                    processed = generated_text[generated_text.index(':') + 1:].strip().lower()
                except ValueError as e:
                    curr_generated.append('PARSING ERROR')
                    curr_matching.append(-200)
                    curr_logprobs.append(-200)
                    curr_translated.append('PARSING ERROR')
                    curr_overall.append(-200)
                    continue
            most_similar_idx, matching_score = top_k_similar(sentence_model, processed, action_list_embedding, device, top_k=1)
            most_similar_idx, matching_score = most_similar_idx[0], matching_score[0]
            overall_score = _get_score(matching_score, logprob)
            '''matching_score + beta * log_prob'''

            '''indexes action list with most similar index for action'''
            translated_action = action_list[most_similar_idx]

            if verbose:
                print(f'** {generated_text} ({translated_action}; matching_score={matching_score:.2f}; mean_logprob={logprob:.2f}); overall={overall_score:.2f}')

            # record metrics for each output
            '''generated_text contains raw LLM output
               translated_action contains matched/converted action from action_list (list of defined actions in VH)
               curr_overal: tracks overall score/performance of plan
            '''
            curr_matching.append(matching_score)
            curr_translated.append(translated_action)
            curr_logprobs.append(logprob)
            curr_generated.append(generated_text)
            # penalize seen actions
            if translated_action in all_translated_actions:
                if verbose:
                    print('=' * 40 + f'\n== {translated_action} has been seen, assigning score 0...\n' + '=' * 40)
                curr_overall.append(-100)
            else:
                curr_overall.append(overall_score)

        # stop when model thinks it's finished or format is wrong (very unlikely in practice)
        num_to_look_at = int(percent_terminate * default_params['n'])
        '''sort the log probabilities of choices in increasing order + pick most 'k' likely as final choice for next step'''
        highest_ids = np.argsort(curr_logprobs)[-num_to_look_at:]

        nogen_terminate = True; score_terminate = False; error_message = None

        for idx in highest_ids:
            if len(curr_generated[idx]) > 0 and (curr_step == 0 or curr_generated[idx][:4] == 'Step'):
                nogen_terminate = False

        if nogen_terminate:
            if verbose:
                print(f'** model thinks it should terminate {generated_text}')

            error_message = f'No plan generated: model thinks it should terminate'

            return None, None, nogen_terminate, score_terminate, error_message

        # calculate most likely step ===================================
        '''compare best score with cutoff threshold: cutoff threshold not implemented by default
           best_idx: proposed action/step with best overall score
        '''
        highest_score = np.max(curr_overall)
        best_idx = np.argsort(curr_overall)[-1]
        if cutoff_threshold != -100 and highest_score < cutoff_threshold:
            score_terminate = True
            if verbose:
                print(f'## STOP GENERATION because best score after {default_params["n"]} attempts was {curr_generated[best_idx]} ({highest_score} < {cutoff_threshold})')

            error_message = f'STOP GENERATION because best score after {default_params["n"]} attempts was {curr_generated[best_idx]} ({highest_score} < {cutoff_threshold})'

            return None, curr_translated[best_idx], nogen_terminate, score_terminate, error_message

        # select the previously generated output whose score is the highest
        '''uses args.translated_condition option: takes best from translated actions (rather than generated text) '''
        if translated_condition:
            best_curr = curr_translated[best_idx]
            best_curr = best_curr[0].upper() + best_curr[1:]
            best_curr = best_curr.replace('_', ' ')
            if curr_step == 0:
                best_curr = f' {best_curr}'
            else:
                best_curr = f'Step {curr_step + 1}: {best_curr}'
        else:
            best_curr = curr_generated[best_idx]
        if verbose:
            print(f'## selecting best-score output "{best_curr}" (score: {highest_score}; raw: {curr_generated[best_idx]}; translated: {curr_translated[best_idx]})\n')

        return best_curr, curr_translated[best_idx], nogen_terminate, score_terminate, error_message



    scene_environment = SceneGym(scene_path, scene_num, task_prompt)
    prompt_generator = PromptGenerator(prompt_args)

    default_params = copy.deepcopy(api_params)
    reasons_params = copy.deepcopy(api_params)

    # stop when seeing a new line since we are generating one action per iter
    default_params['stop'] = '\n'
    

    full_text = example + task_prompt  if not step_by_step else example + task_prompt + '\nLet\'s think step by step.'

    ongoing_text = example + task_prompt + '\nStep 1:' if not step_by_step else example + task_prompt + '\nLet\'s think step by step.' + '\nStep 1:'
    final_text = example + task_prompt

    all_translated_actions = []
    final_translated_actions = []

    all_errors = []

    curr_step = 0; total_steps = 0
    executed = True


    #track errors until escape step

    while curr_step < max_steps and total_steps < max_steps*2:
        #pdb.set_trace()
        no_gen_error = None; score_error = None; parsing_error = None; empty_program_error = None; precond_error = None; check_script_error = None

        best_curr, translated_action, nogen_terminate, score_terminate, error_message = _generate_action(ongoing_text, default_params)

        # if prev. step not executed: remove error and bad step before adding new step
        if not executed:
            ongoing_text = '\n'.join(ongoing_text.split('\n')[:-3]) + '\nStep 1:' if curr_step==0  else '\n'.join(ongoing_text.split('\n')[:-3]) + '\n'
            executed = True


        #failure check 1: no_gen_terminate
        if nogen_terminate:
            executed = True
            no_gen_error = error_message
            break

        #failure check 2: score terminate
        if score_terminate:
            executed = True
            score_error = error_message
            break


        #add best step to plan + continue
        
        full_text += f'\nStep 1: {best_curr}' if curr_step==0 else f'{best_curr}\n'
        ongoing_text += f'{best_curr}\n'

        all_translated_actions.append(translated_action)
        total_steps +=1


        #check for execution/precondition errors
        formatted_text = _format_api_output(ongoing_text.strip())

        if raw_lm:
            program_lines, parse_info = str2program_list(ongoing_text.split('\n')[1:])
            program_text = '\n'.join(program_lines).strip()

        else:
            matched_program_text = '\n'.join(all_translated_actions).strip()
            program_lines, parse_info = str2program_list(all_translated_actions)
            program_lines = remove_same_consecutive(program_lines)
            program_text = '\n'.join(program_lines).strip()

        #failure check 3: parsing error
        if parse_info['parsibility']==0:
            executed = False
            parsing_error = parse_info['parsing_error']

            all_errors.append(parsing_error)

            error_prompt = prompt_generator.generate_prompt('parsibility', parsing_error, total_steps, best_curr, translated_action)

            full_text += '{}\n'.format(error_prompt)
            ongoing_text += '{}\n'.format(error_prompt)
            continue

        parsed_program_lines = arg2abstract(program_lines)

        #failure check 4: empty program error
        if len(parsed_program_lines) == 0:
            executed = False
            empty_program_error = 'Script Fail: empty program'

            all_errors.append(empty_program_error)
            error_prompt = prompt_generator.generate_prompt('empty_program', empty_program_error, total_steps, best_curr, translated_action)

            full_text += '{}\n'.format(error_prompt)
            ongoing_text += '{}\n'.format(error_prompt)

            continue



        #failure check 5: precondition error on the last action taken
        try:

            preconditions = get_preconds_script([parsed_program_lines[-1]], verbose=verbose).printCondsJSON()
        except ScriptFail as e:
            executed = False
            precond_error = 'ScriptFail: {}'.format(e.message)

            all_errors.append(precond_error)

            error_prompt = prompt_generator.generate_prompt('precond', precond_error, total_steps, best_curr, translated_action)

            full_text += '{}\n'.format(error_prompt)
            ongoing_text += '{}\n'.format(error_prompt)
            continue


        #take a single step/action in the VH scene
        try:
            message, message_params, graph_dict, ____, prev_graph_dict, modified_script = scene_environment.step([parsed_program_lines[-1]], preconditions)

        except Exception as e:
            message = "{}: {}".format(e.__class__.__name__, e)

        #failure check 6: executability error
        if not 'is executable' in message:
            executed = False
            check_script_error = message

            all_errors.append(check_script_error)

            error_prompt = prompt_generator.generate_prompt('check_script', check_script_error, total_steps, best_curr, translated_action, message_params)

            full_text += '{}\n'.format(error_prompt)
            ongoing_text += '{}\n'.format(error_prompt)

            #if need to get reasons:
            # append reasons prompt
            # reason_text = _generate_reasons(plan)
            # append reason_text into plan

            continue


        #if all failure checks pass: then increment step
        curr_step += 1
        #add best step to plan + continue
        final_text += f'\n{best_curr}' if curr_step > 1 else f'\nStep 1:{best_curr}'
        final_translated_actions.append(translated_action)

    #pdb.set_trace()
    info = { 'parsed_program': '\n'.join(program_lines).strip(), 'executed': executed, 'scene_path': scene_path,
    'init_graph_dict': scene_environment.initial_graph_dict, 'modified_program': modified_script.to_string(),
    'execution_error': check_script_error, 'precond_error': precond_error, 'parsing_error':parsing_error,
    'empty_program_error':empty_program_error, 'total_steps': total_steps, 'final_steps': curr_step, 'no_gen_error':no_gen_error, 'score_error':score_error,  'all_errors': '\n'.join(all_errors)}


    return _format_api_output(final_text.strip()), final_translated_actions, _format_api_output(full_text.strip()), all_translated_actions, info


def resampling_api_request(example, task_prompt, api_params, sentence_model, action_list_embedding, device, action_list, raw_lm, scene_path, scene_num, max_iters=1000, max_steps=20, verbose=False, cutoff_threshold=-100, beta=0.5, percent_terminate=0.6, engine='davinci-codex', translated_condition=False, step_by_step = False ):

    def _get_score(matching_score, log_prob):
        return matching_score + beta * log_prob

    def _format_api_output(output):
        # exclude examples
        if '\n\n' in output:
            output = output[output.index('\n\n') + 2:]
        return output.strip()

    def _generate_action(full_text, default_params):
        '''tracks all options for generated text + translated actions for the current step'''
        curr_generated = []
        curr_matching = []
        curr_logprobs = []
        curr_translated = []
        curr_overall = []

        # query api ===================================
        #add full text (prompt and query task) as 'prompt' for LLM prediction
        default_params['prompt'] = full_text
        if isinstance(engine, str):
            response = api_retry_if_failed(default_params, max_iters=max_iters, engine=engine)
        else:
            response = engine(default_params)

        '''response format: {'choices': [{'text': '<s><s><s>.....', 'logprobs': {'token_logprobs': array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}}]}

            iterates all responses + chooses best for next step?
        '''
        for i in range(default_params['n']):
            generated_text = response['choices'][i]['text']
            logprob = np.mean(response['choices'][i]['logprobs']['token_logprobs'])

            # calculate score for current step
            '''removes white space from generated text and convert to lower case
               assumes that after curr_step 0 ==> model will auto generate 'Step 2:', 'Step 3:' prefixes to generated_text
            '''
            if curr_step == 0:
                processed = generated_text.strip().lower()
            else:
                try:
                    processed = generated_text[generated_text.index(':') + 1:].strip().lower()
                except ValueError as e:
                    curr_generated.append('PARSING ERROR')
                    curr_matching.append(-200)
                    curr_logprobs.append(-200)
                    curr_translated.append('PARSING ERROR')
                    curr_overall.append(-200)
                    continue
            most_similar_idx, matching_score = top_k_similar(sentence_model, processed, action_list_embedding, device, top_k=1)
            most_similar_idx, matching_score = most_similar_idx[0], matching_score[0]
            overall_score = _get_score(matching_score, logprob)
            '''matching_score + beta * log_prob'''

            '''indexes action list with most similar index for action'''
            translated_action = action_list[most_similar_idx]

            if verbose:
                print(f'** {generated_text} ({translated_action}; matching_score={matching_score:.2f}; mean_logprob={logprob:.2f}); overall={overall_score:.2f}')

            # record metrics for each output
            '''generated_text contains raw LLM output
               translated_action contains matched/converted action from action_list (list of defined actions in VH)
               curr_overal: tracks overall score/performance of plan
            '''
            curr_matching.append(matching_score)
            curr_translated.append(translated_action)
            curr_logprobs.append(logprob)
            curr_generated.append(generated_text)
            # penalize seen actions
            if translated_action in all_translated_actions:
                if verbose:
                    print('=' * 40 + f'\n== {translated_action} has been seen, assigning score 0...\n' + '=' * 40)
                curr_overall.append(-100)
            else:
                curr_overall.append(overall_score)

        # stop when model thinks it's finished or format is wrong (very unlikely in practice)
        '''sort the log probabilities of choices in increasing order + pick most 'k' likely as final choice for next step'''
        highest_ids = np.argsort(curr_logprobs)

        nogen_terminate = True; score_terminate = False; error_message = None

        for idx in highest_ids:
            if len(curr_generated[idx]) > 0 and (curr_step == 0 or curr_generated[idx][:4] == 'Step'):
                nogen_terminate = False

        if nogen_terminate:
            if verbose:
                print(f'** model thinks it should terminate {generated_text}')

            error_message = f'No plan generated: model thinks it should terminate'

            return None, None, nogen_terminate, score_terminate, error_message

        # calculate most likely step ===================================
        '''compare best score with cutoff threshold: cutoff threshold not implemented by default
           best_idx: proposed action/step with best overall score
        '''
        highest_score = np.max(curr_overall)
        sorted_overall = np.argsort(curr_overall)
        best_idx = sorted_overall[-1]

        if cutoff_threshold != -100 and highest_score < cutoff_threshold:
            score_terminate = True
            if verbose:
                print(f'## STOP GENERATION because best score after {default_params["n"]} attempts was {curr_generated[best_idx]} ({highest_score} < {cutoff_threshold})')

            error_message = f'STOP GENERATION because best score after {default_params["n"]} attempts was {curr_generated[best_idx]} ({highest_score} < {cutoff_threshold})'

            return None, curr_translated[best_idx], nogen_terminate, score_terminate, error_message

        # select the previously generated output whose score is the highest
        '''uses args.translated_condition option: takes best from translated actions (rather than generated text) '''
        def _get_curr(curr_translated, curr_generated, translated_condition, idx):
            if translated_condition:
                curr = curr_translated[idx]
                curr = curr[0].upper() + curr[1:]
                curr = curr.replace('_', ' ')
                if curr_step == 0:
                    curr = f' {curr}'
                else:
                    curr = f'Step {curr_step + 1}: {curr}'
            else:
                curr = curr_generated[idx]

            return curr

        best_curr = _get_curr(curr_translated, curr_generated, translated_condition, best_idx)

        #store list of alternative current steps + translated steps
        alternative_curr = [_get_curr(curr_translated, curr_generated, translated_condition, i) for i in reversed(sorted_overall)]
        alternative_translated = [curr_translated[i] for i in reversed(sorted_overall)]

        if verbose:
            print(f'## selecting best-score output "{best_curr}" (score: {highest_score}; raw: {curr_generated[best_idx]}; translated: {curr_translated[best_idx]})\n')

        return alternative_curr, alternative_translated, nogen_terminate, score_terminate, error_message



    scene_environment = SceneGym(scene_path, scene_num, task_prompt)
    # prompt_generator = PromptGenerator(prompt_args)

    default_params = copy.deepcopy(api_params)
    # stop when seeing a new line since we are generating one action per iter
    default_params['stop'] = '\n'

    full_text = example + task_prompt  if not step_by_step else example + task_prompt + '\nLet\'s think step by step.'
    ongoing_text = example + task_prompt + '\nStep 1:' if not step_by_step else example + task_prompt + '\nLet\'s think step by step.' + '\nStep 1:'

    final_text = example + task_prompt

    all_translated_actions = []
    final_translated_actions = []

    all_errors = []

    curr_step = 0; total_steps = 0; curr_idx = 0
    executed = True
    

    #track errors until escape step

    while curr_step < max_steps and total_steps < max_steps*2:
        
        no_gen_error = None; score_error = None; parsing_error = None; empty_program_error = None; precond_error = None; check_script_error = None


        # accumulate output and continue
        if not executed:
            ongoing_text = '\n'.join(ongoing_text.split('\n')[:-2]) + '\nStep 1:' if curr_step==0  else '\n'.join(ongoing_text.split('\n')[:-2]) + '\n'

        if executed or curr_idx == default_params['n'] or 'PARSING ERROR' in alternative_curr[curr_idx]:
            alternative_curr, alternative_translate, nogen_terminate, score_terminate, error_message = _generate_action(ongoing_text, default_params)
            curr_idx = 0

        #best_curr = alternative_curr[curr_idx]
        #translated_action = alternative_translate[curr_idx]
        #executed = True

        #failure check 1: no_gen_terminate
        if nogen_terminate:
            executed = True if total_steps > 0 else False
            no_gen_error = error_message
            break

        #failure check 2: score terminate
        if score_terminate:
            executed = True if total_steps > 0 else False
            score_error = error_message
            break

        best_curr = alternative_curr[curr_idx]
        translated_action = alternative_translate[curr_idx]
        executed = True

        #add best step to plan + continue
        full_text += f'\nStep 1: {best_curr}' if curr_step==0 else f'{best_curr}\n'
        ongoing_text += f'{best_curr}\n'

        all_translated_actions.append(translated_action)
        total_steps +=1
        curr_idx += 1


        #check for execution/precondition errors
        formatted_text = _format_api_output(full_text.strip())

        if raw_lm:
            program_lines, parse_info = str2program_list(full_text.split('\n')[1:])
            program_text = '\n'.join(program_lines).strip()

        else:
            matched_program_text = '\n'.join(all_translated_actions).strip()
            program_lines, parse_info = str2program_list(all_translated_actions)
            program_lines = remove_same_consecutive(program_lines)
            program_text = '\n'.join(program_lines).strip()

        #failure check 3: parsing error
        if parse_info['parsibility']==0:
            executed = False
            parsing_error = parse_info['parsing_error']

            all_errors.append(parsing_error)
            continue

        parsed_program_lines = arg2abstract(program_lines)

        #failure check 4: empty program error
        if len(parsed_program_lines) == 0:
            executed = False
            empty_program_error = 'Script Fail: empty program'

            all_errors.append(empty_program_error)
            continue



        #failure check 5: precondition error on the last action taken
        try:

            preconditions = get_preconds_script([parsed_program_lines[-1]], verbose=verbose).printCondsJSON()
        except ScriptFail as e:
            executed = False
            precond_error = 'ScriptFail: {}'.format(e.message)

            all_errors.append(precond_error)
            continue


        #take a single step/action in the VH scene
        try:
            pdb.set_trace()
            message, message_params, graph_dict, ____, prev_graph_dict, modified_script = scene_environment.step([parsed_program_lines[-1]], preconditions)

        except Exception as e:
            message = "{}: {}".format(e.__class__.__name__, e)

        #failure check 6: executability error
        if not 'is executable' in message:
            executed = False
            check_script_error = message

            all_errors.append(check_script_error)
            continue


        #if all failure checks pass: then increment step
        curr_step += 1
        #add best step to plan + continue
        final_text += f'\n{best_curr}' if curr_step > 1 else f'\nStep 1:{best_curr}'
        final_translated_actions.append(translated_action)

    
    if total_steps==0:
        info = {'parsed_program': None, 'executed': executed, 'scene_path': scene_path, 'init_graph_dict': scene_environment.initial_graph_dict,'modified_program': None,'execution_error': check_script_error, 'precond_error': precond_error, 'parsing_error':parsing_error, 'empty_program_error': empty_program_error, 'total_steps':total_steps, 'final_steps': curr_step, 'no_gen_error': no_gen_error, 'score_error':score_error, 'all_errors': '\n'.join(all_errors)}

    else:
        info = { 'parsed_program': '\n'.join(program_lines).strip(), 'executed': executed, 'scene_path': scene_path,
        'init_graph_dict': scene_environment.initial_graph_dict, 'modified_program': modified_script.to_string(),
        'execution_error': check_script_error, 'precond_error': precond_error, 'parsing_error':parsing_error,
        'empty_program_error':empty_program_error, 'total_steps': total_steps, 'final_steps': curr_step, 'no_gen_error':no_gen_error, 'score_error':score_error,  'all_errors': '\n'.join(all_errors)}


    return _format_api_output(final_text.strip()), final_translated_actions, _format_api_output(full_text.strip()), all_translated_actions, info






def arg2abstract(program_lines):

    def _format_arg(arg):
        if arg in detail2abstract:
            return detail2abstract[arg]
        return arg

    _program_lines = []
    for line in program_lines:
        action, obj_names_corr, inst_nums_corr = parseStrBlock(line)
        assert len(obj_names_corr) == len(inst_nums_corr)
        obj_names_corr = [_format_arg(arg) for arg in obj_names_corr]
        if len(obj_names_corr) == 0:
            inst = f'[{action.upper()}]'
        elif len(obj_names_corr) == 1:
            inst = f'[{action.upper()}] <{obj_names_corr[0]}> ({inst_nums_corr[0]})'
        elif len(obj_names_corr) == 2:
            inst = f'[{action.upper()}] <{obj_names_corr[0]}> ({inst_nums_corr[0]}) <{obj_names_corr[1]}> ({inst_nums_corr[1]})'
        else:
            import pdb; pdb.set_trace()
            raise ValueError
        _program_lines.append(inst)
    return _program_lines

def remove_same_consecutive(program_lines):
    from itertools import groupby
    return [x[0] for x in groupby(program_lines)]

def str2program_list(program_lines):

    def _format_arg(arg):
        arg = arg.lower().strip().replace(' ', '_')
        if arg in detail2abstract:
            return detail2abstract[arg]
        return arg

    # start parsing ==============================
    # pl = program_str[program_str.index('Step 1:'):].split('\n')
    info = dict()
    info['parsing_error'] = []
    pl = program_lines
    parsed_lines = []
    success_count = 0
    for i, line in enumerate(pl):
        line = line.lower().strip()
        if len(line) == 0:
            continue
        if ':' in line:
            line = line[line.index(':') + 1:].strip()
        try:
            # try matching each possible action
            possible_parsed = OrderedDict()
            for action in EvolveGraphAction:
                action_template = action.value[2]
                expected_num_args = action.value[1]
                parsed = parse.parse(action_template, line)
                if parsed is not None:
                    assert action.name not in possible_parsed
                    if len(parsed.fixed) == expected_num_args:
                        # print(action_template, parsed, expected_num_args)
                        possible_parsed[action.name] = parsed
                    else:
                        # skip if number of parsed args does not match expected
                        pass
            assert len(possible_parsed) == 1, f'possible_parsed: {possible_parsed} does not equal to 1'
            parsed_action = list(possible_parsed.keys())[0]
            parsed_args = possible_parsed[parsed_action]
            if len(parsed_args.fixed) == 0:
                pl_str = '[{}]'
                pl_str = pl_str.format(parsed_action)
            elif len(parsed_args.fixed) == 1:
                pl_str = '[{}] <{}> (1)'
                pl_str = pl_str.format(parsed_action, _format_arg(parsed_args[0]))
            elif len(parsed_args.fixed) == 2:
                pl_str = '[{}] <{}> (1) <{}> (1)'
                pl_str = pl_str.format(parsed_action, _format_arg(parsed_args[0]), _format_arg(parsed_args[1]))
            else:
                raise NotImplementedError
            parsed_lines.append(pl_str)
            success_count += 1
        except AssertionError as e:
            message = "| {} | {} | '{}'".format(e.__class__.__name__, e, line)
            info['parsing_error'].append(message)
            line = pl[i]
            if ':' in line:
                line = line[line.index(':') + 1:].strip()
            # none of these is likely going to work, but parse it this way to obey vh format
            if len(line) > 0:
                words = line.split(' ')
                if len(words) == 1:
                    pl_str = '[{}]'.format(words[0].upper())
                elif len(words) == 2:
                    pl_str = '[{}] <{}> (1)'.format(words[0].upper(), words[1])
                elif len(words) == 3:
                    pl_str = '[{}] <{}> (1) <{}> (1)'.format(words[0].upper(), words[1], words[2])
                else:
                    pl_str = '[{}] <{}> (1)'.format(words[0].upper(), '_'.join(words[1:]))
            else:
                pl_str = '[EMPTY]'
            parsed_lines.append(pl_str)
    info['num_parsed_lines'] = len(parsed_lines)
    info['num_total_lines'] = len(pl)
    if len(pl) != 0:
        info['parsibility'] = success_count / len(pl)
    else:
        info['parsibility'] = 0
    return parsed_lines, info

def LCS(X, Y):

    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]
    longest_L = [[[]] * (n + 1) for i in range(m + 1)]
    longest = 0
    lcs_set = []

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
                longest_L[i][j] = []
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
                longest_L[i][j] = longest_L[i - 1][j - 1] + [X[i - 1]]
                if L[i][j] > longest:
                    lcs_set = []
                    lcs_set.append(longest_L[i][j])
                    longest = L[i][j]
                elif L[i][j] == longest and longest != 0:
                    lcs_set.append(longest_L[i][j])
            else:
                if L[i - 1][j] > L[i][j - 1]:
                    L[i][j] = L[i - 1][j]
                    longest_L[i][j] = longest_L[i - 1][j]
                else:
                    L[i][j] = L[i][j - 1]
                    longest_L[i][j] = longest_L[i][j - 1]

    if len(lcs_set) > 0:
        return lcs_set[0]
    else:
        return lcs_set

def preprocess_program_lines_for_lcs(program_lines):
    """given generated/gt program lines, convert to english step using action template + replace all arguments to merged common name"""
    def _format_arg(arg):
        arg = arg.lower().strip().replace(' ', '_')
        if arg in detail2abstract:
            return detail2abstract[arg]
        return arg

    output = []
    for line in program_lines:
        script_action, script_args, _ = parseStrBlock(line)
        script_args = [_format_arg(arg) for arg in script_args]
        action_num_args = None
        action_template = None
        for _action in EvolveGraphAction:
            if _action.name.upper() == script_action.upper():
                action_num_args = _action.value[1]
                action_template = _action.value[2]
                break
        if (action_num_args is not None and action_num_args == len(script_args)):
            action_str = action_template.format(*script_args)
        else:
            action_str = line
            # print(f'** FAILED to process lcs: "{line}"')
        output.append(action_str)

    return output

def get_avg_program_length(program_paths):
    lengths = []
    for path in program_paths:
        program_lines = load_txt(path).strip().split('\n')
        if len(program_lines) == 1 and len(program_lines[0]) == 0:
            lengths.append(0)
        else:
            lengths.append(len(program_lines))
    return np.mean(lengths)
