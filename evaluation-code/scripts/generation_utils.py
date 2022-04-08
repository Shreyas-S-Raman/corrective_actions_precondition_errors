import json
import copy
import os
from enum import Enum
import os
import openai
from sentence_transformers import util as st_utils
import numpy as np
from vh_configs import *
from collections import OrderedDict
import parse
import torch
import time

API_KEYS = ['PUT OPENAI KEY HERE']
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

def one_shot_api_request(example, task_prompt, api_params, sentence_model, action_list_embedding, device, action_list, max_iters=1000, beta=0.5, raw_lm=False, engine='davinci-codex'):

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
    return best_generated, best_translated_actions

def iterative_api_request(example, task_prompt, api_params, sentence_model, action_list_embedding, device, action_list, max_iters=1000, max_steps=20, verbose=False, cutoff_threshold=-100, beta=0.5, percent_terminate=0.6, engine='davinci-codex', translated_condition=False):

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
    full_text = example + task_prompt + '\nStep 1:'
    all_translated_actions = []
    curr_step = 0
    while curr_step < max_steps:

        curr_generated = []
        curr_matching = []
        curr_logprobs = []
        curr_translated = []
        curr_overall = []

        # query api ===================================
        default_params['prompt'] = full_text
        if isinstance(engine, str):
            response = api_retry_if_failed(default_params, max_iters=max_iters, engine=engine)
        else:
            response = engine(default_params)
        for i in range(default_params['n']):
            generated_text = response['choices'][i]['text']
            logprob = np.mean(response['choices'][i]['logprobs']['token_logprobs'])

            # calculate score for current step
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
            translated_action = action_list[most_similar_idx]

            if verbose:
                print(f'** {generated_text} ({translated_action}; matching_score={matching_score:.2f}; mean_logprob={logprob:.2f}); overall={overall_score:.2f}')

            # record metrics for each output
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
        highest_score = np.max(curr_overall)
        best_idx = np.argsort(curr_overall)[-1]
        if cutoff_threshold != -100 and highest_score < cutoff_threshold:
            if verbose:
                print(f'## STOP GENERATION because best score after {default_params["n"]} attempts was {curr_generated[best_idx]} ({highest_score} < {cutoff_threshold})')
            break
        # select the previously generated output whose score is the highest
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
        full_text += f'{best_curr}\n'
        all_translated_actions.append(curr_translated[best_idx])
        curr_step += 1

    return _format_api_output(full_text.strip()), all_translated_actions

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