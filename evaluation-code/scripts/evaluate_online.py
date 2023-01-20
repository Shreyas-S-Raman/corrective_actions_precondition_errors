import json
import sys
import os

sys.path.append('../simulation')
from evolving_graph.utils import *
from evolving_graph.environment import *
from evolving_graph.execution import *
from evolving_graph.scripts import *
from evolving_graph.preparation import *
from evolving_graph.check_programs import *
sys.path.append('../dataset_utils')
from add_preconds import *
import wandb
import multiprocessing as mp
import numpy as np
from arguments_new import get_args
from sentence_transformers import SentenceTransformer
from generation_utils import *
import time
import torch
from tqdm import tqdm
import glob
import pdb

# multiprocessing version
def evaluate_script(kwargs):
    '''kwargs dictionary: {'script_path': '../generated_programs/experiment_2/parsed/browse_internet-1.txt', 'scene_path': '../example_graphs/TrimmedTestScene1_graph.json', 'verbose': False}'''

    script_path, scene_path = kwargs['script_path'], kwargs['scene_path']
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = False
    assert '.txt' in script_path
    script_fname = os.path.basename(script_path)

    '''task name e.g. dust-1'''
    title = script_fname[:-4]

    # load script and inferred preconds
    try:
        '''creates evolving graph object 'script': <evolving_graph.scripts.Script object at 0x7f5571ab2208>'''
        script = read_script(script_path)
    except Exception as e:
        info = {'parsed_program': None,
            'executed': None,
            'scene_path': scene_path,
            'script_path': script_path,
            'init_graph_dict': None,
            'modified_program': None,
            'execution_error': None,
            'precond_error': None,
            }
        info['execution_error'] = "FAILED TO READ SCRIPT | {} | {}".format(e.__class__.__name__, e)
        info['executed'] = False
        if verbose:
            print(info['execution_error'])
            print('[{}] is NOT executable\n'.format(script_fname))
        return info

    '''outputs VH converted actions: e.g. ['[PUTBACK] <chair> (1) <chair> (1)', '[SWITCHON] <computer> (1)', '[WALK] <centerpiece> (1)', '[WALK] <desk> (1)', '[TURNTO] <computer> (1)']'''
    program_lines = script_to_list_string(script)

    # define returned metrics, such that all scripts would have all the keys present
    '''stores precondition or execution error'''
    info = {'parsed_program': '\n'.join(program_lines).strip(),
            'executed': None,
            'scene_path': scene_path,
            'script_path': script_path,
            'init_graph_dict': None,
            'modified_program': None,
            'execution_error': None,
            'precond_error': None,
            }

    '''processes VH converted actions: e.g. ['[PUTBACK] <chair> (1) <chair> (1)', '[SWITCHON] <computer> (1)', '[WALK] <centerpiece> (1)', '[WALK] <desk> (1)', '[TURNTO] <computer> (1)']'''
    program_lines = arg2abstract(program_lines)

    if len(program_lines) == 0:
        info['execution_error'] = 'empty program'
        info['executed'] = False
        if verbose:
            print(info['execution_error'])
            print('[{}] is NOT executable\n'.format(script_fname))
        return info

    if verbose:
        print(script_fname)
        print('\n'.join(program_lines))

    '''converts program_lines (VH converted actions) to dictionary of preconditions
        e.g. {'is_off': ['computer', '1']}, {'atreach': [['computer', '1'], ['desk', '1']]}, {'in': [['computer', '1'], ['desk', '1']]}, {'free': ['chair', '1']}]
    '''
    try:
        precond = get_preconds_script(program_lines, verbose=verbose).printCondsJSON()
    except ScriptFail as e:
        info['precond_error'] = 'ScriptFail: {}'.format(e.message)
        info['executed'] = False
        if verbose:
            print(info['precond_error'])
            print('[{}] is NOT executable\n'.format(script_fname))
        return info

    '''outputs evolving graph object for scene_path | 'env_graph': <evolving_graph.environment.EnvironmentGraph object at 0x7f556f01b828>'''
    env_graph = utils.load_graph(scene_path)
    # prepare env graph from precond and script
    '''generates dictionary containing nodes and edges (preconditions in scene): dict_keys(['nodes', 'edges'])'''
    graph_dict = env_graph.to_dict()

    if verbose:
        print('PRECOND: {}'.format(precond))

    try:
        '''
        INPUTS
        program_lines: VH converted actions e.g. ['[PUTBACK] <chair> (1) <chair> (1)', '[SWITCHON] <computer> (1)', '[WALK] <centerpiece> (1)', '[WALK] <desk> (1)', '[TURNTO] <computer> (1)']
        precond: program_lines (VH converted actions) converted to a dictionary of preconditions based on generated actions e.g. {'is_off': ['computer', '1']}, {'atreach': [['computer', '1'], ['desk', '1']]}, {'in': [['computer', '1'], ['desk', '1']]}, {'free': ['chair', '1']}]
        scene_path: path to associated scene graph e.g. ../example_graphs/TrimmedTestScene1_graph.json
        graph_dict: dictionary containing nodes/edges for scene preconditions: dict_keys(['nodes', 'edges']) ==> precondition DAG
        modify_graph: boolean ==> modify scene graph + precondition or not

        check_script() runs the program_lines on the scene graph whilst checking precond
        '''
        (message, __, init_graph_dict, final_state, graph_state_list, input_graph, id_mapping, _, graph_helper, modified_script, percent_executed) = check_script(
                                        program_lines,
                                        precond,
                                        scene_path,
                                        inp_graph_dict=graph_dict,
                                        modify_graph=True,
                                        id_mapping={},
                                        info={})
        '''
        OUTPUTS
        message: contains 'is executable' string if successfully executed program_lines in graph_dict/scene_path given precond of actions
        init_graph_dict: same as graph_dict i.e. initial precondition DAG for env dict_keys(['nodes', 'edges'])
        final_state: evolving graph output after actions i.e. <class 'evolving_graph.environment.EnvironmentState'>
        graph_state_list: list of dictionaries + final nodes/edges of precondition DAG after actions dict_keys(['nodes', 'edges'])
        input_graph: same as init_graph_dict/graph_dict i.e. precondition DAG (as dict of nodes/edges) input into check_script
        modified_script: evolving graph script i.e. <evolving_graph.scripts.Script object at 0x7fd7902f8ac8>


        modified_script.to_string(): string output of final VH-based action sequence e.g.
        [PUTBACK] <chair> (103) <chair> (103)
        [SWITCHON] <computer> (170)
        [WALK] <centerpiece> (1000)
        [WALK] <desk> (104)
        [TURNTO] <computer> (170)
        '''

        info['init_graph_dict'] = init_graph_dict
        info['modified_program'] = modified_script.to_string()
    except Exception as e:
        message = "{}: {}".format(e.__class__.__name__, e)
        print('** check_script FAILED: ' + message)

    if verbose:
        print(message)

    '''checks message returned by check_script or exception'''
    if 'is executable' in message:
        info['executed'] = True
        if verbose:
            print('[{}] is executable\n'.format(script_fname))
        return info
    else:
        info['executed'] = False
        info['execution_error'] = message
        if verbose:
            print('[{}] is NOT executable\n'.format(script_fname))
        return info
    '''info dict output
    dict_keys(['parsed_program', 'executed', 'scene_path', 'script_path', 'init_graph_dict', 'modified_program', 'execution_error', 'precond_error'])

    parsed_program: read script_path (evolving graph obj) converted to string ==> [PUTBACK] <chair> (1) <chair> (1) [SWITCHON] <computer> (1) [WALK] <centerpiece> (1) [WALK] <desk> (1)
    executed: boolean ==> T/F
    scene_path: path to scene graph e.g. ../example_graphs/TrimmedTestScene1_graph.json
    script_path: path to generated action seq e.g. ../generated_programs/experiment_2/parsed/browse_internet-1.txt
    modified_program: modified_script output (evolving graph obj) converted to string: assigns object ids in VH to objects? ==> [PUTBACK] <chair> (103) <chair> (103) [SWITCHON] <computer> (170) [WALK] <centerpiece> (1000) [WALK] <desk> (104) [TURNTO] <computer> (170)
    init_graph_dict: same as graph_dict i.e. initial precondition DAG for scene dict_keys(['nodes', 'edges'])
    execution_error/precond_error: errors in execute_script or in generating precondition dict from program_lines, respectively
    '''

def evaluate_all_scripts(script_paths, args, evaluated_scenes=range(1, 8)):
    """find all scripts in script_dir and evaluate in all given scenes"""
    # construct args for all 7 scenes
    '''scene_paths contains: list of paths e.g. '../example_graphs/TrimmedTestScene2_graph.json' (scene graph from environment)'''
    '''script paths contains: list of paths e.g. '../generated_programs/experiment_2/parsed/go_to_sleep-19.txt' (generated program)'''

    scene_paths = []
    for scene_num in evaluated_scenes:
        scene_paths += [args.scene_path_format.format(scene_num)] * len(script_paths)
    _script_paths = script_paths
    assert len(_script_paths) == len(scene_paths)
    pool_kwargs = []
    for script_path, scene_path in zip(_script_paths, scene_paths):
        pool_kwargs.append(dict(script_path=script_path, scene_path=scene_path, verbose=False))

    if args.debug or args.num_workers == 1:
        results = []
        for kwargs in pool_kwargs:
            r = evaluate_script(kwargs)
            results.append(r)
    else:
        pool = mp.Pool(processes=args.num_workers)
        results = pool.map(evaluate_script, pool_kwargs)
        pool.close()
        pool.join()

    return results

def generate_program(query_task_desc, example_path, scene_path, scene, sentence_model, action_list, action_list_embedding, generation_info, args):
    #pdb.set_trace()
    query_task, query_desc = query_task_desc
    # determine saving file name
    info = generation_info[(query_task, query_desc, scene)]
    # generate from openai api ============================================
    # format prompt and query openai api
    example_str = construct_example(example_path, add_desc=args.add_desc) if not args.finetuned else ''
    if args.verbose:
        print('*************** EXAMPLE ***************\n{}'.format(example_str.strip()))
    if args.add_desc:
        assert query_desc is not None
        task_prompt_formatted = 'Task: {}\nDescription: {}'.format(query_task, query_desc)
    else:
        task_prompt_formatted = 'Task: {}'.format(query_task)

    if args.iterative and not args.raw_lm:

        if args.resampling:
            #pdb.set_trace()
            final_raw_text, matched_program_lines, full_raw_text, full_generated_lines, full_matched_program_lines, task_info = resampling_api_request(example_str, task_prompt_formatted, args.api_params, sentence_model, action_list_embedding, args.device, action_list, args.raw_lm, scene_path, scene, max_iters=1000, max_steps=args.api_max_steps,
            verbose=args.debug and args.verbose, cutoff_threshold=args.api_cutoff_threshold,
            beta=args.api_beta, percent_terminate=args.api_percent_terminate, engine=args.engine, translated_condition = args.translated_condition, step_by_step = args.step_by_step)

        elif not args.one_error:
            #pdb.set_trace() 
            final_raw_text, matched_program_lines, full_raw_text, full_generated_lines, full_matched_program_lines, task_info = online_api_request(example_str, task_prompt_formatted, args.api_params, sentence_model, action_list_embedding, args.device, action_list, args.raw_lm, scene_path, scene, {'prompt_template': args.prompt_template, 'custom_cause':args.custom_cause, 'error_information':args.error_information, 'suggestion_no':args.suggestion_no, 'third_person':args.third_person, 'chosen_causal_reprompts':args.chosen_causal_reprompts, 'chosen_context': args.chosen_context}, max_iters=1000, max_steps=args.api_max_steps,
            verbose=args.debug and args.verbose, cutoff_threshold=args.api_cutoff_threshold,
            beta=args.api_beta, percent_terminate=args.api_percent_terminate, engine=args.engine, translated_condition = args.translated_condition, step_by_step = args.step_by_step, add_executable_mask = args.add_executable_mask)
        else:
            
            final_raw_text, matched_program_lines, full_raw_text, full_generated_lines, full_matched_program_lines, task_info = online_api_request_one_error(example_str, task_prompt_formatted, args.api_params, sentence_model, action_list_embedding, args.device, action_list, args.raw_lm, scene_path, scene, {'prompt_template': args.prompt_template, 'custom_cause':args.custom_cause, 'error_information':args.error_information, 'suggestion_no':args.suggestion_no, 'third_person':args.third_person,'chosen_causal_reprompts':args.chosen_causal_reprompts, 'chosen_context': args.chosen_context}, max_iters=1000, max_steps=args.api_max_steps,
            verbose=args.debug and args.verbose, cutoff_threshold=args.api_cutoff_threshold,
            beta=args.api_beta, percent_terminate=args.api_percent_terminate, engine=args.engine, translated_condition = args.translated_condition, step_by_step = args.step_by_step, add_executable_mask=args.add_executable_mask)

        task_info['script_path'] = info['parsed_save_path']

    else:
        raise NotImplementedError()

    if args.verbose:
        print('*************** RAW TEXT ***************\n{}'.format(final_raw_text))

    # save the raw output
    save_txt(info['raw_save_path'], final_raw_text)
    save_txt(info['full_save_path'], full_raw_text)


    if args.raw_lm:
        parsed_program_lines, parse_info = str2program_list(final_raw_text.split('\n')[1:])
        parsed_program_text = '\n'.join(parsed_program_lines).strip()

        if args.verbose:
            print('*************** PARSED TEXT ***************\n{}'.format(parsed_program_text))
        save_txt(info['parsed_save_path'], parsed_program_text)

    else:
        # convert matched program to str and save to disk
        '''matched_program_lines: list of converted actions (['find tv', 'find radio', 'watch television', 'watch radio', 'turn to stereo', 'watch check', 'put back pants', 'turn to shirt', 'turn to eye_left', 'watch tv', 'turn to lighter', 'watch lightswitch', 'turn to television', 'turn to orange', 'put lighter on tray'])
           ==> join together converted actions to single string (each step on different line) e.g. matched_program_text
        '''
        matched_program_text = '\n'.join(matched_program_lines).strip()
        full_matched_program_text = '\n'.join(full_matched_program_lines).strip()
        full_generated_text = '\n'.join(full_generated_lines).strip()

        if args.verbose:
            print('*************** MATCHED TEXT ***************\n{}'.format(matched_program_text))

        '''matched_program_text: join together converted actions to single string (each step on different line)'''
        save_txt(info['matched_save_path'], matched_program_text)
        save_txt(info['full_matched_save_path'], full_matched_program_text)
        save_txt(info['full_generated_save_path'], full_generated_text)
        # parse matched actions into vh program ============================================

        parsed_program_lines, parse_info = str2program_list(matched_program_lines)
        full_parsed_program_lines, __ = str2program_list(full_matched_program_lines)
        full_parsed_generated_lines, __ = str2program_list(full_generated_lines)

        # remove consecutive actions
        parsed_program_lines = remove_same_consecutive(parsed_program_lines)
        parsed_program_text = '\n'.join(parsed_program_lines).strip()

        full_parsed_program_lines = remove_same_consecutive(full_parsed_program_lines)
        full_parsed_program_text = '\n'.join(full_parsed_program_lines).strip()

        full_parsed_generated_lines = remove_same_consecutive(full_parsed_generated_lines)
        full_parsed_generated_text = '\n'.join(full_parsed_generated_lines).strip()

        if args.verbose:
            print('*************** PARSED TEXT ***************\n{}'.format(parsed_program_text))
        save_txt(info['parsed_save_path'], parsed_program_text)
        save_txt(info['full_parsed_save_path'], full_parsed_program_text)
        save_txt(info['full_generated_parsed_save_path'], full_parsed_generated_text)

        # save generation info
        generation_info[(query_task, query_desc, scene)]['matched_text'] = matched_program_text
        generation_info[(query_task, query_desc, scene)]['full_matched_text'] = full_matched_program_text
        generation_info[(query_task, query_desc, scene)]['full_generated_text'] = full_generated_text

    # save generation info
    generation_info[(query_task, query_desc, scene)]['example_text'] = example_str
    generation_info[(query_task, query_desc, scene)]['final_raw_text'] = final_raw_text
    generation_info[(query_task, query_desc, scene)]['full_raw_text'] = full_raw_text
    generation_info[(query_task, query_desc, scene)]['parsed_text'] = parsed_program_text
    generation_info[(query_task, query_desc, scene)]['full_parsed_text'] = full_parsed_program_text
    generation_info[(query_task, query_desc, scene)]['parsed_program_lines'] = parsed_program_lines
    generation_info[(query_task, query_desc, scene)]['full_parsed_program_lines'] = full_parsed_program_lines
    generation_info[(query_task, query_desc, scene)]['parsibility'] = parse_info['parsibility']

    return task_info

def generate_all_tasks(generation_info, sentence_model, title_embedding, action_list, action_list_embedding, args):
    bar = tqdm(total=len(generation_info))

    #create list of all scene paths (used to check executability)
    results = []

    for i, (query_task, query_desc, scene) in enumerate(generation_info):

        scene_path = args.scene_path_format.format(scene)

        if args.use_similar_example:
            example_path_idx = select_most_similar_example_idx(sentence_model=sentence_model, query_title=query_task, title_embedding=title_embedding, device=args.device)
            example_path = args.example_paths[example_path_idx]
        else:
            example_path = args.example_path
        generation_info[(query_task, query_desc, scene)]['example_path'] = example_path

        # only generate program if not already exists
        parsed_save_path = generation_info[(query_task, query_desc, scene)]['parsed_save_path']
        if not os.path.exists(parsed_save_path) or args.debug or args.fresh:
            
            info = generate_program((query_task, query_desc), example_path, scene_path, scene, sentence_model, action_list, action_list_embedding, generation_info, args)
            results.append(info)
        bar.update(1)
    #pdb.set_trace()
    return results

def evaluate_pairwise_precision(generation_info, executable_only=False):


    def _generate_line_pair_counts(lines, gt=False):
        
        if gt is False:
            output_counts = {}

            for i in range(len(lines)):
                for j in range(i+1, len(lines)):

                    pair = ' '.join([lines[i], lines[j]])

                    if pair not in output_counts:
                        output_counts[pair] = 0
                    
                    output_counts[pair] += 1
        
        else:

            output_counts = []

            for k in range(len(lines)):

                pair_counts = {}

                for i in range(len(lines[k])):
                    for j in range(i+1, len(lines[k])):

                        pair = ' '.join([lines[k][i], lines[k][j]])

                        if pair not in output_counts:
                            pair_counts[pair] = 0
                        
                        pair_counts[pair] += 1
                
                output_counts.append(pair_counts)

        return output_counts
    
    def _compute_precision(program_lines, gt_program_lines):

        precision = 0.0

        for gt_count_dict in gt_program_lines:


            for pair in program_lines.keys():

                precision += min(program_lines[pair], gt_count_dict[pair] if pair in gt_count_dict else 0.0)
        

        return precision/len(list(program_lines.keys())) if len(list(program_lines.keys())) > 0.0 else 0.0
                





    #evaluate the n-step similarity for each task
    task_pairwise_precision = dict()

    for (task, desc, scene), info in generation_info.items():

        try:
            program_lines = info['parsed_program_lines']
        except KeyError as e:
            program_lines = load_txt(info['parsed_save_path']).split('\n')
        
        if executable_only and info['executed'] is False:
            stop_idx = int(info['percent_executed']*len(program_lines))
            program_lines = program_lines[:stop_idx]
        

        #if program is empty assign 0.0 n-step similarity
        if len(program_lines) == 0 or (len(program_lines)==1 and len(program_lines[0]) == 0):
            task_pairwise_precision[(task, desc, scene)] = 0.0
    
        else:
            program_lines = preprocess_program_lines_for_lcs(program_lines)

            gt_program_lines = [ preprocess_program_lines_for_lcs(x) for x in info['gt_program_lines'] ]


            mean_gt_length = np.mean(list(map(lambda x: len(x), gt_program_lines)))
            brevity_pen = min(1, np.exp( 1 - (len(program_lines)/mean_gt_length) ))


            program_lines = _generate_line_pair_counts(program_lines, gt=False)
            gt_program_lines = _generate_line_pair_counts(gt_program_lines, gt = True)


            precision_sum = _compute_precision(program_lines, gt_program_lines)
            precision_sum = (precision_sum/len(gt_program_lines))*brevity_pen

            assert (task, desc, scene) not in task_pairwise_precision
            

            task_pairwise_precision[(task, desc, scene)] = precision_sum
            

        info['pairwise_precision'] = task_pairwise_precision[(task, desc, scene)]
            

    avg_pairwise_precision = np.mean(list(task_pairwise_precision.values()))
    

    return avg_pairwise_precision

    

def evaluate_n_step_similarity(generation_info, n=4, executable_only=False):


    def _get_precision(n_step_windows, n_step_gt_windows):

        gen_windows = {}; gt_windows = {}

        for i in range(len(n_step_windows)):
            window = ' '.join(n_step_windows[i])

            if window not in gen_windows:
                gen_windows[window] = 0
            gen_windows[window] += 1
        
        for i in range(len(n_step_gt_windows)):
            for j in range(len(n_step_gt_windows[i])):
                
                gt_window = ' '.join(n_step_gt_windows[i][j])

                if gt_window not in gt_windows:
                    gt_windows[gt_window] = 0
                gt_windows[gt_window] += 1
        

        precision = 0.0; precision_denom = np.sum(list(gen_windows.values()))

        for window in gen_windows.keys():
            precision += min(gen_windows[window], gt_windows[window] if window in gt_windows else 0.0)
        
        return precision/precision_denom if precision_denom > 0 else 0.0

    def _get_sliding_windows(array, window_size):
        start = 0

        sub_windows = (start + np.expand_dims(np.arange(window_size),0) + np.expand_dims(np.arange(len(array)-window_size+1), 0).T)
        
        return array[sub_windows]


    #evaluate the n-step similarity for each task
    task_nstep_similarity_sum = dict()

    for (task, desc, scene), info in generation_info.items():

        try:
            program_lines = info['parsed_program_lines']
        except KeyError as e:
            program_lines = load_txt(info['parsed_save_path']).split('\n')
        
        if executable_only and info['executed'] is False:
            stop_idx = int(info['percent_executed']*len(program_lines))
            program_lines = program_lines[:stop_idx]
        

        #if program is empty assign 0.0 n-step similarity
        if len(program_lines) == 0 or (len(program_lines)==1 and len(program_lines[0]) == 0):
            task_nstep_similarity_sum[(task, desc, scene)] = 0.0
    
        else:
            program_lines = preprocess_program_lines_for_lcs(program_lines)

            gt_program_lines = [ preprocess_program_lines_for_lcs(x) for x in info['gt_program_lines'] ]
            
            mean_gt_length = np.mean(list(map(lambda x: len(x), gt_program_lines)))
            brevity_pen = min(1, np.exp( (1 - len(program_lines) )/mean_gt_length) )

            precision_sum = 0.0

            for i in range(1, n+1):
                
                # shape: (num_windows , n)
                n_step_windows = _get_sliding_windows(np.array(program_lines), i)
                
                # shape: (num_examples, num_windows, n)
                n_step_gt_windows = [_get_sliding_windows(np.array(line), i) for line in gt_program_lines]


                precision_sum += _get_precision(n_step_windows, n_step_gt_windows)
            

            precision_sum = (precision_sum/n)*brevity_pen

            assert (task, desc, scene) not in task_nstep_similarity_sum
            

            task_nstep_similarity_sum[(task, desc, scene)] = precision_sum
            

        info['n_step_similarity'] = task_nstep_similarity_sum[(task, desc, scene)]
            

    avg_nstep_similarty_sum = np.mean(list(task_nstep_similarity_sum.values()))
    

    return avg_nstep_similarty_sum

        

def evaluate_lcs_score(generation_info, verbose=False):
    # evaluate lcs score for each task
    task_lcs = dict()
    task_sketch_lcs = dict()
    for (task, desc, scene), info in generation_info.items():
        try:
            program_lines = info['parsed_program_lines']
        except KeyError as e:
            program_lines = load_txt(info['parsed_save_path']).split('\n')
        # init default values
        most_similar_gt_program_text = info['gt_program_text'][0]
        most_similar_gt_sketch_text = ''
        task_sketch_lcs[(task, desc, scene)] = -1
        # if the program is empty, simply assign lcs of 0
        if len(program_lines) == 0 or (len(program_lines) == 1 and len(program_lines[0]) == 0):
            task_lcs[(task, desc, scene)] = 0
            if verbose:
                print('*' * 10 + f' {task} ' + '*' * 10)
                print('*' * 5 + f' program length is 0 ' + '*' * 5)
                print('*' * 40)
                print()
        else:
            program_lines = preprocess_program_lines_for_lcs(program_lines)
            # iterate through all gt programs and use the highest lcs obtained
            curr_lcs = []
            for gt_program_lines in info['gt_program_lines']:
                gt_program_lines = preprocess_program_lines_for_lcs(gt_program_lines)
                lcs = LCS(program_lines, gt_program_lines)
                lcs_score = len(lcs) / (float(max(len(program_lines), len(gt_program_lines))))
                curr_lcs.append(lcs_score)
            assert (task, desc, scene) not in task_lcs
            most_similar_gt_idx = np.argsort(curr_lcs)[-1]
            task_lcs[(task, desc, scene)] = curr_lcs[most_similar_gt_idx]
            most_similar_gt_program_text = info['gt_program_text'][most_similar_gt_idx]
            if verbose:
                print('*' * 10 + f' {task} ' + '*' * 10)
                print('*' * 5 + f' {curr_lcs} ' + '*' * 5)
                print('\n* '.join(program_lines))
                print('-' * 40)
                print('\n* '.join(info['gt_program_lines'][np.argsort(curr_lcs)[-1]]))
                print('*' * 40)
                print()
            # iterate through all gt sketches and use the highest lcs obtained
            if 'gt_sketch_lines' in info:
                curr_lcs = []
                for gt_sketch_lines in info['gt_sketch_lines']:
                    gt_sketch_lines = preprocess_program_lines_for_lcs(gt_sketch_lines)
                    lcs = LCS(program_lines, gt_sketch_lines)
                    lcs_score = len(lcs) / (float(max(len(program_lines), len(gt_sketch_lines))))
                    curr_lcs.append(lcs_score)
                most_similar_gt_idx = np.argsort(curr_lcs)[-1]
                task_sketch_lcs[(task, desc, scene)] = curr_lcs[most_similar_gt_idx]
                most_similar_gt_sketch_text = info['gt_sketch_text'][most_similar_gt_idx]
                if verbose:
                    print('*' * 10 + f' {task} ' + '*' * 10)
                    print('*' * 5 + f' {curr_lcs} ' + '*' * 5)
                    print('\n* '.join(program_lines))
                    print('-' * 40)
                    print('\n* '.join(info['gt_sketch_lines'][np.argsort(curr_lcs)[-1]]))
                    print('*' * 40)
                    print()
        info['lcs_ep'] = task_lcs[(task, desc, scene)]
        info['sketch_lcs'] = task_sketch_lcs[(task, desc, scene)]
        info['most_similar_gt_program_text'] = most_similar_gt_program_text
        info['most_similar_gt_sketch_text'] = most_similar_gt_sketch_text
    avg_lcs = np.mean(list(task_lcs.values()))
    sketch_lcs_sum, count = 0, 0
    for v in task_sketch_lcs.values():
        if v != -1:
            sketch_lcs_sum += v
            count += 1
    print(f'** calculating sketch lcs across {count}/{len(task_lcs)} examples')
    if sketch_lcs_sum == 0 and count == 0:
        avg_sketch_lcs = -1
    else:
        avg_sketch_lcs = sketch_lcs_sum / count
    return avg_lcs, avg_sketch_lcs

def construct_generation_dict(args, evaluated_scenes):
    """init info dict to save relavent infos"""
    #pdb.set_trace()
    sketch_dict = load_dict(SKETCH_PATH)
    generation_info = dict()
    # iterate through all test programs and save the ground truth for later evaluation
    for test_path in args.test_paths:
        task = load_txt(test_path).strip().split('\n')[0]

        
        for scene in evaluated_scenes:
            #pdb.set_trace()
            lines = load_txt(test_path).strip().split('\n')
            task = lines[0]
            if args.add_desc:
                desc = lines[1]
            else:
                desc = ''
            program_lines = lines[4:]
            program_text = '\n'.join(program_lines).strip()
            # init the dict for each program
            if (task, desc, scene) in generation_info:
                # if the same task has appeared before, keep the current one in a list of ground truth
                generation_info[(task, desc, scene)]['gt_program_text'].append(program_text)
                generation_info[(task, desc, scene)]['gt_program_lines'].append(program_lines)
            else:
                generation_info[(task, desc, scene)] = dict()
                generation_info[(task, desc, scene)]['gt_path'] = test_path
                generation_info[(task, desc, scene)]['gt_program_text'] = [program_text]
                generation_info[(task, desc, scene)]['gt_program_lines'] = [program_lines]
                # find the highest number to use as id, such that within the task, the id is unique
                num_existing = len([_desc for (_task, _desc, _scene) in generation_info if _task == task and _scene==scene])
                generation_info[(task, desc, scene)]['id'] = num_existing
                generation_info[(task, desc, scene)]['formatted_task_title'] = task.lower().strip().replace(' ', '_')
                generation_info[(task, desc, scene)]['base_save_name'] = '{}-{}-scene{}.txt'.format(generation_info[(task, desc, scene)]['formatted_task_title'], num_existing, scene)

                #final raw plan text
                generation_info[(task, desc, scene)]['raw_save_path'] = os.path.join(args.api_save_path, generation_info[(task, desc, scene)]['base_save_name'])

                #full raw plan text 
                generation_info[(task, desc, scene)]['full_save_path'] = os.path.join(args.full_save_path, generation_info[(task, desc, scene)]['base_save_name'])

                #full generated plan text (w/o translation)
                generation_info[(task, desc, scene)]['full_generated_save_path'] = os.path.join(args.full_generated_save_path, generation_info[(task, desc, scene)]['base_save_name'])

                # matched plan text (for raw plan)
                generation_info[(task, desc, scene)]['matched_save_path'] = os.path.join(args.matched_save_path, generation_info[(task, desc, scene)]['base_save_name'])

                # full matched plan text (for full raw plan)
                generation_info[(task, desc, scene)]['full_matched_save_path'] = os.path.join(args.full_matched_save_path, generation_info[(task, desc, scene)]['base_save_name'])

                # parsed plan text (for raw plan)
                generation_info[(task, desc, scene)]['parsed_save_path'] = os.path.join(args.parsed_save_path, generation_info[(task, desc, scene)]['base_save_name'])

                # parsed plan text (for full raw plan)
                generation_info[(task, desc, scene)]['full_parsed_save_path'] = os.path.join(args.full_parsed_save_path, generation_info[(task, desc, scene)]['base_save_name'])

                # full generated plan text (w/o translation)
                generation_info[(task, desc, scene)]['full_generated_parsed_save_path'] = os.path.join(args.full_generated_parsed_save_path, generation_info[(task, desc, scene)]['base_save_name'])

            # if the file has sketch annotation, store those too
            base_fname = '/'.join(test_path.split('/')[-2:])[:-4]
            if base_fname in sketch_dict:
                sketch_lines = sketch_dict[base_fname]
                sketch_text = '\n'.join(sketch_lines).strip()
                if 'gt_sketch_text' in generation_info[(task, desc, scene)]:
                    # if the same task has appeared before, keep the current one in a list of ground truth
                    generation_info[(task, desc, scene)]['gt_sketch_text'].append(sketch_text)
                    generation_info[(task, desc, scene)]['gt_sketch_lines'].append(sketch_lines)
                else:
                    generation_info[(task, desc, scene)]['gt_sketch_text'] = [sketch_text]
                    generation_info[(task, desc, scene)]['gt_sketch_lines'] = [sketch_lines]
    percent_w_annotation = sum(["gt_sketch_text" in info for info in generation_info.values()]) / len(generation_info)
    #pdb.set_trace()
    print(f'** percent of tasks having sketch annotation: {percent_w_annotation:.2f}')
    return generation_info



def transformers_engine(model_id, device, seed):
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    set_seed(seed)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except OSError as e:
        tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if 'gpt-j' in model_id else torch.float32, pad_token_id=tokenizer.eos_token_id).to(device)

    def _generator(kwargs):
        input_ids = tokenizer(kwargs['prompt'], return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[-1]
        #pdb.set_trace()
        output_dict = model.generate(input_ids,
                        do_sample=True,
                        max_length=prompt_len + kwargs['max_tokens'],
                        repetition_penalty=kwargs['presence_penalty'],
                        num_return_sequences=kwargs['n'],
                        top_p=kwargs['top_p'],
                        temperature=kwargs['temperature'],
                        use_cache=True,
                        output_scores=True,
                        return_dict_in_generate=True)
        return_dict = dict(choices=[dict() for _ in range(kwargs['n'])])
        # discard the prompt
        generated = tokenizer.batch_decode(output_dict.sequences[:, prompt_len:])
        # get logprob for all samples [n, length, vocab_size]
        log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(-1)
        log_probs = torch.gather(log_probs, 2, output_dict.sequences[:, prompt_len:, None]).squeeze(-1)
        # truncate the sequences if stop word occurs, and add to return dict
        for i, sequence in enumerate(generated):
            sequence = sequence.strip('\n')
            if kwargs['stop'] in sequence:
                stop_idx = sequence.index(kwargs['stop'])
            else:
                stop_idx = None
            return_dict['choices'][i]['text'] = sequence[:stop_idx]
            # truncate the log prob as well
            return_dict['choices'][i]['logprobs'] = dict(token_logprobs=log_probs[i, :stop_idx].detach().cpu().numpy())
        return return_dict

    def _generator_multi(kwargs):
        """n could be too large to run, so split the request into multiple ones"""
        if kwargs['n'] <= 10:
            return _generator(kwargs)
        else:
            remaining = kwargs['n']
            full_return_dict = None
            while remaining > 0:
                curr_kwargs = copy.deepcopy(kwargs)
                curr_kwargs['n'] = min(10, remaining)
                curr_return_dict = _generator(curr_kwargs)
                if full_return_dict is None:
                    full_return_dict = curr_return_dict
                else:
                    for elem in curr_return_dict['choices']:
                        full_return_dict['choices'].append(elem)
                remaining -= curr_kwargs['n']
        assert len(full_return_dict['choices']) == kwargs['n']
        return full_return_dict

    return _generator_multi

def main(args):
    # define lm used for generation
    try:
        args.engine = transformers_engine(args.engine, args.device, args.seed)
    except Exception as e:
        print(e.__class__.__name__, str(e))
        print('** Using OpenAI API')
        if not 'codex' and not 'code' in args.engine:
            assert args.allow_charges
    start = time.time()
    if args.skip_load and not args.use_similar_example:
        print('skipping loading sentence model for faster debugging... ')
        sentence_model = None
    else:
        print('loading sentence model... ', end='')
        sentence_model = SentenceTransformer(args.sentence_model).to(args.device)
        print('loaded! (time taken: {:.2f} s)'.format(time.time() - start))
    # first get all action embeddings
    action_list = load_dict(args.allowed_action_path)
    # load action embedding if exists; otherwise create it and cache it to disk
    start = time.time()
    if os.path.exists(args.action_embedding_path):
        print('loading action_embedding... ', end='')
        action_list_embedding = torch.load(args.action_embedding_path).to(args.device)
    else:
        print('creating action_embedding... ', end='')
        action_list_embedding = sentence_model.encode(action_list, batch_size=args.batch_size, convert_to_tensor=True, device=args.device)
        torch.save(action_list_embedding.detach().cpu(), args.action_embedding_path)
    print('done! (time taken: {:.2f} s)'.format(time.time() - start))
    # [if use_similar_example] load action embedding if exists; otherwise create it and cache it to disk
    start = time.time()
    if args.use_similar_example and os.path.exists(args.title_embedding_path):
        print('loading title embedding for "use_similar_example"... ', end='')
        title_embedding = torch.load(args.title_embedding_path)
        # if specified that only using a subset of examples, adjust corresponding title embeddings
        if args.use_example_subset:
            title_embedding = title_embedding[args.selected_example_idx]
        title_embedding = title_embedding.to(args.device)
        print('done! (time taken: {:.2f} s)'.format(time.time() - start))
    elif args.use_similar_example:
        print('creating title embedding for "use_similar_example"... ', end='')
        titles = []
        for example_path in args.example_paths:
            example = load_txt(example_path)
            program_lines = example.strip().split('\n')
            title = program_lines[0]
            titles.append(title)
        title_embedding = sentence_model.encode(titles, batch_size=args.batch_size, convert_to_tensor=True, device=args.device)
        # cache to device
        torch.save(title_embedding.detach().cpu(), args.title_embedding_path)
        print('done! (time taken: {:.2f} s)'.format(time.time() - start))
    else:
        title_embedding = None
    # generate vh proram ============================================
    evaluated_scenes = range(1, 8) if args.scene_num is None else [args.scene_num]
    generation_info = construct_generation_dict(args, evaluated_scenes)

    #{ 'parsed_program', 'executed', 'scene_path', 'script_path', 'init_graph_dict', 'modified_program', 'execution_error', 'precond_error', 'parsing_error', 'empty_program_error', 'total_steps'}
    #{'parsed_program','executed','scene_path', 'script_path','init_graph_dict','modified_program','execution_error','precond_error'}

    execution_results = generate_all_tasks(generation_info, sentence_model, title_embedding, action_list, action_list_embedding, args)
    #pdb.set_trace()
    parsed_program_paths = []
    for k in generation_info:
        parsed_program_paths.append(generation_info[k]['parsed_save_path'])

    #pdb.set_trace()
    # save graph and unity-modified scripts for visualization
    for r in execution_results:
        # pop init_graph_dict from execution_results and save separately for visualization, and such that it's not uploaded to wandb
        init_graph_dict = r.pop('init_graph_dict')
        if r['executed']:
            assert init_graph_dict is not None

            title = os.path.basename(r['script_path'])[:-6]
            save_dict(os.path.join(args.init_graph_save_path, '{}.json'.format(title)), init_graph_dict)
            #pdb.set_trace()
            # save modified scripts for visualization
            save_txt(os.path.join(args.unity_parsed_save_path, '{}.txt'.format(title)), r['modified_program'])

    #pdb.set_trace()
    # log generation info
    generation_info = update_info_with_execution(generation_info, execution_results)

    
    # log to wandb ========================================================
    # log executability
    executability = sum([r['executed'] for r in execution_results]) / len(execution_results)
    wandb.run.summary["executability"] = executability
    print('** executability: {:.2f}'.format(executability))

    #log closed loop percent executed
    avg_percent_executed_inloop = sum([r['percent_executed_inloop'] for r in execution_results])/len(execution_results)
    wandb.run.summary["avg_percent_executed_inloop"] = avg_percent_executed_inloop
    print('** average percent executed (in-loop): {:.2f}'.format(avg_percent_executed_inloop))

    #log percent executed on final plan
    avg_percent_executed = sum([r['percent_executed'] for r in execution_results])/len(execution_results)
    wandb.run.summary["avg_percent_executed"] = avg_percent_executed
    print('** average percent executed (final plan): {:.2f}'.format(avg_percent_executed_inloop))

    #log number of corrective steps
    avg_num_replans = sum(r['num_replans'] for r in execution_results) / len(execution_results)
    wandb.run.summary["num_replans"] = avg_num_replans
    print('** average num replans: {:.2f}'.format(avg_num_replans))

    # evaluate lcs score
    avg_lcs, avg_sketch_lcs = evaluate_lcs_score(generation_info, verbose=False)
    wandb.run.summary["avg_lcs"] = avg_lcs
    print('** avg_lcs: {:.2f}'.format(avg_lcs))
    wandb.run.summary["avg_sketch_lcs"] = avg_sketch_lcs
    print('** avg_sketch_lcs: {:.2f}'.format(avg_sketch_lcs))

    wandb.run.summary["avg_lcs_ep"] = avg_lcs
    print('** avg_lcs_ep: {:.2f}'.format(avg_lcs))

    #evaluate the n-step similarity score (with clipping)
    n_step_similarity = evaluate_n_step_similarity(generation_info, n=3)

    wandb.run.summary["n_step_similarity"] = n_step_similarity
    print('** avg n_step_similarity: {:.2f}'.format(n_step_similarity))

    #evaluate the pairwise precision score (with clipping)
    pairwise_precision = evaluate_pairwise_precision(generation_info)

    wandb.run.summary["pairwise_precision"] = pairwise_precision
    print('** avg pairwise precision: {:.2f}'.format(pairwise_precision))

    # get average program lengths
    avg_parsed_length = get_avg_program_length(parsed_program_paths)
    wandb.run.summary['avg_no_steps'] = avg_parsed_length
    print('** avg_no_steps: {:.2f}'.format(avg_parsed_length))

    # get average parsibility
    avg_parsibility = np.mean([info['parsibility'] for info in generation_info.values()])
    wandb.run.summary['avg_parsibility'] = avg_parsibility
    print('** avg_parsibility: {:.2f}'.format(avg_parsibility))

    # get normalized overall score for hparam sweep ranking
    normalized_exec = normalize(executability, min_v=.09, max_v=.88)
    normalized_lcs = normalize(avg_lcs, min_v=.10, max_v=.24)
    overall_score = normalized_exec + normalized_lcs
    wandb.run.summary['normalized_exec'] = normalized_exec
    wandb.run.summary['normalized_lcs'] = normalized_lcs
    wandb.run.summary['overall_score'] = overall_score
    print('** normalized_exec: {:.2f}'.format(normalized_exec))
    print('** normalized_lcs: {:.2f}'.format(normalized_lcs))
    print('** overall_score: {:.2f}'.format(overall_score))


    #pdb.set_trace() 
    summary_keys = ['task', 'description', 'scene', 'example_text', 'final_raw_text', 'full_raw_text', 'all_errors', 'matched_text', 'full_matched_text', 'full_generated_text', 'parsibility', 'executed', 'percent_executed','percent_executed_inloop', 'lcs_ep', 'n_step_similarity', 'pairwise_precision','most_similar_gt_program_text', 'execution_error', 'precond_error', 'parsing_error','empty_program_error', 'final_steps', 'total_steps', 'num_replans', 'parsed_text','full_parsed_text', 'sketch_lcs', 'most_similar_gt_sketch_text','no_gen_error','score_error']
    table_data = []
    for (task, desc, scene), info in generation_info.items():
        data_list = [task, desc, scene]
        for k in summary_keys[3:]:
            if k not in info:
                data_list.append('')
                continue
            curr_value = copy.deepcopy(info[k])
            if isinstance(curr_value, list):
                for idx, e in enumerate(curr_value):
                    if e is None:
                        curr_value[idx] = 'None'
            if k == 'executed':
                curr_value = np.mean(curr_value)
            if 'text' in k:
                if isinstance(curr_value, list):
                    curr_value = [e.replace('\n', ', ') for e in curr_value]
                else:
                    curr_value = curr_value.replace('\n', ', ')
            data_list.append(curr_value)
        table_data.append(data_list)

    #pdb.set_trace()
    # construct table and log to wandb
    table = wandb.Table(data=table_data, columns=summary_keys)
    wandb.run.summary["execution_infos"] = table

    wandb.log({
        'avg_lcs_ep': avg_lcs,
        'avg_sketch_lcs': avg_sketch_lcs,
        'avg_no_steps': avg_parsed_length,
        'avg_num_replans': avg_num_replans,
        'avg_parsibility': avg_parsibility,
        'avg_executability': executability,
        'avg_percent_executed': avg_percent_executed,
        'avg_percent_executed_inloop': avg_percent_executed_inloop,
        'avg_n_step_similarity': n_step_similarity,
        'avg_pairwise_precision': pairwise_precision,
        'execution_infos': table,
        'normalized_exec': normalized_exec,
        'normalized_lcs': normalized_lcs,
        'overall_score': overall_score
    })

def update_info_with_execution(generation_info, execution_results):

    '''
    execution_results: list of dictionaries with keys ==> { 'parsed_program', 'executed', 'scene_path', 'script_path', 'init_graph_dict', 'modified_program', 'execution_error', 'precond_error', 'parsing_error', 'empty_program_error', 'total_steps'}

    '''

    # aggregate execution_results by parsed script path
    script2results = dict()
    for r in execution_results:
        scene_num = int(parse.parse(args.scene_path_format, r['scene_path'])[0])
        if r['script_path'] not in script2results:
            script2results[r['script_path']] = dict()
        assert scene_num not in script2results[r['script_path']]
        script2results[r['script_path']][scene_num] = dict(executed=r['executed'], percent_executed = r['percent_executed'], percent_executed_inloop = r['percent_executed_inloop'], execution_error=r['execution_error'], precond_error=r['precond_error'], parsing_error=r['parsing_error'], empty_program_error=r['empty_program_error'], total_steps=r['total_steps'], final_steps = r['final_steps'], no_gen_error = r['no_gen_error'], score_error = r['score_error'], all_errors = r['all_errors'] )

    for (task, desc, scene), info in generation_info.items():
        for script_path, script_results in script2results.items():
            if info['parsed_save_path'] == script_path:
                info['scene_nums'] = [scene_num for scene_num in script_results.keys()]
                info['executed'] = [scene_result['executed'] for scene_result in script_results.values()]

                info['percent_executed'] = [scene_result['percent_executed'] for scene_result in script_results.values()]
                info['percent_executed_inloop'] = [scene_result['percent_executed_inloop'] for scene_result in script_results.values()]

                # log the error information for execution across scenes
                info['execution_error'] = [scene_result['execution_error'] for scene_result in script_results.values()]
                info['precond_error'] = [scene_result['precond_error'] for scene_result in script_results.values()]
                info['parsing_error'] = [scene_result['parsing_error'] for scene_result in script_results.values()]
                info['empty_program_error'] = [scene_result['empty_program_error'] for scene_result in script_results.values()]
                info['no_gen_error'] = [scene_result['no_gen_error'] for scene_result in script_results.values()]
                info['score_error'] = [scene_result['score_error'] for scene_result in script_results.values()]
                info['all_errors'] = [scene_result['all_errors'] for scene_result in script_results.values()]

                # log the step information for execution across scenes
                info['total_steps'] = [scene_result['total_steps'] for scene_result in script_results.values()]
                info['final_steps'] = [scene_result['final_steps'] for scene_result in script_results.values()]
                

    return generation_info


if __name__ == '__main__':
    # env setting ========================================================================
    # always raise numpy error
    # np.seterr(all='warn')
    # do not enable wandb output
    os.environ["WANDB_SILENT"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    pdb.set_trace()
    args = get_args()
    wandb.config.update(args, allow_val_change=True)
    main(args)
