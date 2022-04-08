import json
import sys
import os

sys.path.append('../simulation')
from evolving_graph.scripts import *
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
from arguments import get_args
from sentence_transformers import SentenceTransformer
from generation_utils import *
import time
import torch
from tqdm import tqdm
import glob

# multiprocessing version
def evaluate_script(kwargs):
    script_path, scene_path = kwargs['script_path'], kwargs['scene_path']
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
    else:
        verbose = False
    assert '.txt' in script_path
    script_fname = os.path.basename(script_path)
    title = script_fname[:-4]

    # load script and inferred preconds
    try:
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
    program_lines = script_to_list_string(script)

    # define returned metrics, such that all scripts would have all the keys present
    info = {'parsed_program': '\n'.join(program_lines).strip(),
            'executed': None,
            'scene_path': scene_path,
            'script_path': script_path,
            'init_graph_dict': None,
            'modified_program': None,
            'execution_error': None,
            'precond_error': None,
            }

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

    try:
        precond = get_preconds_script(program_lines, verbose=verbose).printCondsJSON()
    except ScriptFail as e:
        info['precond_error'] = 'ScriptFail: {}'.format(e.message)
        info['executed'] = False
        if verbose:
            print(info['precond_error'])
            print('[{}] is NOT executable\n'.format(script_fname))
        return info

    env_graph = utils.load_graph(scene_path)
    # prepare env graph from precond and script
    graph_dict = env_graph.to_dict()

    if verbose:
        print('PRECOND: {}'.format(precond))

    try:
        (message, init_graph_dict, final_state, graph_state_list, input_graph, 
                                id_mapping, _, graph_helper, modified_script) = check_script(
                                        program_lines, 
                                        precond, 
                                        scene_path,
                                        inp_graph_dict=graph_dict,
                                        modify_graph=True,
                                        id_mapping={},
                                        info={})
        info['init_graph_dict'] = init_graph_dict
        info['modified_program'] = modified_script.to_string()
    except Exception as e:
        message = "{}: {}".format(e.__class__.__name__, e)
        print('** check_script FAILED: ' + message)

    if verbose:
        print(message)
    
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

def evaluate_all_scripts(script_paths, args, evaluated_scenes=range(1, 8)):
    """find all scripts in script_dir and evaluate in all given scenes"""
    # construct args for all 7 scenes
    scene_paths = []
    for scene_num in evaluated_scenes:
        scene_paths += [args.scene_path_format.format(scene_num)] * len(script_paths)
    _script_paths = script_paths * len(evaluated_scenes)
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

def generate_program(query_task_desc, example_path, sentence_model, action_list, action_list_embedding, generation_info, args):
    query_task, query_desc = query_task_desc
    # determine saving file name
    info = generation_info[(query_task, query_desc)]
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
        full_raw_text, matched_program_lines = iterative_api_request(example_str, task_prompt_formatted, args.api_params,
                                                                    sentence_model, action_list_embedding, args.device,
                                                                    action_list, max_iters=1000, max_steps=args.api_max_steps,
                                                                    verbose=args.debug and args.verbose, cutoff_threshold=args.api_cutoff_threshold,
                                                                    beta=args.api_beta, percent_terminate=args.api_percent_terminate,
                                                                    engine=args.engine, translated_condition=args.translated_condition)
    else:
        full_raw_text, matched_program_lines = one_shot_api_request(example_str, task_prompt_formatted, args.api_params,
                                                                    sentence_model, action_list_embedding, args.device,
                                                                    action_list, max_iters=1000,
                                                                    beta=args.api_beta, raw_lm=args.raw_lm,
                                                                    engine=args.engine)
    if args.verbose:
        print('*************** RAW TEXT ***************\n{}'.format(full_raw_text))
    # save the raw output
    save_txt(info['raw_save_path'], full_raw_text)
    if args.raw_lm:
        parsed_program_lines, parse_info = str2program_list(full_raw_text.split('\n')[1:])
        parsed_program_text = '\n'.join(parsed_program_lines).strip()
        if args.verbose:
            print('*************** PARSED TEXT ***************\n{}'.format(parsed_program_text))
        save_txt(info['parsed_save_path'], parsed_program_text)
        # save generation info
        generation_info[(query_task, query_desc)]['example_text'] = example_str
        generation_info[(query_task, query_desc)]['full_raw_text'] = full_raw_text
        generation_info[(query_task, query_desc)]['parsed_text'] = parsed_program_text
        generation_info[(query_task, query_desc)]['parsed_program_lines'] = parsed_program_lines
        generation_info[(query_task, query_desc)]['parsibility'] = parse_info['parsibility']
    else:
        # convert matched program to str and save to disk
        matched_program_text = '\n'.join(matched_program_lines).strip()
        if args.verbose:
            print('*************** MATCHED TEXT ***************\n{}'.format(matched_program_text))
        save_txt(info['matched_save_path'], matched_program_text)
        # parse matched actions into vh program ============================================
        parsed_program_lines, parse_info = str2program_list(matched_program_lines)
        # remove consecutive actions
        parsed_program_lines = remove_same_consecutive(parsed_program_lines)
        parsed_program_text = '\n'.join(parsed_program_lines).strip()
        if args.verbose:
            print('*************** PARSED TEXT ***************\n{}'.format(parsed_program_text))
        save_txt(info['parsed_save_path'], parsed_program_text)
        # save generation info
        generation_info[(query_task, query_desc)]['example_text'] = example_str
        generation_info[(query_task, query_desc)]['full_raw_text'] = full_raw_text
        generation_info[(query_task, query_desc)]['matched_text'] = matched_program_text
        generation_info[(query_task, query_desc)]['parsed_text'] = parsed_program_text
        generation_info[(query_task, query_desc)]['parsed_program_lines'] = parsed_program_lines
        generation_info[(query_task, query_desc)]['parsibility'] = parse_info['parsibility']
    

def evaluate_lcs_score(generation_info, verbose=False):
    # evaluate lcs score for each task
    task_lcs = dict()
    task_sketch_lcs = dict()
    for (task, desc), info in generation_info.items():
        try:
            program_lines = info['parsed_program_lines']
        except KeyError as e:
            program_lines = load_txt(info['parsed_save_path']).split('\n')
        # init default values
        most_similar_gt_program_text = info['gt_program_text'][0]
        most_similar_gt_sketch_text = ''
        task_sketch_lcs[(task, desc)] = -1
        # if the program is empty, simply assign lcs of 0
        if len(program_lines) == 0 or (len(program_lines) == 1 and len(program_lines[0]) == 0):
            task_lcs[(task, desc)] = 0
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
            assert (task, desc) not in task_lcs
            most_similar_gt_idx = np.argsort(curr_lcs)[-1]
            task_lcs[(task, desc)] = curr_lcs[most_similar_gt_idx]
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
                task_sketch_lcs[(task, desc)] = curr_lcs[most_similar_gt_idx]
                most_similar_gt_sketch_text = info['gt_sketch_text'][most_similar_gt_idx]
                if verbose:
                    print('*' * 10 + f' {task} ' + '*' * 10)
                    print('*' * 5 + f' {curr_lcs} ' + '*' * 5)
                    print('\n* '.join(program_lines))
                    print('-' * 40)
                    print('\n* '.join(info['gt_sketch_lines'][np.argsort(curr_lcs)[-1]]))
                    print('*' * 40)
                    print()
        info['lcs'] = task_lcs[(task, desc)]
        info['sketch_lcs'] = task_sketch_lcs[(task, desc)]
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

def construct_generation_dict(args):
    """init info dict to save relavent infos"""
    sketch_dict = load_dict(SKETCH_PATH)
    generation_info = dict()
    # iterate through all test programs and save the ground truth for later evaluation
    for test_path in args.test_paths:
        lines = load_txt(test_path).strip().split('\n')
        task = lines[0]
        if args.add_desc:
            desc = lines[1]
        else:
            desc = ''
        program_lines = lines[4:]
        program_text = '\n'.join(program_lines).strip()
        # init the dict for each program
        if (task, desc) in generation_info:
            # if the same task has appeared before, keep the current one in a list of ground truth
            generation_info[(task, desc)]['gt_program_text'].append(program_text)
            generation_info[(task, desc)]['gt_program_lines'].append(program_lines)
        else:
            generation_info[(task, desc)] = dict()
            generation_info[(task, desc)]['gt_path'] = test_path
            generation_info[(task, desc)]['gt_program_text'] = [program_text]
            generation_info[(task, desc)]['gt_program_lines'] = [program_lines]
            # find the highest number to use as id, such that within the task, the id is unique
            num_existing = len([_desc for (_task, _desc) in generation_info if _task == task])
            generation_info[(task, desc)]['id'] = num_existing
            generation_info[(task, desc)]['formatted_task_title'] = task.lower().strip().replace(' ', '_')
            generation_info[(task, desc)]['base_save_name'] = '{}-{}.txt'.format(generation_info[(task, desc)]['formatted_task_title'], num_existing)
            generation_info[(task, desc)]['raw_save_path'] = os.path.join(args.api_save_path, generation_info[(task, desc)]['base_save_name'])
            generation_info[(task, desc)]['matched_save_path'] = os.path.join(args.matched_save_path, generation_info[(task, desc)]['base_save_name'])
            generation_info[(task, desc)]['parsed_save_path'] = os.path.join(args.parsed_save_path, generation_info[(task, desc)]['base_save_name'])
        # if the file has sketch annotation, store those too
        base_fname = '/'.join(test_path.split('/')[-2:])[:-4]
        if base_fname in sketch_dict:
            sketch_lines = sketch_dict[base_fname]
            sketch_text = '\n'.join(sketch_lines).strip()
            if 'gt_sketch_text' in generation_info[(task, desc)]:
                # if the same task has appeared before, keep the current one in a list of ground truth
                generation_info[(task, desc)]['gt_sketch_text'].append(sketch_text)
                generation_info[(task, desc)]['gt_sketch_lines'].append(sketch_lines)
            else:
                generation_info[(task, desc)]['gt_sketch_text'] = [sketch_text]
                generation_info[(task, desc)]['gt_sketch_lines'] = [sketch_lines]
    percent_w_annotation = sum(["gt_sketch_text" in info for info in generation_info.values()]) / len(generation_info)
    print(f'** percent of tasks having sketch annotation: {percent_w_annotation:.2f}')
    return generation_info

def generate_all_tasks(generation_info, sentence_model, title_embedding, action_list, action_list_embedding, args):
    bar = tqdm(total=len(generation_info))
    for query_task, query_desc in generation_info:
        if args.use_similar_example:
            example_path_idx = select_most_similar_example_idx(sentence_model=sentence_model, query_title=query_task, title_embedding=title_embedding, device=args.device)
            example_path = args.example_paths[example_path_idx]
        else:
            example_path = args.example_path
        generation_info[(query_task, query_desc)]['example_path'] = example_path
        # only generate program if not already exists
        parsed_save_path = generation_info[(query_task, query_desc)]['parsed_save_path']
        if not os.path.exists(parsed_save_path) or args.debug or args.fresh:
            generate_program((query_task, query_desc), example_path, sentence_model, action_list, action_list_embedding, generation_info, args)
        bar.update(1)

def transformers_engine(model_id, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    except OSError as e:
        tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16 if 'gpt-j' in model_id else torch.float32, pad_token_id=tokenizer.eos_token_id).to(device)
    
    def _generator(kwargs):
        input_ids = tokenizer(kwargs['prompt'], return_tensors="pt").input_ids.to(device)
        prompt_len = input_ids.shape[-1]
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
        args.engine = transformers_engine(args.engine, args.device)
    except Exception as e:
        print(e.__class__.__name__, str(e))
        print('** Using OpenAI API')
        if not 'codex' in args.engine:
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
    generation_info = construct_generation_dict(args)
    generate_all_tasks(generation_info, sentence_model, title_embedding, action_list, action_list_embedding, args)
    # evaluate in the vh simulator ============================================
    parsed_program_paths = []
    for k in generation_info:
        parsed_program_paths.append(generation_info[k]['parsed_save_path'])
    evaluated_scenes = range(1, 8) if args.scene_num is None else [args.scene_num]
    execution_results = evaluate_all_scripts(parsed_program_paths, args, evaluated_scenes=evaluated_scenes)
    # save graph and unity-modified scripts for visualization
    for r in execution_results:
        # pop init_graph_dict from execution_results and save separately for visualization, and such that it's not uploaded to wandb
        init_graph_dict = r.pop('init_graph_dict')
        if r['executed']:
            assert init_graph_dict is not None
            scene_num = int(parse.parse(args.scene_path_format, r['scene_path'])[0])
            title = os.path.basename(r['script_path'])[:-4]
            save_dict(os.path.join(args.init_graph_save_path, 'scene{}-{}.json'.format(scene_num, title)), init_graph_dict)
            # save modified scripts for visualization
            save_txt(os.path.join(args.unity_parsed_save_path, 'scene{}-{}.txt'.format(scene_num, title)), r['modified_program'])
    # log to wandb ========================================================
    # log executability
    percent_executed = sum([r['executed'] for r in execution_results]) / len(execution_results)
    wandb.run.summary["percent_executed"] = percent_executed
    print('** percent_executed: {:.2f}'.format(percent_executed))
    # evaluate lcs score
    avg_lcs, avg_sketch_lcs = evaluate_lcs_score(generation_info, verbose=False)
    wandb.run.summary["avg_lcs"] = avg_lcs
    print('** avg_lcs: {:.2f}'.format(avg_lcs))
    wandb.run.summary["avg_sketch_lcs"] = avg_sketch_lcs
    print('** avg_sketch_lcs: {:.2f}'.format(avg_sketch_lcs))
    # get average program lengths
    avg_parsed_length = get_avg_program_length(parsed_program_paths)
    wandb.run.summary['avg_parsed_length'] = avg_parsed_length
    print('** avg_parsed_length: {:.2f}'.format(avg_parsed_length))
    # get average parsibility
    avg_parsibility = np.mean([info['parsibility'] for info in generation_info.values()])
    wandb.run.summary['avg_parsibility'] = avg_parsibility
    print('** avg_parsibility: {:.2f}'.format(avg_parsibility))
    # get normalized overall score for hparam sweep ranking
    normalized_exec = normalize(percent_executed, min_v=.09, max_v=.88)
    normalized_lcs = normalize(avg_lcs, min_v=.10, max_v=.24)
    overall_score = normalized_exec + normalized_lcs
    wandb.run.summary['normalized_exec'] = normalized_exec
    wandb.run.summary['normalized_lcs'] = normalized_lcs
    wandb.run.summary['overall_score'] = overall_score
    print('** normalized_exec: {:.2f}'.format(normalized_exec))
    print('** normalized_lcs: {:.2f}'.format(normalized_lcs))
    print('** overall_score: {:.2f}'.format(overall_score))

    # log generation info
    generation_info = update_info_with_execution(generation_info, execution_results)

    summary_keys = ['task', 'description', 'full_raw_text', 'matched_text', 'example_text', 'parsibility', 'executed', 'lcs', 'most_similar_gt_program_text', 'execution_error', 'parsed_text', 'precond_error', 'sketch_lcs', 'most_similar_gt_sketch_text']
    table_data = []
    for (task, desc), info in generation_info.items():
        data_list = [task, desc]
        for k in summary_keys[2:]:
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

    # construct table and log to wandb
    table = wandb.Table(data=table_data, columns=summary_keys)
    wandb.run.summary["execution_infos"] = table

    wandb.log({
        'avg_lcs': avg_lcs,
        'avg_sketch_lcs': avg_sketch_lcs,
        'avg_parsed_length': avg_parsed_length,
        'avg_parsibility': avg_parsibility,
        'percent_executed': percent_executed,
        'execution_infos': table,
        'normalized_exec': normalized_exec,
        'normalized_lcs': normalized_lcs,
        'overall_score': overall_score
    })

def update_info_with_execution(generation_info, execution_results):
    # aggregate execution_results by parsed script path
    script2results = dict()
    for r in execution_results:
        scene_num = int(parse.parse(args.scene_path_format, r['scene_path'])[0])
        if r['script_path'] not in script2results:
            script2results[r['script_path']] = dict()
        assert scene_num not in script2results[r['script_path']]
        script2results[r['script_path']][scene_num] = dict(executed=r['executed'],
                                                            execution_error=r['execution_error'],
                                                            precond_error=r['precond_error'])

    for (task, desc), info in generation_info.items():
        for script_path, script_results in script2results.items():
            if info['parsed_save_path'] == script_path:
                info['scene_nums'] = [scene_num for scene_num in script_results.keys()]
                info['executed'] = [scene_result['executed'] for scene_result in script_results.values()]
                info['execution_error'] = [scene_result['execution_error'] for scene_result in script_results.values()]
                info['precond_error'] = [scene_result['precond_error'] for scene_result in script_results.values()]
    
    return generation_info


if __name__ == '__main__':
    # env setting ========================================================================
    # always raise numpy error
    # np.seterr(all='warn')
    # do not enable wandb output
    os.environ["WANDB_SILENT"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = get_args()
    wandb.config.update(args, allow_val_change=True)
    main(args)
