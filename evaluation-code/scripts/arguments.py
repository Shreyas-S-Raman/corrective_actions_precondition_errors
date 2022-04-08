import argparse
import glob
import os
from vh_configs import *
import torch
import numpy as np
import wandb
from generation_utils import load_txt

def get_args():

    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--debug', type=str2bool, default=False)
    parser.add_argument('--skip_load', type=str2bool, default=False, help='skip loading sentence model for faster debugging')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--fresh', type=str2bool, default=False)
    parser.add_argument('--expID', type=int, default=1)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--scene_num', type=int, default=None)
    # parser.add_argument('--mode', type=str, default='none')
    parser.add_argument('--use_similar_example', type=str2bool, default=False)
    parser.add_argument('--sentence_model', type=str, default='stsb-roberta-large', help='default for best quality; ')
    parser.add_argument('--query_task', type=str, default=None)
    parser.add_argument('--example_path', type=str, default=None)
    parser.add_argument('--example_id', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=10000, help='used to do semantic matching for sentence model')
    # openai api params
    parser.add_argument('--api_max_tokens', type=int, default=8)
    parser.add_argument('--api_temperature', type=float, default=0.6)
    parser.add_argument('--api_top_p', type=float, default=0.85)
    parser.add_argument('--api_n', type=int, default=1)
    parser.add_argument('--api_logprobs', type=float, default=1)
    parser.add_argument('--api_echo', type=str2bool, default=False)
    parser.add_argument('--api_presence_penalty', type=float, default=0.2)
    parser.add_argument('--api_frequency_penalty', type=float, default=0.2)
    parser.add_argument('--api_best_of', type=int, default=1)
    # other codex generation params
    parser.add_argument('--api_max_steps', type=int, default=20)
    parser.add_argument('--use_cutoff_threshold', type=str2bool, default=True)
    parser.add_argument('--api_cutoff_threshold', type=float, default=-100)
    parser.add_argument('--api_beta', type=float, default=0.2)
    parser.add_argument('--api_percent_terminate', type=float, default=0.2)
    # other
    parser.add_argument('--add_desc', type=str2bool, default=False)
    parser.add_argument('--iterative', type=str2bool, default=True)
    parser.add_argument('--raw_lm', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--use_example_subset', type=str2bool, default=False)
    parser.add_argument('--num_available_examples', type=int, default=-1, help='restrict the number of available example when user uses use_similar_example; -1 means no restriction imposed')
    parser.add_argument('--translated_condition', type=str2bool, default=False)
    parser.add_argument('--engine', type=str, default='davinci-codex')
    parser.add_argument('--allow_charges', type=str2bool, default=False, help='allow using non-codex models from openai api')
    parser.add_argument('--finetuned', type=str2bool, default=False)
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)

    if args.finetuned:
        project_name = 'inst-decomp-finetune'
        exp_prefix = 'vh-finetune'
    else:
        project_name = 'inst-decomp'
        exp_prefix = 'vh-zero'

    if args.sweep:
        wandb.init(project=project_name)
        args.exp_name = wandb.run.name
        if args.exp_name is None:
            args.exp_name = 'sweep-' + wandb.run.id
    else:
        resume = not args.debug and not args.fresh
        if ':ft-' in args.engine:
            engine_name = 'finetuned-' + args.engine[:args.engine.index(':')]
        else:
            engine_name = args.engine
        args.exp_name = '{}-{}-{:04d}'.format(exp_prefix, engine_name, args.expID) if args.exp_name is None else args.exp_name
        wandb.init(project=project_name,
                name=args.exp_name, id=args.exp_name, resume=resume, save_code=True)
        if wandb.run.resumed:
            print(f'*** resuming run {args.exp_name}')
    
    if args.sweep:
        args.fresh = True
    
    args.api_cutoff_threshold = args.api_cutoff_threshold if args.use_cutoff_threshold else -100
    args.num_available_examples = args.num_available_examples if (args.use_similar_example and args.use_example_subset) else -1
    
    
    args.save_dir = '../generated_programs'
    args.exp_path = os.path.join(args.save_dir, args.exp_name)
    args.api_save_path = os.path.join(args.exp_path, 'raw')
    args.matched_save_path = os.path.join(args.exp_path, 'matched')
    args.parsed_save_path = os.path.join(args.exp_path, 'parsed')
    args.init_graph_save_path = os.path.join(args.exp_path, 'init_graphs')
    args.unity_parsed_save_path = os.path.join(args.exp_path, 'unity_parsed')
    if os.path.exists(args.exp_path) and args.sweep:
        print(f'** removing previously existed sweep dir [{args.exp_path}]')
        os.system(f'rm -rf {args.exp_path}')
    os.makedirs(args.api_save_path, exist_ok=True)
    os.makedirs(args.matched_save_path, exist_ok=True)
    os.makedirs(args.parsed_save_path, exist_ok=True)
    os.makedirs(args.init_graph_save_path, exist_ok=True)
    os.makedirs(args.unity_parsed_save_path, exist_ok=True)
    args.action_embedding_path = os.path.join(args.save_dir, '{}_action_embedding.pt'.format(args.sentence_model))
    args.title_embedding_path = os.path.join(args.save_dir, '{}_train_title_embedding.pt'.format(args.sentence_model))
    

    if args.example_path is None and args.example_id is None:
        assert args.use_similar_example or args.finetuned
    
    args.allowed_action_path = os.path.join(RESOURCE_DIR, 'allowed_actions.json')
    args.name_equivalence_path = os.path.join(RESOURCE_DIR, 'class_name_equivalence.json')
    args.scene_path_format = os.path.join(SCENE_DIR, 'TrimmedTestScene{}_graph.json')

    args.test_paths = load_txt(os.path.join(RESOURCE_DIR, 'test_task_paths.txt')).strip().split('\n')
    args.test_paths = sorted([os.path.join(DATASET_DIR, path) for path in args.test_paths])
    args.train_paths = load_txt(os.path.join(RESOURCE_DIR, 'train_task_paths.txt')).strip().split('\n')
    args.train_paths = sorted([os.path.join(DATASET_DIR, path) for path in args.train_paths])
    if args.scene_num is not None:
        # retrieve examples from specified scene
        args.example_paths = [path for path in args.train_paths if 'TrimmedTestScene{}_graph'.format(args.scene_num) in path]
    else:
        # allowed to use examples from all scenes
        args.example_paths = args.train_paths
    
    # only select a subset of examples if specified
    if args.use_similar_example and args.use_example_subset:
        args.selected_example_idx = np.random.choice(range(len(args.example_paths)), size=args.num_available_examples, replace=False)
        args.example_paths = [args.example_paths[i] for i in args.selected_example_idx]

    if args.example_id is not None and not args.use_similar_example:
        print(f'taking example_id {args.example_id} % {len(args.example_paths)} = {args.example_id % len(args.example_paths)}')
        args.example_path = args.example_paths[args.example_id % len(args.example_paths)]

    args.query_task = args.query_task.lower()
    if args.query_task == 'all':
        args.test_paths = args.test_paths
        if args.debug:
            args.test_paths = np.random.choice(args.test_paths, size=10)
    else:
        args.query_task = args.query_task[0].upper() + args.query_task[1:]
        raise NotImplementedError
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.config.update(args, allow_val_change=True)

    args.api_params = \
        {
            "max_tokens": args.api_max_tokens,
            "temperature": args.api_temperature,
            "top_p": args.api_top_p,
            "n": args.api_n,
            "logprobs": args.api_logprobs,
            "echo": args.api_echo,
            "presence_penalty": args.api_presence_penalty,
            "frequency_penalty": args.api_frequency_penalty,
            # "best_of": args.api_best_of,
            "stop": '\n'
        }



    return args