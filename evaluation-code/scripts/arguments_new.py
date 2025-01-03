import argparse
import glob
import os
from vh_configs import *
from prompt_config import CausalErrors
import torch
import numpy as np
import wandb
from generation_utils import load_txt
import pdb


class Arguments:


    use_robot = True
    robot_hostname='138.16.161.22' #NOTE: replace with whichever hotname required

    '''default directories storing datasets'''
    RESOURCE_DIR = os.path.relpath('../resources') if not use_robot else os.path.relpath('../resources_robot') #contains allowed actions, class name equivalences, train and test tasks (txt files)
    
    SCENE_DIR = os.path.relpath('../example_graphs') if not use_robot else os.path.relpath('../../../spot_sdk_interface/nlmap_spot/scene_graphs')
    
    DATASET_DIR = os.path.relpath('../dataset/programs_processed_precond_nograb_morepreconds') if not use_robot else os.path.relpath('../resources_robot/dataset')

    '''experiment configs'''
    debug = False #original : False
    skip_load = False #skip loading sentence model for faster debugging
    verbose = False
    fresh = True #start new experiment?

    #both used to generate save path for experiment results e.g. init graph, unity output, parsed string, matched string
    expID = 1035
    exp_name = 'experiment_{}'.format(expID)
    num_workers = 40 #original: 40
    scene_num = None #take example train paths/tasks from specific VH scene [if None: uses all train paths/tasks (without scene restriction)]

    '''LLM configs'''
    use_similar_example = True
    sentence_model = 'stsb-roberta-large' #use 'stsb-roberta-large' for best quality 'roberta-base'
    query_task = 'all'

    example_path = None
    example_id = None #selecting specific example id [example_id % num tasks] from the set of train tasks/paths
    batch_size = 10000 #original: 10k; for semantic matching for sentence model [try smaller]


    '''OpenAI API configs'''
    api_max_tokens = 10
    
    api_temperature = 0.5 #from 0.5
    api_top_p = 0.9
    api_n = 10
    api_logprobs = 1 #was 5 for saycan baseline, 1 for all others
    api_echo = False #was True for SayCan baseline
    api_presence_penalty = 0.5 #0.3 best 
    api_frequency_penalty = 0.3 #original: 0.3
    #api_best_of = 1

    '''Codex generation params'''
    api_max_steps = 20
    use_cutoff_threshold = True
    api_cutoff_threshold = 0.45 #0.5 best (for score sum)
    api_beta = 0.3 #original: 0.3
    api_percent_terminate = 0.5


    '''Other configs'''
    add_desc = False #adds description for task in verbose output
    iterative = True #calls iterative api request  if True else calls one shot api request
    raw_lm = False #if True parses program text, if False matches program text
    seed = 4562 #setting random seed
    use_example_subset = False
    num_available_examples = -1  #restrict the number of available example when user uses use_similar_example; -1 means no restriction imposed
    translated_condition = True
    engine = 'davinci-instruct-beta' #gpt2 (0.1B) gpt2-medium (0.4B) is free | to run with GPT-3 use 'davinci' | to run with Codex use 'davinci-codex' #davinci-instruct-beta
    allow_charges = True #allow non-codex models from openai api
    finetuned = False #using finetuned LLM (after pretraining)


    '''Re prompting configs'''
    online_planning = True
    
    prompt_template = 1
    custom_cause = True
    third_person = False
    error_information = 'cause_1'
    suggestion_no = 1

    chosen_causal_reprompts = {
        'internally_contained': CausalErrors.INTERNALLY_CONTAINED1,
        'unflipped_boolean_state': CausalErrors.UNFLIPPED_BOOL_STATE1,
        'hands_full': CausalErrors.HANDS_FULL1,
        'already_sitting': CausalErrors.ALREADY_SITTING1,
        'already_standing':CausalErrors.ALREADY_STANDING1,
        'no_path': CausalErrors.NO_PATH1,
        'door_closed':CausalErrors.DOOR_CLOSED1,
        'proximity':CausalErrors.PROXIMITY1,
        'not_find': CausalErrors.NOT_FIND1,
        'invalid_action':CausalErrors.INVALID_ACTION1,
        'max_occupancy':CausalErrors.MAX_OCCUPANCY1,
        'not_holding':CausalErrors.NOT_HOLDING1 if third_person else CausalErrors.NOT_HOLDING3,
        'not_holding_any':CausalErrors.NOT_HOLDING_ANY1,
        'not_facing':CausalErrors.NOT_FACING2,
        'missing_step':CausalErrors.MISSING_STEP1,
        'invalid_room':CausalErrors.INVALID_ROOM1,
        'already_holding': CausalErrors.ALREADY_HOLDING1
    }

    #context prior to reprompting string: full-history, task-history, step-history
    chosen_context = 'task-history'


    step_by_step = False #add 'Let's think step by step to prompt'
    add_executable_mask = False #mask that only allows executable actions (in current state) to be chosen
    one_error = True
    resampling = False #promting only by resampling (next most viable step)
    saycan = False
    param_tuning = False

    #either 'zero-shot' (for prompt template), 'few-shot' for few-shot examples, 'reasoning' for step by step reasoning
    learned_method = 'zero-shot'
    num_examples = 1
    api_generation_temperature = 0.7
    api_generation_presence_penalty = 0.6
    api_generation_frequency_penalty = 0.3
    api_generation_n = 1
    default_error = 'Task failed. A correct step would be to'


def get_args():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--sweep', action='store_true')

    parsed_args = parser.parse_args()
    args = Arguments()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.finetuned:
        project_name = 'inst-decomp-finetune'
        exp_prefix = 'vh-finetune'
    else:
        project_name = 'inst-decomp'
        exp_prefix = 'vh-zero'

    if parsed_args.sweep:
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

    if parsed_args.sweep:
        args.fresh = True

    args.api_cutoff_threshold = args.api_cutoff_threshold if args.use_cutoff_threshold else -100
    args.num_available_examples = args.num_available_examples if (args.use_similar_example and args.use_example_subset) else -1


    args.save_dir = '../generated_programs'
    args.exp_path = os.path.join(args.save_dir, args.exp_name)
    args.api_save_path = os.path.join(args.exp_path, 'raw')
    args.full_save_path = os.path.join(args.exp_path, 'full')
    args.matched_save_path = os.path.join(args.exp_path, 'matched')
    args.full_matched_save_path = os.path.join(args.exp_path, 'full_matched')
    args.parsed_save_path = os.path.join(args.exp_path, 'parsed')
    args.full_parsed_save_path = os.path.join(args.exp_path, 'full_parsed')
    args.init_graph_save_path = os.path.join(args.exp_path, 'init_graphs')
    args.unity_parsed_save_path = os.path.join(args.exp_path, 'unity_parsed')

    args.full_generated_save_path = os.path.join(args.exp_path, 'full_generated')
    args.full_generated_parsed_save_path = os.path.join(args.exp_path, 'full_generated_parsed')

    if os.path.exists(args.exp_path) and parsed_args.sweep:
        print(f'** removing previously existed sweep dir [{args.exp_path}]')
        os.system(f'rm -rf {args.exp_path}')
    os.makedirs(args.api_save_path, exist_ok=True)
    os.makedirs(args.matched_save_path, exist_ok=True)
    os.makedirs(args.parsed_save_path, exist_ok=True)

    if args.online_planning:
        os.makedirs(args.full_matched_save_path, exist_ok = True)
        os.makedirs(args.full_save_path, exist_ok=True)
        os.makedirs(args.full_parsed_save_path, exist_ok = True)
        os.makedirs(args.full_generated_save_path, exist_ok=True)
        os.makedirs(args.full_generated_parsed_save_path, exist_ok=True)

    os.makedirs(args.init_graph_save_path, exist_ok=True)
    os.makedirs(args.unity_parsed_save_path, exist_ok=True)


    action_embed_file = '{}_action_embedding.pt'.format(args.sentence_model) if not args.use_robot else '{}_action_embedding_robot.pt'.format(args.sentence_model)
    args.action_embedding_path = os.path.join(args.save_dir, action_embed_file)


    title_embed_file = '{}_train_title_embedding.pt'.format(args.sentence_model) if not args.use_robot else '{}_train_title_embedding_robot.pt'.format(args.sentence_model)
    args.title_embedding_path = os.path.join(args.save_dir, title_embed_file) if not args.param_tuning else os.path.join(args.save_dir, '{}_val_title_embedding.pt'.format(args.sentence_model))


    correction_embed_file = '{}_correction_embedding_full_plan.pt'.format(args.sentence_model) if not args.use_robot else '{}_correction_embedding_full_plan_robot.pt'.format(args.sentence_model)
    args.correction_embedding_path = os.path.join(args.save_dir, correction_embed_file)



    if args.example_path is None and args.example_id is None:
        assert args.use_similar_example or args.finetuned
    
    

    args.allowed_action_path = os.path.join(args.RESOURCE_DIR, 'allowed_actions.json')
    args.name_equivalence_path = os.path.join(args.RESOURCE_DIR, 'class_name_equivalence.json')
    args.scene_path_format = os.path.join(args.SCENE_DIR, 'TrimmedTestScene{}_graph.json') if not args.use_robot else os.path.join(args.SCENE_DIR, 'scene_{}.json')

    args.test_paths = load_txt(os.path.join(args.RESOURCE_DIR, 'test_task_paths.txt')).strip().split('\n') if not args.param_tuning else load_txt(os.path.join(args.RESOURCE_DIR, 'val_task_paths.txt')).strip().split('\n')
    args.test_paths = list(sorted([os.path.join(args.DATASET_DIR, path) for path in args.test_paths]))
    args.train_paths = load_txt(os.path.join(args.RESOURCE_DIR, 'train_task_paths.txt')).strip().split('\n') if not args.param_tuning else load_txt(os.path.join(args.RESOURCE_DIR, 'val_task_paths.txt')).strip().split('\n')
    args.train_paths = list(sorted([os.path.join(args.DATASET_DIR, path) for path in args.train_paths])) if not args.use_robot else list(sorted([os.path.join(args.DATASET_DIR, 'train_dataset', path) for path in args.train_paths]))

    args.correction_example_paths = list(sorted(os.listdir(os.path.join(args.RESOURCE_DIR, 'reprompt_examples'))))
    args.correction_example_paths = list(map(lambda x: os.path.join(args.RESOURCE_DIR, 'reprompt_examples',x), args.correction_example_paths))
    

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
            "stop": '\n',
            "seed": args.seed
        }

    args.api_generation_params = \
        {
            "max_tokens": args.api_max_tokens,
            "temperature": args.api_generation_temperature,
            "top_p": args.api_top_p,
            "n": args.api_generation_n,
            "logprobs": args.api_logprobs,
            "echo": args.api_echo,
            "presence_penalty": args.api_generation_presence_penalty,
            "frequency_penalty":args.api_generation_frequency_penalty,
            "stop": '\n',
            "seed": args.seed
        }
    
    


    return args
