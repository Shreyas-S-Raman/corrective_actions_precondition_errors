# Language Planner Evaluation
This repo is modified from VirtualHome (v1.0). Website is here: [www.virtual-home.org](http://virtual-home.org). Work in progress.

## Set Up

### Clone repository and install the dependencies
Note that this code base is based on Python 3.

To install dependencies:

```bash
pip install -r requirements.txt
```

### Download dataset

Download VirtualHome dataset [here](http://virtual-home.org/release/programs/programs_processed_precond_nograb_morepreconds.zip). Once downloaded and unzipped, move the programs into the `dataset` folder.

The dataset should follow the following structure:

```
dataset
└── programs_processed_precond_nograb_morepreconds
	|── initstate
	├── withoutconds
	├── executable_programs
	|   ├── TrimmedTestScene7_graph
	|	└── ...
	└── state_list
		├── TrimmedTestScene7_graph
	   	└── ...	
```

The folders `withoutconds` and `initstate` contain the original programs and pre-conditions. 

When a script is executed in an environment, the script changes by aligning the original objects with instances in the environment. You can view the resulting script in `executable_programs/{environment}/{script_name}.txt`.

To view the graph of the environment, and how it changes throughout the script execution of a program, check   `state_list/{environment}/{script_name}.json`.

You can find more details of the programs and environment graphs in [dataset/README.md](dataset/README.md).

### Obtain OpenAI API Key (optional)

If you wish to use OpenAI's models (e.g. codex, gpt-3), obtain an OpenAI API Key and put it in `generation_utils.py`. Otherwise, the code may also work with any models from `HuggingFace Transformers`.

## Evaluation

Most eval code is in `scripts` folder.

Example command to run `Translated-Codex`:
```bash
python evaluate.py --expID 1 --query_task=all --add_desc=false --fresh=true --iterative=true --raw_lm=false --translated_condition=true --api_max_steps=15 --api_top_p=0.9 --seed=123 --use_cutoff_threshold=True --use_example_subset=False --use_similar_example=True --api_beta=0.3 --api_cutoff_threshold=0.8 --api_frequency_penalty=0.3 --api_n=10 --api_percent_terminate=0.5 --api_presence_penalty=0.5 --api_temperature=0.3 --engine davinci-codex
```

Example command to run `Translated-GPT-3`:
```bash
python evaluate.py --expID 2 --query_task=all --add_desc=false --fresh=true --iterative=true --raw_lm=false --translated_condition=true --api_max_steps=15 --api_top_p=0.9 --seed=123 --use_cutoff_threshold=true --use_example_subset=False --use_similar_example=True --api_beta=0.3 --api_cutoff_threshold=0.8 --api_frequency_penalty=0.3 --api_n=10 --api_presence_penalty=0.5 --api_temperature=0.3 --engine davinci --allow_charges true
```


