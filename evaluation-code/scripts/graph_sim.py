import sys
import json
import getopt
import pandas as pd

# -- we want to import the scene_gym.py file compiled by Shreyas:
sys.path.append('/home/shreyas/LLM_Planner/vh_scripts/')
import scene_gym as gym
   
from utils import prepare_script

sys.path.append('/home/shreyas/LLM_Planner/evaluation-code/dataset_utils')
from add_preconds import *

# -- loading JSON containing all allowable states in VirtualHome:
object_states = json.load(open('/home/shreyas/LLM_Planner/evaluation-code/resources/object_states.json', 'r'))

# -- compiling paths and file names for scene graphs:
scene_graph_dir = '/home/shreyas/LLM_Planner/evaluation-code/example_graphs/'
scene_graph_paths = ['TrimmedTestScene{}_graph.json'.format(x) for x in range(1, 8)]

class_equivalence = json.load(open('/home/shreyas/LLM_Planner/evaluation-code/resources/class_name_equivalence.json', 'r'))

def measure_graph_similarity(gt_plan_file, gen_plan_file, scene_num):

    def _executePlan(plan_file):
        # 1. create a VH-gym object for stepping through a scene given a plan:
        stepper = gym.SceneGym(scene_graph_dir + scene_graph_paths[scene_num-1], scene_num, '')
        
        # 2. create a list of parsed instructions for execution (using initial scene graph):
        plan, _ = prepare_script(stepper.graph_dict, plan_file, execute=False)

        if plan == False:
            return None

        for x in range(len(plan)):
            #print(plan[x])

            # -- use function from dataset_utils/add_preconds.py for reading preconditions for stepper:
            preconditions = get_preconds_script([plan[x]]).printCondsJSON()

            #print(preconditions)

            # -- take a "step" with the step() method:
            message, _, graph_dict, _, _, _ = stepper.step([plan[x]], preconditions)
            if not 'is executable' in message:
                # -- sometimes, the execution will fail; nevertheless, we need to take the graph at this point:
                break

        # -- we will return the graph at the point where execution ends:
        return stepper.graph_dict, plan
   
    def _getObjectStateDict(graph):
        # -- to make similarity easier to measure, we will reform the scene graph as a dictionary of objects:
        obj_dict = {}
        for N in graph['nodes']:
            # -- map each object name to a list of states:
            obj_dict[N['class_name']] = N['states']

        return obj_dict

    def _getInteractedObjects(plan):
        used_objects = []
        for step in plan:
            for el in step.split(' '):
                if '<' in el:
                    obj_name = el.split('<')[1].split('>')[0]
                    used_objects.append(obj_name)

        return used_objects

    (gt_scene_graph, gt_plan) = _executePlan(gt_plan_file)
    (gen_scene_graph, gen_plan) = _executePlan(gen_plan_file)

    if gt_scene_graph == None or gen_scene_graph == None:
        return -1

    gt_objs = _getObjectStateDict(gt_scene_graph); gen_objs = _getObjectStateDict(gen_scene_graph)

    # -- we will compute the distance by calculating number of mismatching states and then 
    #       subtracting that from the total number of object-state combinations possible across the scene graph:
    distance = 0

    total_num_states = 0

    all_used_objects = set(_getInteractedObjects(gt_plan)) | set(_getInteractedObjects(gen_plan))
    #print(all_used_objects)
    #print(gen_objs)
    #input(gt_objs)
    check_all_objects = False

    # -- we will do a union over all objects across both ground-truth and generated program scene graphs:
    num_mismatches = 0
    
    # NOTE: GT objects -> GEN objects:
    for O in set(gt_objs.keys()) | set(gen_objs.keys()):
        #num_mismatches = 0
        
        if not check_all_objects:
            if O not in all_used_objects:
                # -- checking only for objects used in the plan:
                continue

        #if O not in object_states:
        #   # -- we will only focus on objects which are listed in the object-state JSON index:
        #   continue

        gen_obj_alias = None
        gt_obj_alias = None
        if O not in gen_objs and O in gt_objs:
            # -- if an object does not exist in one map but in another, then we check if there are any equivalent objects:
            match = False
            if O in class_equivalence:
                for alias in class_equivalence[O]:
                    if alias in gen_objs:
                        print(alias, O)
                        gen_obj_alias = alias
                        match = True
                        break
            
            if not match:
                num_mismatches += len(gt_objs[O]) 
                total_num_states += len(gt_objs[O])
                print('missing gt obj:', O, gt_objs[O])
                print(gen_objs.keys())
                continue

        if O not in gt_objs and O in gen_objs:
            match = False
            if O in class_equivalence:
                for alias in class_equivalence[O]:
                    if alias in gt_objs:
                        print(alias, O)
                        gt_obj_alias = alias
                        match = True
                        break

            if not match:
                num_mismatches += len(gen_objs[O])
                total_num_states += len(gen_objs[O])
                print('missing gen obj:', O, gen_objs[O])
                print(gt_objs.keys())
                continue
        
        # -- objects may not exactly match by name but there may be some alias for a similar object:
        gen_states = set(gen_objs[gen_obj_alias]) if gen_obj_alias else set(gen_objs[O])
        gt_states = set(gt_objs[gt_obj_alias]) if gt_obj_alias else set(gt_objs[O])
        
        intersection = len(gen_states & gt_states)
        union =  len(gen_states | gt_states)
        diff = union - intersection
        num_mismatches += diff
        
        if diff > 0:
            print(O, (gen_obj_alias if gen_obj_alias else ''), gen_states)
            print(O, (gt_obj_alias if gt_obj_alias else ''), gt_states)
            print('union', union)
            print('inter', intersection)
        
        total_num_states += union

        # -- checking to see if a given state of one object is found i

        # -- add number of mismatches to the total:
        #distance += num_mismatches

        # -- also, tally up the total number of states possible:
        #total_num_states += 1
    #endfor

    # NOTE: GEN objects -> GT objects
    #for O in gen_objs.keys():
    #    num_mismatches = 0
    #    if not check_all_objects:
    #        if O not in all_used_objects:
    #            # -- checking only for objects used in the plan:
    #            continue
    #    
    #    if O not in object_states:
    #        # -- we will only focus on objects which are listed in the object-state JSON index:
    #        continue
    #
    #    elif O not in gt_objs:
    #        # -- we will penalize for any object that could possibly exist but is not found in the scene:
    #        num_mismatches += len(object_states[O])
    #        continue
    #
    #    for state in gen_objs[O]:
    #    # -- checking to see if a given state of one object is found in the states list for the same object in the other scene:
    #        if state not in gt_objs[O]:
    #            num_mismatches += 1
    #
    #    # -- add number of mismatches to the total:
    #    distance += num_mismatches
    #
    #    # -- also, tally up the total number of states possible:
    #    total_num_states += len(object_states[O])
    #endfor

    print(num_mismatches)
    print(total_num_states)

    if total_num_states > 0:
        return (total_num_states - num_mismatches) / float(total_num_states) * 100.0

    return 100.0
#enddef


# intuition: we will provide two programs/scripts for VirtualHome, run through them,
#               and then we compare the scene graph at the end of the execution.

if __name__ == '__main__':
    groundtruth_file_name = None
    generated_file_name = None
    scene_num = None

    # -- we can also use the JSON file from the construct_generation_dict() function to load GT files:
    gt_dict = None

    # -- we need to provide the location to the generated programs to map to their related GT file:
    gen_program_dir = None

    # -- read the arguments given to script:
    opts, _ = getopt.getopt(
            sys.argv[1:], 
            'gt:exp:gen:gro:sce:csv:h', 
            ['gt_json_file=', 'gen_program_dir=', 'gen_program=', 'gt_program=', 'scene=', 'csv=', 'help']
    )

    log_df = None

    overall_avg_similarity = 0
    num_tasks = 0

    if opts:
        for opt, arg in opts:
            if opt in ('-jso', '--gt_json_file'):
                gt_dict = json.load( open(str(arg), 'r'))
                #print(gt_dict.keys())

            elif opt in ('exp', '--gen_program_dir'):
                gen_program_dir = str(arg)

            elif opt in ('-gen', '--gen_program'):
                generated_file_name = str(arg)

            elif opt in ('-gro', '--gt_program'):
                groundtruth_file_name = str(arg)

            elif opt in ('-sce', '--scene'):
                scene_num = int(arg)

            elif opt in ('-csv', '--csv'):
                log_df = pd.read_csv(arg)

    if log_df is not None:
        # -- this means we are using a CSV file that was logging data from W&B:
        for i in range(0, len(log_df)):
            # -- create a temp file for the ground-truth plan and the generated plan:
            gt_plan_script = log_df['most_similar_gt_program_text'][i]
            gen_plan_script = log_df['parsed_text'][i]

            #print(gen_plan_script)
            #print(gt_plan_script)

            if not gt_dict:
                temp_file_gt = open('temp_gt.txt', 'w')
                for line in gt_plan_script.split(', '):
                    temp_file_gt.write(line + '\n')
                temp_file_gt.close()
            else:
                temp_file_gt = gt_dict["('{0}', '')".format(log_df['task'][i])]['gt_path']

            temp_file_gen = open('temp_gen.txt', 'w')
            try:
                for line in gen_plan_script.split(', '):
                    temp_file_gen.write(line + '\n')
                temp_file_gen.close()
            except Exception:
                continue

            avg_similarity = 0

            if 'scene' not in log_df:
                for x in range(1, 8):
                    # -- go through each scene graph:
                    sim_value = measure_graph_similarity('temp_gt.txt', 'temp_gen.txt', x)
                    avg_similarity += sim_value

                avg_similarity /= 7

            else:
                avg_similarity += measure_graph_similarity('temp_gt.txt', 'temp_gen.txt', int(log_df['scene'][i]))
    
            print('task ' + log_df['task'][i] + ': ' + str(avg_similarity) + '%')

            if avg_similarity >= 0:
                overall_avg_similarity += avg_similarity
                num_tasks += 1

        overall_avg_similarity /= num_tasks

        print('--------------------------')
        print('num_tasks:', num_tasks)
        print('overall similarity (avg over all tasks) : ' + str(overall_avg_similarity))

    elif gt_dict and gen_program_dir:
        overall_avg_similarity = 0
        num_tasks = 0

        for key in gt_dict.keys():

            # -- groundtruth file name is given as a mapped value in the JSON/dict:
            groundtruth_file_name = gt_dict[key]['gt_path']

            for x in range(1, 8):
                # -- each generated file's name consists of the task description with some textual modifications AND scene number:
                x = int(key[-1])
                generated_file_name = gen_program_dir + '/' + str(key)[:-1].split(',')[0][:-1].lower().replace(' ', '_') + '-1-scene' + str(x) + '.txt'

                sim_value = measure_graph_similarity(groundtruth_file_name, generated_file_name, x)
                avg_similarity += sim_value
            
            avg_similarity /= 7
    
            print('task ' + key + ': ' + str(sim_value) + '%')

            if sim_value >= 0:
                overall_avg_similarity += sim_value
                num_tasks += 1

        overall_avg_similarity /= num_tasks
        
        print('--------------------------')
        print('overall similarity (avg over all tasks) : ' + str(overall_avg_similarity)) 


    else:
        if scene_num is None:
            # -- iterate through all scenes and compute an average graph similarity value:
            avg_similarity = 0
            for x in range(1, 8):
                avg_similarity += measure_graph_similarity(groundtruth_file_name, generated_file_name, x)

            print(avg_similarity / 7)

        else:
            # -- we will only compute the graph similarity for a given scene:
            similarity = measure_graph_similarity(groundtruth_file_name, generated_file_name, scene_num)
            print(similarity)

