import sys
import json
import getopt

# -- we want to import the scene_gym.py file compiled by Shreyas:
sys.path.append('/home/shreyas/LLM_Planner/evaluation-code/scripts/')
import scene_gym as gym
   
from utils import prepare_script

# -- loading JSON containing all allowable states in VirtualHome:
object_states = json.load(open('/home/shreyas/LLM_Planner/evaluation-code/resources/object_states.json', 'r'))

# -- compiling paths and file names for scene graphs:
scene_graph_dir = '/home/shreyas/LLM_Planner/evaluation-code/example_graphs/'
scene_graph_paths = ['TrimmedTestScene{}_graph.json'.format(x) for x in range(1, 8)]

def measure_graph_similarity(gt_plan_file, gen_plan_file, scene_num):

    def _executePlan(plan_file):
        # 1. create a VH-gym object for stepping through a scene given a plan:
        stepper = gym.SceneGym(scene_graph_dir + scene_graph_paths[scene_num], scene_num, '')

        # 2. create a list of parsed instructions for execution:
        plan, _ = prepare_script(stepper.initial_scene_graph, plan_file)

        for x in range(len(plan)):
            message, _, graph_dict, _, _, _ = stepper.step([plan[x]], preconditions)
            if not 'is executable' in message:
                # -- sometimes, the execution will fail; nevertheless, we need to take the graph at this point:
                break

        # -- we will return the graph at the point where execution ends:
        return stepper.graph_dict
   
    def _getObjectStateDict(graph):
        # -- to make similarity easier to measure, we will reform the scene graph as a dictionary of objects:
        obj_dict = {}
        for N in graph['nodes']:
            # -- map each object name to a list of states:
            obj_dict[N['class_name']] = N['states']

        return obj_dict


    gt_scene_graph = _executePlan(gt_plan_file)
    gen_scene_graph = _executePlan(gen_plan_file)

    gt_objs = _getObjectStateDict(gt_scene_graph);  gen_objs = _getObjectStateDict(gen_scene_graph)

    total_possible_combinations = 0

    # -- we will compute the distance by calculating number of mismatching states and then subtracting that from the total number of object-state combinations possible across the scene graph:
    distance = 0
    for O in gt_objs.keys():
        num_mismatches = 0
        
        for state in gt_objs[O]:
        # -- checking to see if a given state of one object is found in the states list for the same object in the other scene:
            if state not in gen_objs[O]:
                num_mismatches += 1

        # -- add number of mismatches to the total:
        distance += num_mismatches

        # -- also, tally up the total number of states possible:
        total_possible_combinations += len(object_states[O])


    return (total_possible_combinations - distance) / float(total_possible_combinations) * 100.0
    


# intuition: we will provide two programs/scripts for VirtualHome, run through them,
#               and then we compare the scene graph at the end of the execution.

if __name__ == '__main__':
    groundtruth_file_name = None
    generated_file_name = None
    
    # -- read the arguments given to script:
    opts, _ = getopt.getopt(sys.argv[1:], 'gen:gro:h', ['gen_program=', 'gt_program=', 'help'])

    if opts:
        for opt, arg in opts:
            if opt in ('-gen', '--gen_program'):
                generated_file_name = str(arg)

            elif opt in ('-gro', '--gt_program'):
                groundtruth_file_name = str(arg)

            
    for x in range(1, 8):
        print(measure_graph_similarity(groundtruth_file_name, generated_file_name, x))


