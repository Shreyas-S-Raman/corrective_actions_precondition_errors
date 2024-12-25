import copy
import sys
sys.path.append('../simulation')
from evolving_graph.utils import *
from evolving_graph.environment import *
from evolving_graph.execution import *
from evolving_graph.scripts import *
from evolving_graph.preparation import *
from evolving_graph.check_programs import *

'''
Implementation Notes:

    - self.prev_graph_stack: stack containing evolving graph dictionaries for each step in plan ==> allows us to pop() to return to prev. state

'''
max_nodes = 300


class SceneGymAllSteps():

    def __init__(self, scene_path, scene_num, task):
        self.reset(scene_path, scene_num, task)
    

    def check_execution(self, program_lines, precond):

        helper = utils.graph_dict_helper(max_nodes=max_nodes)

        try:
            script = read_script_from_list_string(program_lines)
        except Exception as e:
            print('Parsing Error - check_execution in SceneGymAllSteps: ', e)
        script, precond = modify_objects_unity2script(helper, script, precond)

        executable, new_graph_dict = check_executability_saycan((script, self.initial_graph_dict))

        return executable

    def step(self, program_lines, precond, modify_prev=True):

        #NOTE: assign objects and modify internal graph only on first step
        (message, message_params, init_graph_dict, final_state, graph_state_list, input_graph, id_mapping, info, graph_helper, modified_script, ___) = check_script(program_lines, precond, self.scene_path, inp_graph_dict=copy.deepcopy(self.initial_graph_dict), modify_graph=True)

        #new graph dictionary is final dictionary in list of dicts
        if modify_prev:
            self.prev_graphs_stack = graph_state_list

        #update step count
        self.steps += 1

        return message, message_params, final_state, self.steps, init_graph_dict, modified_script

    def backtrack_step(self):

        pass

    def reset(self, scene_path, scene_num, task):

        env_graph = utils.load_graph(scene_path)
        self.initial_graph_dict = env_graph.to_dict()
       
        self.scene_path = scene_path

        self.prev_graphs_stack = []

        self.scene = scene_num; self.task = task; self.steps = 0

    def reset_graph_dict(self, target_step):
        '''resets the graph_dict field to the graph at the required target step'''
        
        pass
