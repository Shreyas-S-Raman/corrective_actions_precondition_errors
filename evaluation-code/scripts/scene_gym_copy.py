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

class SceneGymAllSteps():

    def __init__(self, scene_path, scene_num, task):
        self.reset(scene_path, scene_num, task)


    def step(self, program_lines, precond):

        #NOTE: assign objects and modify internal graph only on first step
        (message, message_params, init_graph_dict, final_state, graph_state_list, input_graph, id_mapping, info, graph_helper, modified_script, ___) = check_script(program_lines, precond, self.scene_path, inp_graph_dict=self.initial_graph_dict, modify_graph=True)

        #new graph dictionary is final dictionary in list of dicts
        self.prev_graphs_stack = graph_state_list

        #update step count
        self.steps += 1

        return message, message_params, self.graph_dict, self.steps, init_graph_dict, modified_script

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
