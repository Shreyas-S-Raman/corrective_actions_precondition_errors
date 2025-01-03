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

class SceneGym():

    def __init__(self, scene_path, scene_num, task):
        self.reset(scene_path, scene_num, task)


    def step(self, program_lines, precond):

        #NOTE: assign objects and modify internal graph only on first step
        (message, message_params, init_graph_dict, final_state, graph_state_list, input_graph, id_mapping, info, graph_helper, modified_script, ___) = check_script(program_lines, precond, self.scene_path, inp_graph_dict=self.graph_dict, modify_graph=True if self.steps==0 else False, id_mapping=self.id_mapping, info=self.info, graph_helper = self.graph_helper)

        #print(message)

        #new graph dictionary is final dictionary in list of dicts
        self.prev_graphs_stack.append((self.steps, self.graph_dict))
        self.graph_dict = graph_state_list[-1]

        #update the assigned id_mapping and room mapping (in self.info) for objects from first step
        self.id_mapping = id_mapping
        self.info = info
        self.graph_helper = graph_helper

        #update step count
        self.steps += 1

        return message, message_params, self.graph_dict, self.steps, init_graph_dict, modified_script

    def backtrack_step(self):

        (self.steps, self.graph_dict) = self.prev_graphs_stack.pop()

        return self.graph_dict

    def reset(self, scene_path, scene_num, task):

        env_graph = utils.load_graph(scene_path)
        self.graph_dict = env_graph.to_dict()
        self.id_mapping = {}
        self.info = {}
        self.graph_helper = None
        self.scene_path = scene_path

        self.prev_graphs_stack = []
        self.initial_graph_dict = self.graph_dict

        self.scene = scene_num; self.task = task; self.steps = 0

    def reset_graph_dict(self, target_step):
        '''resets the graph_dict field to the graph at the required target step'''

        while self.prev_graphs_stack[-1][0]!=target_step:
            self.prev_graphs_stack.pop()

        self.graph_dict = self.prev_graphs_stack[-1]
