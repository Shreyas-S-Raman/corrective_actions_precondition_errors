import copy
import sys
sys.path.append('../../../spot_sdk_interface/nlmap_spot')
from actions import Action, WalkToAction, GrabAction, StandSitAction, OpenCloseAction, PutDownAction
import os
import pdb

class SlangGymEnvironment():

    def __init__(self, hostname, scene_num, scene_path, with_robot, task):

        self.slang_action = Action(hostname, scene_num, scene_path, with_robot)

        self.walk_executor = WalkToAction(hostname, self.slang_action)
        self.pickup_executor = GrabAction(hostname, self.slang_action)
        self.standsit_executor = StandSitAction(hostname, self.slang_action)
        self.openclose_executor = OpenCloseAction(hostname, self.slang_action)
        self.putdown_executor = PutDownAction(hostname, self.slang_action)

        self.skill_suite = {'walk_to': self.walk_executor.navigate, 'pick_up': self.pickup_executor.clipseg_grab, 
        'stand_up': self.standsit_executor.stand_up, 'sit_down': self.standsit_executor.sit_down, 
        'open': self.openclose_executor.open, 'close': self.openclose_executor.close, 'put_down': self.putdown_executor.put_down}

        self.steps = 0
        self.scene = scene_num; self.task = task
    
    def step(self, generated_step, translated_step):
        obj, skill, location, precondition_error, __, __, __ = self.slang_action._parse_step_to_action(generated_step, translated_step)

        assert (precondition_error is False), "ERROR: precondition check in generation_utils_robot.py failed, NON-EXECUTABLE step provided to slang_gym"


        if self.slang_action.use_translated:
            executed = self.skill_suite[skill](translated_step)
        else:
            executed = self.skill_suite[skill](generated_step)
        
        self.steps += 1

        return executed, self.steps, self.slang_action.object_states, self.slang_action.room_state, self.slang_action._get_curr_spot_state()
    
    def get_current_state(self):

        return {'object_states': self.slang_action.object_states, 'room_state': self.slang_action.room_state, 'spot_state': self.slang_action._get_curr_spot_state()}