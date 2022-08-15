from distutils.log import error
from prompt_config import  prompt_templates, error_provided, suggestion_provided
import re


class ErrorParsing():

    def __init__(self, custom_cause):
        self.custom_cause = custom_cause

    def _get_action_and_objs(self, block_str):
        """ Given a str block [Rinse] <CLEANING SOLUTION> (1)
            parses the block, returning Action, List Obj, List Instance
        """
        action = block_str[1:block_str.find(']')]
        block_str = block_str[block_str.find(']')+3:-1]
        block_split = block_str.split(') <') # each element is <name_obj> (num)

        obj_names = [block[0:block.find('>')].lower().strip().replace('_',' ') for block in block_split if len(len(block[0:block.find('>')].strip()) > 0)]
        
        return action, obj_names[0]
    
    def _get_error_reason(self, error_message, error_cause, obj, action):

        return self._parse_precond_error(error_message, obj, action) if error_cause=='precond' else self._parse_execution_error(error_message, obj, action)
    
    def _parse_precond_error(self, error_message, obj, action):

        unflipped_state = {'PLUGOUT': 'plugged out', 'PLUGIN':'plugged in', 'SWITCHOFF': 'turned off', 'SWITCHON': 'truned on', 'PUTOFF': 'off', 'SIT': 'sitting', 'STANDUP': 'standing up', 'WAKEUP': 'standing up'}

        if not self.custom_cause:
            return error_message.split(',')[1].strip().lower()
        else: 
            return '{} is already {}'.format(obj, unflipped_state[action.upper()])
        
    def _parse_execution_error(self, error_message, obj, action):
        
        if not self.custom_cause:

            '''
            The cause of error always occurs after the first '<' character

            e.g. Script is not executable, since <character> (65) is not holding <mail> (1000) when executing "[PUTBACK] <mail> (1000) <table> (107) [5]"

            character is not holding mail when executing putback mail table
            '''
            error_message = error_message[error_message.find('<'):]

            #replace bracket and quotation formatting in error message
            error_message = error_message.replace('[','').replace(']','')
            error_message = error_message.replace('<','').replace('>','')
            error_message = error_message.replace('(','').replace(')','')
            error_message = error_message.replace('"','')
            error_message = re.sub('\d','',error_message)
            #remove extra spaces caused by earlier replacements
            error_message = error_message.replace('  ', ' ')
            error_message = error_message.replace('executing', 'trying to')
            
            return error_message.strip().lower()

        elif 'does not have a free hand' in error_message:
            return 'I cannot {} {}. My hands are full'.format(action, obj)
        
        elif 'is not close to' in error_message:
            return 'I am not near the {}'.format(obj)
        
        elif 'does not face' in error_message:
            return 'I am not facing the {}'.format(obj)
        
        elif 'not holding' or 'not grabbed' in error_message:
            return 'I don\'t have the {}'.format(obj)

        elif 'inside another closed thing' in error_message:
            return 'the {} is inside something'.format(obj)
        
        elif 'is sitting' in error_message:
            return 'I am sitting'
        
        elif 'Door(s)' in error_message:
            return 'The door to {} is closed'.format(obj)
        
        elif 'is not' in error_message:
            return '{} {} is not allowed'.format(action, obj)
        
        else:
            return 'I cannot {} {}'.format(action, obj)


class PromptGenerator():

    def __init__(self, prompt_args):

        (self.prompt_template, self.prompt_inserts) = prompt_templates[prompt_args['prompt_template']]


        self.error_information = error_provided[prompt_args['error_information']]
        self.error_info_type = prompt_args['error_information']
        self.suggestion = suggestion_provided[prompt_args['suggestion_no']]

        self.error_parser = ErrorParsing(prompt_args['custom_cause'])

    
    def _create_inference2_prompt(self, **kwargs):
        return self.error_information.format(kwargs['action'], kwargs['obj'])
    
    def _create_inference1_prompt(self, **kwargs):
        return self.error_information.format(kwargs['best_curr'].lower())
    
    def _create_notion_prompt(self, **kwargs):
        return self.error_information
    
    def _create_cause1_prompt(self, **kwargs):
        return self.error_information.format(kwargs['error_cause'])
    
    def _create_cause2_prompt(self, **kwargs):
        return self.error_information.format(kwargs['action'], kwargs['obj'], kwargs['error_cause'])
    
    def _create_emptyprogram_prompt(self):
        return 'generate a list of steps'
    
    def _create_parsibility_prompt(self, **kwargs):
        pass
    def _cerate_nogen_prompt(self,**kwargs):
        pass


    def generate_prompt(self, error_type, error_message, step, best_curr, program_line):

        generator_functions = {'inference_1': self._create_inference1_prompt, 'inference_2': self._create_inference2_prompt, 
        'notion': self._create_notion_prompt,
        'cause_1': self._create_cause1_prompt, 
        'cause_2': self._create_cause2_prompt}

        

        #extract the action and object causing error from program line
        action, obj = self.error_parser._get_action_and_objs(program_line)


        #format the error information for the prompt template
        if error_type == 'empty_program':
            error_info = self._create_emptyprogram_prompt()
        if error_type == 'parsibility':
            error_info = self._create_parsibility_prompt()
        else:
            #extract the cause for the error
            error_cause = self.error_parser._get_error_reason(error_message, error_type, obj, action)

            error_info = generator_functions[self.error_info_type]({'obj':obj, 'action':action, 'error_cause': error_cause, 'best_curr': best_curr})


        #format the final error prompt template
        


       

        
