from prompt_config import  prompt_templates, error_provided, suggestion_provided

class PromptGenerator():

    def __init__(self, prompt_args):

        self.prompt_template = prompt_templates[prompt_args['prompt_template']]
        self.error_information = error_provided[prompt_args['error_information']]
        self.error_info_type = prompt_args['error_information']
        self.suggestion = suggestion_provided[prompt_args['suggestion_no']]



    def _nogen_error_prompt(self, nogen_error, best_curr, translated_action):
        pass

    def _parsing_error_prompt(self, parsing_error, best_curr, translated_action):
        pass


    def _empty_program_prompt(self, empty_program_error, best_curr, translated_action):
        pass


    def _precond_error_prompt(self, precond_error, best_curr, translated_action):
        pass

    def _check_script_error_prompt(self, executability_error, best_curr, translated_action):
        pass

    def _parse_inference_error(self, error_message):
        pass

    def generate_prompt(self, error_type, error_message, step, best_curr, translated_action):


        #format the error information for the prompt template
        if self.error_info_type=='inference':
            action, object = self._parse_inference_error(error_message)
            
        
        elif self.error_info_type=='cause':

            prompt_functions = {'parsibility': self._parsing_error_prompt, 'empty_program': self._empty_program_prompt, 'precond': self._precond_error_prompt, 'check_script':self._check_script_error_prompt, 'nogen':self._nogen_error_prompt }

            cause = prompt_functions[error_type](error_message, best_curr, translated_action)


       

        
