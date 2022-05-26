

class PromptGenerator():

    def __init__(self, prompt_args):

        self.fixed = prompt_args.fixed_prompt
        self.question = prompt_args.question_prompt

        self.prompt_template = 'Error: {}. A corrective step would be:'
        self.corrective_prompt = 'the task has failed'

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

    def generate_prompt(self, error_type, error_message, best_curr, translated_action):

        #fixed corrective prompt i.e. not unique based on error type
        if self.fixed:
            corrective_prompt = self.corrective_prompt

        #customized error message based on error type
        else:

            prompt_functions = {'parsibility': self._parsing_error_prompt, 'empty_program': self._empty_program_prompt, 'precond': self._precond_error_prompt, 'check_script':self._check_script_error_prompt, 'nogen':self._nogen_error_prompt }

            corrective_prompt = prompt_functions[error_type](error_message, best_curr, translated_action)


        return self.prompt_template.format(corrective_prompt)
