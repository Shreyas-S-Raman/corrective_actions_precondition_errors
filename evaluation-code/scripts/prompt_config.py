
PROMPT_TEMPLATE_1 = 'Error: {}. {}'
PROMPT_TEMPLATE_2 = 'Error: {}. Generate a list of reasons why:'
PROMPT_TEMPLATE_3 = 'Step failed. {}. {}'
PROMPT_TEMPLATE_4 = 'Step failed. {}. Generate a list of reasons why:'

prompt_templates = {1:PROMPT_TEMPLATE_1, 2:PROMPT_TEMPLATE_2, 3:PROMPT_TEMPLATE_3, 4:PROMPT_TEMPLATE_4}

#information about 'existence' of error i.e. an error occured
ERROR_NOTION = 'Task failed'

#information objects related to error w/o state i.e. error occured with {object}
ERROR_INFERENCE = 'I cannot {} {}'

#information about the "state" of objects related to error e.g. error occured with {object} because {reason}
ERROR_CAUSE = '{}'

error_provided = {'notion':ERROR_NOTION, 'inference':ERROR_INFERENCE, 'cause':ERROR_CAUSE}

SUGGESTION_1 = 'A correct step would be:'
SUGGESTION_2 = 'Therefore, a correct step would be:'
SUGGESTION_3 = 'Next time you should:'

suggestion_provided = {1:SUGGESTION_1, 2:SUGGESTION_2, 3:SUGGESTION_3}
