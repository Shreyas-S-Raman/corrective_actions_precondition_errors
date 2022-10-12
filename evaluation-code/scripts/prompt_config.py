
PROMPT_TEMPLATE_1 = 'Error: {}. {}'
PROMPT_TEMPLATE_2 = 'Error: {}. Generate a list of reasons why:'
PROMPT_TEMPLATE_3 = 'Step failed. {}. {}'
PROMPT_TEMPLATE_4 = 'Step failed. {}. Generate a list of reasons why:'
PROMPT_TEMPLATE_5 = 'Step {} failed. {}. {}'

prompt_templates = {1:(PROMPT_TEMPLATE_1, ['error_info','suggestion']), 2:(PROMPT_TEMPLATE_2, ['error_info','generate_list']), 3:(PROMPT_TEMPLATE_3,['error_info','suggestion']), 4:(PROMPT_TEMPLATE_4, ['error_info']), 5:(PROMPT_TEMPLATE_5, ['step_no', 'error_info', 'suggestion'])}

#information about 'existence' of error i.e. an error occured
ERROR_NOTION_1 = 'Task failed'

#information objects related to error w/o state i.e. error occured with {object}
ERROR_INFERENCE_1 = 'I cannot {}'
ERROR_INFERENCE_2 = 'I cannot {} the {}'

#information about the "state" of objects related to error e.g. error occured with {object} because {reason}
ERROR_CAUSE_1 = '{}'
ERROR_CAUSE_2 = 'I cannot {} the {} because {}'

error_provided = {'notion':ERROR_NOTION_1, 'inference_1':ERROR_INFERENCE_1, 'inference_2': ERROR_INFERENCE_2, 'cause_1':ERROR_CAUSE_1, 'cause_2':ERROR_CAUSE_2}

SUGGESTION_1 = 'A correct step would be to'
SUGGESTION_2 = 'Therefore, a correct step would be to'
SUGGESTION_3 = 'Next time you should'
SUGGESTION_4 = 'Correct the step:'
SUGGESTION_5 = 'Instead I should have'

suggestion_provided = {1:SUGGESTION_1, 2:SUGGESTION_2, 3:SUGGESTION_3, 4: SUGGESTION_4, 5: SUGGESTION_5}


'''
Prompts for specific error causes
'''
class CausalErrors:

    #internally contained: obj, subtype, type
    INTERNALLY_CONTAINED1 = ['the {} is inside something', ['obj']]

    #unflipped bool state: obj, state, subtype, type
    UNFLIPPED_BOOL_STATE1 = ['{} is already {}', ['obj', 'state']]

    #hands full: character, subtype, type
    HANDS_FULL1 = ['I cannot {} {}. My hands are full', ['obj','action']]

    #already sitting: character, subtype, type
    ALREADY_SITTING1 = ['{} sitting', ['character']]

    #no path: char_room, target_room, subtype, type
    NO_PATH1 = ['The {} and {} are not connected', ['char_room','target_room']]

    #door closed: doorlist, char_room, target_room, subtype, type
    DOOR_CLOSED1 = ['The door to {} is closed', ['target_room']]

    #proximity: character, obj, subtype, type
    PROXIMITY1 = ['{} not near the {}',['character','obj']]

    #not find: obj, subtype, type
    NOT_FIND1 = ['I cannot find {}',['obj']]

    #invalid action: obj, subtype, type
    INVALID_ACTION1 = ['{} {} is not allowed', ['action','obj']]
    INVALID_ACTION2 = ['{} {}',['obj','subtype']]

    #max occupancy: obj, subtype, type
    MAX_OCCUPANCY1 = ['The {} is full', ['obj']]

    #not holding: character, obj, subtype, type
    NOT_HOLDING1 = ['{} doesn\'t have the {}', ['character','obj']]
    NOT_HOLDING2 = ['{} don\'t have the {}', ['character','obj']]

    #not holding any: character, subtype, type
    NOT_HOLDING_ANY1 = ['{} not holding anything', ['character']]

    #not facing: character, obj, subtype, type
    NOT_FACING1 = ['{} not facing the {}',['character','obj']]

    #missing step: character, subtype, type
    MISSING_STEP1 = ['{} {}',['character','subtype']]

    #invalid room: char_room, target_room, subtype, type
    INVALID_ROOM1 = ['{} doesn\'t exist', ['target_room']]
