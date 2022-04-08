from enum import Enum
import json
import os

DATASET_DIR = '../dataset/programs_processed_precond_nograb_morepreconds'
RESOURCE_DIR = '../resources'
SCENE_DIR = '../example_graphs'
SKETCH_PATH = os.path.join(RESOURCE_DIR, 'reformatted_sketch_annotation.json')

def build_unity2object_script(script_object2unity_object):
    """Builds mapping from Unity 2 Script objects. It works by creating connected
      components between objects: A: [c, d], B: [f, e]. Since they share
      one object, A, B, c, d, f, e should be merged
    """
    unity_object2script_object = {}
    object_script_merge = {}
    for k, vs in script_object2unity_object.items():
        vs = [x.lower().replace('_', '') for x in vs]
        kmod = k.lower().replace('_', '')
        object_script_merge[k] = [kmod] + vs
        if kmod in unity_object2script_object:
            prev_parent = unity_object2script_object[kmod]
            dest_parent = prev_parent
            source_parent = k
            if len(k) < len(prev_parent) and prev_parent != 'computer':
                dest_parent = k
                source_parent = prev_parent
            children_source = object_script_merge[source_parent]
            object_script_merge[dest_parent] += children_source
            for child in children_source: unity_object2script_object[child] = dest_parent

        else:
            unity_object2script_object[kmod] = k
        for v in vs:
            if v in unity_object2script_object:
                prev_parent = unity_object2script_object[v]
                dest_parent = prev_parent
                source_parent = k
                if len(k) < len(prev_parent) and prev_parent != 'computer':
                    dest_parent = k
                    source_parent = prev_parent
                children_source = object_script_merge[source_parent]
                object_script_merge[dest_parent] += children_source
                for child in children_source: unity_object2script_object[child] = dest_parent
            else:
                unity_object2script_object[v] = k

    return unity_object2script_object

def merge_add(d, k, v):
    if k == v:
        return
    # print(f'adding {k} --> {v}')
    if k in d:
        prev_v = d[k]
        # print(f'existing: {k} --> {prev_v}')
        merge_add(d, v, prev_v)
    else:
        d[k] = v

name_equivalence_path = os.path.join(RESOURCE_DIR, 'class_name_equivalence.json')
with open(name_equivalence_path, 'r') as f:
    abstract2detail = json.load(f)

detail2abstract = dict()
for abstract, details in abstract2detail.items():
    for detail in details:
        merge_add(detail2abstract, detail, abstract)
# detail2abstract = build_unity2object_script(abstract2detail)

class EvolveGraphAction(Enum):
    """
    All supported actions, value of each enum is a pair (humanized name, required_number of parameters)
    """
    CLOSE = ("Close", 1, 'close {}')
    DRINK = ("Drink", 1, 'drink {}')
    FIND = ("Find", 1, 'find {}')
    WALK = ("Walk", 1, 'walk to {}')
    GRAB = ("Grab", 1, 'grab {}')
    LOOKAT = ("Look at", 1, 'look at {}')
    # LOOKAT_SHORT = ("Look at short", 1, 'look at {}')
    # LOOKAT_MEDIUM = LOOKAT
    # LOOKAT_LONG = ("Look at long", 1, 'look at {}')
    OPEN = ("Open", 1, 'open {}')
    POINTAT = ("Point at", 1, 'point at {}')
    PUTBACK = ("Put", 2, 'put {} on {}')
    #PUT = ("Put", 2, '')
    #PUTBACK = PU, ''T
    PUTIN = ("Put in", 2, 'put {} in {}')
    PUTOBJBACK = ("Put back", 1, 'put back {}')
    RUN = ("Run", 1, 'run to {}')
    SIT = ("Sit", 1, 'sit on {}')
    STANDUP = ("Stand up", 0, 'stand up')
    SWITCHOFF = ("Switch off", 1, 'switch off {}')
    SWITCHON = ("Switch on", 1, 'switch on {}')
    TOUCH = ("Touch", 1, 'touch {}')
    TURNTO = ("Turn to", 1, 'turn to {}')
    WATCH = ("Watch", 1, 'watch {}')
    WIPE = ("Wipe", 1, 'wipe {}')
    PUTON = ("PutOn", 1, 'put on {}')
    PUTOFF = ("PutOff", 1, 'take off {}')
    GREET = ("Greet", 1, 'greet {}')
    DROP = ("Drop", 1, 'drop {}')
    READ = ("Read", 1, 'read {}')
    LIE = ("Lie", 1, 'lie on {}')
    POUR = ("Pour", 2, 'pour {} into {}')
    TYPE = ("Type", 1, 'type on {}')
    PUSH = ("Push", 1, 'push {}')
    PULL = ("Pull", 1, 'pull {}')
    MOVE = ("Move", 1, 'move {}')
    WASH = ("Wash", 1, 'wash {}')
    RINSE = ("Rinse", 1, 'rinse {}')
    SCRUB = ("Scrub", 1, 'scrub {}')
    SQUEEZE = ("Squeeze", 1, 'squeeze {}')
    PLUGIN = ("PlugIn", 1, 'plug in {}')
    PLUGOUT = ("PlugOut", 1, 'plug out {}')
    CUT = ("Cut", 1, 'cut {}')
    EAT = ("Eat", 1, 'eat {}') 
    SLEEP = ("Sleep", 0, 'sleep')
    WAKEUP = ("WakeUp", 0, 'wake up')
    RELEASE = ("Release", 1, 'release')

TASKSET = \
    ['Add paper to printer', 'Added meat to freezer', 'Admire art', 'Answer door', 'Answer emails', 'Apply lotion', 'Arrange folders', 'Arrange furniture', 'Breakfast', 'Bring dirty plate to sink', 'Bring me red cookbook', 'Browse computer', 'Browse internet', 'Brush teeth', 'Call family member with skype application', 'Carry groceries to kitchen', 'Change TV channel', 'Change TV channels', 'Change clothes', 'Change light', 'Change sheets and pillow cases', 'Change toilet paper roll', 'Check appearance in mirror', 'Check email', 'Chop vegetables', 'Clean', 'Clean  mirror', 'Clean bathroom', 'Clean dishes', 'Clean floor', 'Clean mirror', 'Clean room', 'Clean screen', 'Clean sink', 'Clean toilet', 'Close door', 'Collect napkin rings', 'Come home', 'Come in and leave home', 'Complete surveys on amazon turk', 'Compute', 'Computer work', 'Cook some food', 'Cut bread', 'Cutting', 'De-wrinkle sheet', 'Decorate it', 'Deficate', 'Do dishes', 'Do homework', 'Do laundry', 'Do taxes', 'Do work', 'Do work on computer', 'Draft home', 'Draw picture', 'Drink', 'Dry soap bottles', 'Dust', 'Eat', 'Eat cereal', 'Eat cheese', 'Eat dinner', 'Eat snacks and drink tea', 'Empty dishwasher and fill dishwasher', 'Enjoy view out window', 'Enter home', 'Entertain', 'Feed me', 'File documents', 'File expense reports', 'Find dictionary', 'Fix snack', 'Gaze out window', 'Get dressed', 'Get drink', 'Get glass of milk', 'Get glass of water', 'Get out dish', 'Get ready for day', 'Get ready for school', 'Get ready to leave', 'Get some water', 'Get something to drink', 'Get toilet paper', 'Getting  dresses', 'Give milk to cat', 'Go to sleep', 'Go to toilet', 'Grab some juice', 'Grab things', 'Greet guests', 'Greet people', 'Hand washing', 'Hang car keys', 'Hang keys', 'Hang pictures', 'Hang up car keys', 'Hang up jacket', 'Hang with friends', 'Have conversation with boyfriend', 'Have snack', 'Homework', 'Iron shirt', 'Juggling', 'Keep an eye on stove as something is cooking', 'Keep cats inside while door is open', 'Keep cats out of room', 'Leave home', 'Let baby learn how to walk', 'Light candles', 'Listen to music', 'Load dishwasher', 'Lock door', 'Look at mirror', 'Look at painting', 'Look in refrigerator', 'Look out window', 'Make bed', 'Make coffee', 'Make popcorn', 'Make toast', 'Manage emails', 'Mop floor', 'Movie', 'Oil dining room', 'Open bathroom window', 'Open curtains', 'Open door', 'Open front door', 'Open window', 'Organize', 'Organize closet', 'Organize pantry', 'Paint ceiling', 'Pay bills', 'Pet cat', 'Pet dog', 'Pick up', 'Pick up cat hair', 'Pick up dirty dishes', 'Pick up phone', 'Pick up spare change on dresser', 'Pick up toys', 'Place centerpiece', 'Play games', 'Play musical chairs', 'Play on laptop', 'Playing video game', 'Plug in nightlight', 'Prepare Dinner', 'Prepare pot of boiling water', 'Print out document', 'Print out papers', 'Pull up carpet', 'Push all chairs in', 'Push in desk chair', 'Push in dining room chair', 'Put alarm clock in bedroom', 'Put away clean clothes', 'Put away groceries', 'Put away jackets', 'Put away keys', 'Put away shoes', 'Put away toys', 'Put clothes away', 'Put down bags', 'Put groceries in Fridge', 'Put in chair', 'Put mail in mail organizer', 'Put new books in shelves', 'Put on coat', 'Put on coat and shoes', 'Put on glasses', 'Put on your shoes', 'Put out flowers', 'Put shoes in shoe rack', 'Put toys away', 'Put umbrella away', 'Put up decoration', 'Rain welcome', 'Raise blinds', 'Re arrange office', 'Read', 'Read book', 'Read magazine', 'Read news', 'Read newspaper', 'Read on sofa', 'Read to child', 'Read yourself to sleep', 'Rearrange photo frames', 'Receive credit card', 'Relax', 'Relax on sofa', 'Research', 'Restock', 'Rotate stock in refrigerator', 'Say goodbye to guests leaving', 'Scrubbing living room tile floor is once week activity for me', 'Send  email', 'Sent email', 'Set mail on table', 'Set up buffet area', 'Set up table', 'Settle in', 'Sew', 'Shampoo hair', 'Shave', 'Shop', 'Shred receipts', 'Shredding', 'Shut front door', 'Shut off alarm', 'Sit', 'Sit in chair', 'Sleep', 'Social media  checks', 'Sort laundry', 'Spread table with appropriate supplies', 'Start computer', 'Story reading time', 'Straighten paintings on wall', 'Straighten pictures on wall', 'Study', 'Style hair', 'Surf internet', 'Surf net', 'Surf web for money legitimate making opportunities', 'Sweep hallway please', 'Switch on lamp', 'Take dishes out of dishwasher', 'Take jacket off', 'Take nap', 'Take off coat', 'Take off outerwear', 'Take off shoes', 'Take shoes off', 'Take shower', 'Tale off shoes', 'Text friends while sitting on couch', 'Throw away newspaper', 'Throw away paper', 'Toast bread', 'Try yourself off', 'Turking', 'Turn light off', 'Turn night light on', 'Turn off TV', 'Turn off light', 'Turn on TV', 'Turn on TV with remote', 'Turn on computer', 'Turn on light', 'Turn on lights', 'Turn on radio', 'Type up document', 'Unload dishwasher', 'Unload various items from pockets and place them in bowl on table', 'Use computer', 'Use laptop', 'Use toilet', 'Vacuum', 'Wake kids up', 'Walk through', 'Walk to room', 'Wash clothes', 'Wash dirty dishes', 'Wash dishes', 'Wash dishes by hand', 'Wash dishes with dishwasher', 'Wash face', 'Wash hands', 'Wash monitor', 'Wash sink', 'Wash teeth', 'Watch  TV', 'Watch  horror  movie', 'Watch TV', 'Watch fly', 'Watch movie', 'Watch youtube', 'Wipe down baseboards please', 'Wipe down counter', 'Wipe down sink', 'Work', 'Write  school paper', 'Write an email', 'Write book', 'Write report', 'Write story', 'vacuum carpet']

TRAIN_TASKS = \
['Walk through', 'Start computer', 'Clean floor', 'Do taxes', 'Make coffee', 'Have snack', 'Get ready to leave', 'Put toys away', 'Arrange furniture', 'Turn on TV', 'Let baby learn how to walk', 'Watch fly', 'Pet dog', 'Call family member with skype application', 'Shut off alarm', 'Place centerpiece', 'Watch movie', 'Story reading time', 'Sit', 'Surf net', 'Play games', 'Wipe down baseboards please', 'Have conversation with boyfriend', 'Getting  dresses', 'Rearrange photo frames', 'Pull up carpet', 'Social media  checks', 'Turn on light', 'Come in and leave home', 'Re arrange office', 'Enjoy view out window', 'Shave', 'Drink', 'Shut front door', 'Get drink', 'File documents', 'Rain welcome', 'Put umbrella away', 'Turn on lights', 'Put away shoes', 'Change toilet paper roll', 'Wash hands', 'Sit in chair', 'Print out papers', 'Greet people', 'Check email', 'Put down bags', 'Surf internet', 'Study', 'Oil dining room', 'Print out document', 'Open door', 'Pick up', 'Add paper to printer', 'Take shower', 'Bring dirty plate to sink', 'Write report', 'Take off coat', 'Browse computer', 'Sent email', 'File expense reports', 'Change TV channel', 'Throw away newspaper', 'Put away jackets', 'Turking', 'Eat dinner', 'Answer emails', 'Clean sink', 'Open bathroom window', 'Get dressed', 'Get ready for school', 'Put shoes in shoe rack', 'Carry groceries to kitchen', 'Straighten pictures on wall', 'Take dishes out of dishwasher', 'Close door', 'Spread table with appropriate supplies', 'Sweep hallway please', 'Get toilet paper', 'Write  school paper', 'Hang with friends', 'Wash clothes', 'Juggling', 'Wash dishes with dishwasher', 'Read book', 'Wipe down counter', 'Turn on computer', 'Push in dining room chair', 'Turn night light on', 'Grab some juice', 'Settle in', 'Lock door', 'Prepare Dinner', 'Toast bread', 'Clean', 'Do dishes', 'Do laundry', 'Clean  mirror', 'Admire art', 'Relax on sofa', 'Answer door', 'Movie', 'De-wrinkle sheet', 'Set up buffet area', 'Wash sink', 'Sleep', 'Get glass of water', 'Hang car keys', 'Pick up cat hair', 'Play on laptop', 'Unload dishwasher', 'Get some water', 'Send  email', 'Shop', 'Read news', 'Take off shoes', 'Greet guests', 'Shampoo hair', 'Mop floor', 'Work', 'Watch  TV', 'vacuum carpet', 'Get out dish', 'Wake kids up', 'Take nap', 'Put on coat', 'Write an email', 'Homework', 'Do work on computer', 'Set mail on table', 'Computer work', 'Put away clean clothes', 'Pick up dirty dishes', 'Take off outerwear', 'Change TV channels', 'Gaze out window', 'Eat', 'Light candles', 'Clean bathroom', 'Hang up jacket', 'Put on glasses', 'Hang up car keys', 'Sew', 'Get ready for day', 'Sort laundry', 'Raise blinds', 'Wash dishes by hand', 'Look in refrigerator', 'Change light', 'Relax', 'Put on coat and shoes', 'Added meat to freezer', 'Text friends while sitting on couch', 'Clean screen', 'Keep an eye on stove as something is cooking', 'Turn light off', 'Watch youtube', 'Turn off light', 'Clean room', 'Come home', 'Research', 'Set up table', 'Open curtains', 'Make toast', 'Check appearance in mirror', 'Go to toilet', 'Look out window', 'Chop vegetables', 'Open front door', 'Get something to drink', 'Shredding', 'Load dishwasher', 'Pet cat', 'Clean dishes', 'Use toilet', 'Playing video game', 'Clean toilet', 'Manage emails', 'Organize', 'Surf web for money legitimate making opportunities', 'Bring me red cookbook', 'Cut bread', 'Clean mirror', 'Write story', 'Deficate', 'Read magazine', 'Use computer', 'Put away keys', 'Cook some food', 'Cutting', 'Pick up spare change on dresser', 'Rotate stock in refrigerator', 'Wash dishes', 'Shred receipts', 'Straighten paintings on wall', 'Open window', 'Watch TV', 'Plug in nightlight', 'Put new books in shelves', 'Say goodbye to guests leaving', 'Put in chair', 'Pick up phone', 'Put groceries in Fridge', 'Enter home']

TEST_TASKS = \
['Put mail in mail organizer', 'Dry soap bottles', 'Try yourself off', 'Write book', 'Grab things', 'Do work', 'Leave home', 'Eat snacks and drink tea', 'Read on sofa', 'Prepare pot of boiling water', 'Wash dirty dishes', 'Apply lotion', 'Pick up toys', 'Turn on radio', 'Collect napkin rings', 'Put away groceries', 'Vacuum', 'Breakfast', 'Push in desk chair', 'Draw picture', 'Take jacket off', 'Switch on lamp', 'Find dictionary', 'Pay bills', 'Style hair', 'Walk to room', 'Change clothes', 'Hang keys', 'Take shoes off', 'Look at painting', 'Eat cereal', 'Decorate it', 'Unload various items from pockets and place them in bowl on table', 'Play musical chairs', 'Read yourself to sleep', 'Turn off TV', 'Make bed', 'Compute', 'Dust', 'Browse internet', 'Go to sleep', 'Put clothes away', 'Organize pantry', 'Do homework', 'Use laptop', 'Entertain', 'Hand washing', 'Tale off shoes', 'Type up document', 'Read newspaper', 'Wash face', 'Put on your shoes', 'Make popcorn', 'Wash monitor', 'Read', 'Draft home', 'Fix snack', 'Look at mirror', 'Restock', 'Push all chairs in', 'Listen to music', 'Read to child', 'Put alarm clock in bedroom', 'Put away toys', 'Change sheets and pillow cases', 'Watch  horror  movie', 'Iron shirt', 'Get glass of milk', 'Hang pictures', 'Scrubbing living room tile floor is once week activity for me', 'Paint ceiling', 'Wipe down sink', 'Keep cats inside while door is open', 'Throw away paper', 'Turn on TV with remote', 'Keep cats out of room', 'Wash teeth', 'Feed me', 'Complete surveys on amazon turk', 'Put out flowers', 'Receive credit card', 'Arrange folders', 'Brush teeth', 'Put up decoration', 'Give milk to cat', 'Organize closet', 'Empty dishwasher and fill dishwasher', 'Eat cheese']