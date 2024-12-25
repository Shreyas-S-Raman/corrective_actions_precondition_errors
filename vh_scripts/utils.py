
def prepare_script(graph, script_file_name, execute=True):
    try:
        # -- open the file containing the plan sequence:
        script_file = open(script_file_name, 'r')
    except Exception:
        # -- run default/example script (putting salmon into the microwave)
        # NOTE: taken from here: http://virtual-home.org/documentation/master/get_started/get_started.html#generating-videos:
        script_file = open('~LLM_Planner/vh_scripts/test_script.txt', 'r')
    #endtry

   # -- script :- list of all instructions parsed from the text file
    script = []

    # -- dictionary mapping object names to their ids in the scene graph:
    objects = {}

    for L in script_file.readlines():
        # -- command/instruction will be in format [ACTION] <OBJECT_1> (ID_1) ... <OBJECT_n> (ID_n);
        #       we should take the object name and find the ID for it in the current scene.
        unparsed_instruction = L.split(' ')

        action = unparsed_instruction[0]

        parsed_instruction = ('<char0> ' if execute else '') + action

        # -- skip_step :- if a certain line refers to an object which is not in the scene,
        #       we will omit writing that instruction to the script (list of instructions)
        skip_step = False

        # -- parse through all parts for all objects:

        for x in range(1, len(unparsed_instruction)):
            # -- we take the string...
            string = unparsed_instruction[x]

            #... and see if it is enclosed by the appropriate symbols:
            if string.startswith('<') and string.endswith('>'):

                # -- remove surrounding symbols:
                parsed_object = string[1:-1]

                # -- remove any underscores from the name,
                #       as they seem to not have any in 'class_name' field of nodes:
                parsed_object = parsed_object.replace('_', '')

                #print(parsed_object)

                # -- following the salmon-microwave example from above, we will build a dictionary
                #       connecting object names to their ID in the scene:
                if parsed_object not in objects:
                    for node in graph['nodes']:
                        if node['class_name'] == parsed_object and 'id' in node and isinstance(node['id'], int):
                            objects[parsed_object] = node['id']
                            break

                # -- if we have indeed found the right id for the object, we will append it to the prototype instruction:
                if parsed_object in objects:
                    parsed_instruction += ' ' + '<' + parsed_object + '>' + ' ' + '(' + str(objects[parsed_object]) + ')'
                else:
                    # -- in this case, we found an object that's strangely not in the scene, so we cannot write this instruction:
                    skip_step = True
                    break

        # -- add the parsed instruction to the instruction set script:
        if not skip_step:
            script.append(parsed_instruction)

    script_file.close()

    return script, objects.keys()
