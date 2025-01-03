import pdb
import time
import queue
from typing import Optional
import evolving_graph.common as common
from evolving_graph.environment import *
from evolving_graph.scripts import Action, ScriptLine, Script


# ExecutionInfo
###############################################################################

class ExecutionInfo(object):

    def __init__(self):
        self.messages = []
        self.message_params = []
        self.current_line = None

    def error(self, msg: str, err_type:str, error_params):

        params = tuple(error_params.values())[:-1] if 'error_state' not in error_params else tuple(error_params.values())[:-2]

        self.messages.append(msg.format(*params) + ' when executing "' + self.current_line_info() + '"')
        
        
        for k in error_params:
            error_params[k] = '{}'.format(*(error_params[k],))
        
        
        error_params['type'] = err_type
        self.message_params.append(error_params)


    def set_current_line(self, sl: ScriptLine):
        self.current_line = sl

    def script_object_found_error(self, so: ScriptObject):
        self.error(str(so) + " cannot be found")

    def object_found_error(self):
        if self.current_line is not None:
            self.script_object_found_error(self.current_line.object())

    def current_line_info(self):
        return '' if self.current_line is None else str(self.current_line)

    def get_error_string(self):
        return ','.join(self.messages)
    
    def get_error_params(self):
        return self.message_params


# ActionExecutor-s
###############################################################################

class ActionExecutor(object):

    @abstractmethod
    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        """
        :param script: future script, scripts.Script object containing script lines
            yet to be executed, has at least one item
        :param state: current state, environment.EnvironmentState object
        :param info: information (typically an error) about execution
        :return: enumerates possible states after execution of script_line
        """
        pass


class UnknownExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        raise ExecutionException("Execution of {0} is not supported", script[0].action)


class JoinedExecutor(ActionExecutor):

    def __init__(self, *args):
        self.executors = args

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        for e in self.executors:
            for s in e.execute(script, state, info):
                yield s


class WalkExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        current_obj = current_line.object()

        # select objects based on current_obj
        for node in state.select_nodes(current_obj):
            node_room = _get_room_node(state, node)
            if self.check_walk(state, node, info) and node_room is not None:

                changes = [DeleteEdges(CharacterNode(),
                                       [Relation.INSIDE, Relation.CLOSE, Relation.FACING],
                                       AnyNode(), delete_reverse=True),
                           AddEdges(CharacterNode(), Relation.CLOSE, BoxObjectNode(node), add_reverse=True),
                           AddEdges(CharacterNode(), Relation.CLOSE, BodyNode(), add_reverse=True),
                           AddEdges(CharacterNode(), Relation.INSIDE, NodeInstance(node_room)),
                           AddEdges(CharacterNode(), Relation.CLOSE, NodeInstance(node), add_reverse=True)
                     ]

                # close to object in hands
                char_node = _get_character_node(state)
                char_room = _get_room_node(state, char_node)
                nodes_in_hands = _find_nodes_from(state, char_node, relations=[Relation.HOLDS_LH, Relation.HOLDS_RH])
                for node_in_hands in nodes_in_hands:
                    changes.append(DeleteEdges(NodeInstance(node_in_hands), [Relation.INSIDE, Relation.CLOSE, Relation.FACING], AnyNode(), delete_reverse=True))

                for node_in_hands in nodes_in_hands:
                    changes.append(AddEdges(CharacterNode(), Relation.CLOSE, NodeInstance(node_in_hands), add_reverse=True))
                    changes.append(AddEdges(NodeInstance(node_in_hands), Relation.INSIDE, NodeInstance(char_room)))

                # close to all objects on node
                if Property.SURFACES in node.properties:
                    changes.append(AddEdges(CharacterNode(), Relation.CLOSE, ObjectOnNode(node), add_reverse=True))

                yield state.change_state(changes, node, current_obj)

    def check_walk(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)
        if State.SITTING in char_node.states or State.LYING in char_node.states:
            info.error('{} is sitting', 'already_sitting', {'character': char_node, 'subtype':None})
            return False
        
        char_room = _get_room_node(state, char_node)
        node_room = _get_room_node(state, node)

        doors = state.get_nodes_by_attr('class_name', 'door')
        doorjambs = state.get_nodes_by_attr('class_name', 'doorjamb')

        if node in (doors + doorjambs):
            # door that connect the char_room
            if state.evaluate(ExistsRelation(NodeInstance(node), Relation.BETWEEN, NodeInstanceFilter(char_room))) or \
                state.evaluate(ExistsRelation(NodeInstance(char_room), Relation.BETWEEN, NodeInstanceFilter(node))):
                return True

        # the return list is in reverse orders, living room --> door.1 --> dining room --> door.181 --> bathroom --> door.16 --> living room
        # suppose all the doors are closed, the return list would be like [door.16, door.181, door.1]
        closed_doors = _check_closed_doors(state, char_room, node_room)

        if closed_doors is None:
            info.error('No path between between {} and {}', 'no_path', {'char_room':char_room, 'target_room': node_room, 'subtype':None})
            return False
        if len(closed_doors) > 0:
            # walk to the nearest closed door is fine
            if node.id != closed_doors[-1].id:
                info.error("Door(s) {} between {} and {} is (are) closed", 'door_closed', {'door_list':', '.join([str(d) for d in closed_doors]), 'char_room':
                            char_room, 'target_room': node_room, 'subtype': None})
                return False

        return True


class _FindExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        current_obj = current_line.object()

        # select objects based on current_obj
        for node in state.select_nodes(current_obj):
            if self.check_find(state, node, info):
                yield state.change_state(
                    [DeleteEdges(CharacterNode(), [Relation.FACING], AnyNode()), 
                     AddEdges(CharacterNode(), Relation.CLOSE, NodeInstance(node), add_reverse=True)],
                    node,
                    current_obj
                )

    def check_find(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        if not _is_character_close_to(state, node):
            char_node = _get_character_node(state)
            info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj': node, 'subtype':None})
            return False
        
        return True


_walk_find_executor = JoinedExecutor(WalkExecutor(), _FindExecutor())
_only_find_executor = _FindExecutor()


class FindExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        current_obj = current_line.object()
        for node in state.select_nodes(current_obj):
            char_node = _get_character_node(state)

            if state.evaluate(ExistsRelation(NodeInstance(node), Relation.ON, NodeInstanceFilter(char_node))):
                return _only_find_executor.execute(script, state, info)
            elif Property.BODY_PART in node.properties:
                return _only_find_executor.execute(script, state, info)
            elif _is_character_close_to(state, node):
                return _only_find_executor.execute(script, state, info)
            elif State.SITTING in char_node.states or State.LYING in char_node.states:
                return _only_find_executor.execute(script, state, info)
            else:
                return _walk_find_executor.execute(script, state, info)
        info.error('Could not find object {}'.format(current_obj.name), 'not_find', {'obj':current_obj.name, 'subtype':None})


class GreetExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        else:
            if self.check_if_person(state, node, info):
                yield state.change_state([])

    def check_if_person(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        if Property.PERSON not in node.properties:
            info.error('{} is not person', 'invalid_action', {'obj':node, 'subtype':'not a person'})
            return False
        return True


class SitExecutor(ActionExecutor):

    _MAX_OCCUPANCIES = {
        'couch': 4,
        'bed': 4,
        'chair': 1,
        'loveseat': 2,
        'sofa': 4,
        'toilet': 1,
        'pianobench': 2,
        'bench': 2
    }

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_sittable(state, node, info):
            char_node = _get_character_node(state)
            new_char_node = char_node.copy()
            new_char_node.states.discard(State.LYING)
            new_char_node.states.add(State.SITTING)
            
            yield state.change_state(
                [AddEdges(CharacterNode(), Relation.ON, NodeInstance(node)),
                 AddEdges(CharacterNode(), Relation.FACING, RelationFrom(node, Relation.FACING)),
                 ChangeNode(new_char_node)]
            )

    def check_sittable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)

        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj': node, 'subtype':None})
            return False
        if State.SITTING in char_node.states:
            info.error('{} is sitting', 'already_sitting', {'character': char_node, 'subtype':None})
            return False
        if Property.SITTABLE not in node.properties:
            info.error('{} is not sittable', 'invalid_action', {'obj':node,'subtype': 'not sittable'})
            return False
        max_occupancy = self._MAX_OCCUPANCIES.get(node.class_name, 1)
        if state.evaluate(CountRelations(AnyNode(), Relation.ON, NodeInstanceFilter(node),
                                         min_value=max_occupancy)):
            info.error('Too many things on {}', 'max_occupancy', {'obj': node, 'subtype':None})
            return False

        return True


class StandUpExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        info.set_current_line(script[0])
        char_node = _get_character_node(state)
        if State.SITTING in char_node.states or State.LYING in char_node.states:
            new_char_node = char_node.copy()
            new_char_node.states.discard(State.SITTING)
            new_char_node.states.discard(State.LYING)
            yield state.change_state([ChangeNode(new_char_node)])
        else:
            info.error('{} is not sitting', 'missing_step', {'character': char_node,'subtype':'not sitting'})
            

class GrabExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        else:
            new_relation = self.check_grabbable(state, node, info)
            if new_relation is not None:
                char_node = _get_character_node(state)
                char_room = _get_room_node(state, char_node)
                changes = [DeleteEdges(NodeInstance(node), [Relation.ON, Relation.INSIDE, Relation.CLOSE], AnyNode(), delete_reverse=True),
                           AddEdges(CharacterNode(), Relation.CLOSE, NodeInstance(node), add_reverse=True), 
                           AddEdges(CharacterNode(), new_relation, NodeInstance(node)), 
                           AddEdges(NodeInstance(node), Relation.INSIDE, NodeInstance(char_room))]
                new_close, relation = _find_first_node_from(state, node, [Relation.ON, Relation.INSIDE, Relation.CLOSE])
                if new_close is not None:
                    changes += [AddEdges(CharacterNode(), Relation.CLOSE, NodeInstance(new_close), add_reverse=True),
                                AddExecDataValue((Action.GRAB, node.id), (new_close, relation))]
                yield state.change_state(changes)

    def check_grabbable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo) -> Optional[Relation]:
        if Property.GRABBABLE not in node.properties and node.class_name not in ['water', 'child']:
            info.error('{} is not grabbable', 'invalid_action', {'obj':node,'subtype':'not grabbale'})
            return None
        if not _is_character_close_to(state, node):
            char_node = _get_character_node(state)
            info.error('{} is not close to {}', 'proximity', {'character':char_node, 'obj':node, 'subtype':None})
            return None
        if _is_inside(state, node):
            info.error('{} is inside other closed thing', 'internally_contained', {'obj':node, 'subtype':None})
            return None
        new_relation = _find_free_hand(state)
        if new_relation is None:
            char_node = _get_character_node(state)
            info.error('{} does not have a free hand', 'hands_full', {'character':char_node, 'subtype':None})
            return None
        return new_relation


class OpenExecutor(ActionExecutor):

    def __init__(self, close: bool):
        """
        :param close: False: OpenExecutor, True: CloseExecutor

        """
        self.close = close

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_openable(state, node, info):
            new_node = node.copy()
            new_node.states.discard(State.OPEN if self.close else State.CLOSED)
            new_node.states.add(State.CLOSED if self.close else State.OPEN)
            yield state.change_state([ChangeNode(new_node)])

    def check_openable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        
        if Property.CAN_OPEN not in node.properties and node.class_name not in ["desk", "window"]:
            info.error('{} can not be opened', 'invalid_action', {'obj':node,'subtype':'not openable'})
            return False

        if not _is_character_close_to(state, node):
            char_node = _get_character_node(state)
            info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj': node, 'subtype':None})
            return False
        
        if not self.close and _find_free_hand(state) is None:
            char_node = _get_character_node(state)
            info.error('{} does not have a free hand', 'hands_full', {'character': char_node, 'subtype':None})
            return False

        s = State.OPEN if self.close else State.CLOSED
        s_err = State.CLOSED if self.close else State.OPEN

        if s not in node.states:
            info.error('{} is not {}', 'unflipped_boolean_state', {'obj': node, 'state' : s.name.lower(), 'error_state': s_err.name.lower(), 'subtype':None})
            return False

        if not self.close and State.ON in node.states:
            info.error('{} is still {}', 'unflipped_boolean_state', {'obj':node,'state':'on', 'error_state':'on', 'subtype':None})
            return False
        return True


class PutExecutor(ActionExecutor):

    def __init__(self, relation: Relation):
        """
        :param relation: Relation.ON: PutExecutor, Relation.INSIDE: PutInExecutor:

        """
        self.relation = relation

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        src_node = state.get_state_node(current_line.object())
        dest_node = state.get_state_node(current_line.subject())
        if src_node is None or dest_node is None:
            info.script_object_found_error(current_line.object() if src_node is None else current_line.subject())
        elif _check_puttable(state, src_node, dest_node, self.relation, info):
            yield state.change_state(
                [DeleteEdges(CharacterNode(), [Relation.HOLDS_LH, Relation.HOLDS_RH], NodeInstance(src_node)),
                 AddEdges(CharacterNode(), Relation.CLOSE, NodeInstance(dest_node), add_reverse=True),
                 AddEdges(NodeInstance(src_node), Relation.CLOSE, NodeInstance(dest_node), add_reverse=True),
                 AddEdges(NodeInstance(src_node), self.relation, NodeInstance(dest_node)),
                 ClearExecDataKey((Action.GRAB, src_node.id))]
            )


class PutBackExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        src_node = state.get_state_node(current_line.object())
        if src_node is None:
            info.object_found_error()
        else:
            (dest_node, relation) = state.executor_data.get((Action.GRAB, src_node.id), (None, None))
            if dest_node is None:
                char_node = _get_character_node(state)
                info.error('{} not grabbed', 'not_holding', {'character': char_node,'obj':src_node, 'subtype':None})
            else:
                dest_node = state.get_node(dest_node.id)
                if _check_puttable(state, src_node, dest_node, relation, info):
                    yield state.change_state(
                        [DeleteEdges(CharacterNode(), [Relation.HOLDS_LH, Relation.HOLDS_RH], NodeInstance(src_node)),
                         AddEdges(CharacterNode(), Relation.CLOSE, NodeInstance(dest_node), add_reverse=True),
                         AddEdges(NodeInstance(src_node), Relation.CLOSE, NodeInstance(dest_node), add_reverse=True),
                         AddEdges(NodeInstance(src_node), relation, NodeInstance(dest_node)),
                         ClearExecDataKey((Action.GRAB, src_node.id))]
                    )


def _check_puttable(state: EnvironmentState, src_node: GraphNode, dest_node: GraphNode, relation: Relation,
                    info: ExecutionInfo):
    hand_rel = _find_holding_hand(state, src_node)
    if hand_rel is None:
        char_node = _get_character_node(state)
        info.error('{} is not holding {}', 'not_holding', {'character': char_node, 'obj':src_node, 'subtype':None})
        return False
    if not _is_character_close_to(state, dest_node):
        char_node = _get_character_node(state)
        info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj':dest_node, 'subtype':None})
        return False
    if relation == Relation.INSIDE:
        if Property.CAN_OPEN not in dest_node.properties or State.OPEN in dest_node.states:
            return True
        else:
            info.error('{} is not open or is not openable', 'invalid_action', {'obj' : dest_node, 'subtype': 'is not openable'})
            return False
    return True


class SwitchExecutor(ActionExecutor):

    def __init__(self, switch_on: bool):
        """
            SwitchOnExecutor: True
            SwitchOffExecutor: False
        """
        self.switch_on = switch_on

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_switchable(state, node, info):
            new_node = node.copy()
            new_node.states.discard(State.OFF if self.switch_on else State.ON)
            new_node.states.add(State.ON if self.switch_on else State.OFF)
            yield state.change_state([ChangeNode(new_node)])

    def check_switchable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        s = State.OFF if self.switch_on else State.ON
        s_err = State.ON if self.switch_on else State.OFF

        if Property.HAS_SWITCH not in node.properties:
            info.error('{} does not have a switch', 'invalid_action', {'obj': node, 'subtype':'does not have a switch'})
            return False
        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character' : _get_character_node(state), 'obj' : node, 'subtype':None})
            return False
        #if _find_free_hand(state) is None:
        #    info.error('{} does not have a free hand', _get_character_node(state))
        #    return False
        if s not in node.states:
            info.error('{} is {}', 'unflipped_boolean_state', {'obj' : node, 'state': 'not' + s.name.lower(), 'error_state': s_err.name.lower(), 'subtype':None})
            return False
        if self.switch_on and State.PLUGGED_OUT in node.states:
            info.error('{} is {}', 'unflipped_boolean_state', {'obj': node, 'state': 'unplugged', 'error_state': 'unplugged', 'subtype':None})
            return False
        return True


class DrinkExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_drinkable(state, node, info):
            yield state.change_state([])

    def check_drinkable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        if Property.DRINKABLE not in node.properties and Property.RECIPIENT not in node.properties:
            info.error('{} is not drinkable or not recipient', 'invalid_action', {'obj': node, 'subtype': 'not drinkable'})
            return False
        hand_rel = _find_holding_hand(state, node)
        if hand_rel is None:
            info.error('{} is not holding {}', 'not_holding', {'character':_get_character_node(state),'obj': node, 'subtype':None})
            return False
        return True


class TurnToExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_turn_to(state, node, info):
            yield state.change_state(
                [DeleteEdges(CharacterNode(), [Relation.FACING], AnyNode()), 
                 AddEdges(CharacterNode(), Relation.FACING, NodeInstance(node))]
            )

    def check_turn_to(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        return True


class LookAtExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_lookat(state, node, info):
            yield state.change_state([])

    def check_lookat(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)
        if not _is_character_face_to(state, node):
            info.error('{} does not face {}', 'not_facing', {'character':char_node, 'obj':node, 'subtype':None})
            return False

        return True


class WipeExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_wipe(state, node, info):
            new_node = node.copy()
            new_node.states.discard(State.DIRTY)
            new_node.states.add(State.CLEAN)
            yield state.change_state([ChangeNode(new_node)])

    def check_wipe(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)

        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj':node, 'subtype':None})
            return False

        #if Property.SURFACES not in node.properties:
        #    info.error('{} is not a surface', node)
        #    return False

        nodes_in_hands = _find_nodes_from(state, char_node, [Relation.HOLDS_RH, Relation.HOLDS_LH])
        if len(nodes_in_hands) == 0:
            info.error('{} does not hold anything in hands', 'not_holding_any', {'character': char_node, 'subtype':None})
            return 

        return True


class PutOnExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_puton(state, node, info):
            yield state.change_state([
                AddEdges(NodeInstance(node), Relation.ON, CharacterNode()),
                DeleteEdges(CharacterNode(), [Relation.HOLDS_LH, Relation.HOLDS_RH], NodeInstance(node)),
            ])

    def check_puton(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)
        nodes_in_hands = _find_nodes_from(state, char_node, relations=[Relation.HOLDS_LH, Relation.HOLDS_RH])
        
        if not any([n.id == node.id for n in nodes_in_hands]):
            info.error('{} is not holding {}', 'not_holding', {'character' : char_node, 'obj': node, 'subtype':None})
            return False
        if Property.CLOTHES not in node.properties:
            info.error('{} is not clothes', 'invalid_action', {'obj': node, 'subtype':'not clothes'})
            return False

        return True


class PutOffExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_putoff(state, node, info):
            yield state.change_state([
                DeleteEdges(NodeInstance(node), [Relation.ON], CharacterNode())
            ])

    def check_putoff(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)
        if not state.evaluate(ExistsRelation(NodeInstance(node), Relation.ON, NodeInstanceFilter(char_node))):
            info.error('{} is {}', 'unflipped_boolean_state', {'obj': node, 'state': 'not on' + char_node.class_name, 'error_state':  'off me', 'subtype':None})
            return False
        if Property.CLOTHES not in node.properties:
            info.error('{} is not clothes', 'invalid_action', {'obj': node, 'subtype': 'not clothes'})
            return False
        return True


class DropExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_drop(state, node, info):
            char_node = _get_character_node(state)
            char_room = _get_room_node(state, char_node)
            yield state.change_state(
                [DeleteEdges(CharacterNode(), [Relation.HOLDS_LH, Relation.HOLDS_RH], NodeInstance(node)),
                 AddEdges(NodeInstance(node), Relation.INSIDE, NodeInstance(char_room)),
                 ClearExecDataKey((Action.GRAB, node.id))]
            )

    def check_drop(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)
        nodes_in_hands = _find_nodes_from(state, char_node, relations=[Relation.HOLDS_LH, Relation.HOLDS_RH])
        if not any([n.id == node.id for n in nodes_in_hands]):
            info.error('{} is not holding {}', 'not_holding', {'character': char_node, 'obj': node, 'subtype':None})
            return False

        return True


class ReadExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_readable(state, node, info):
            yield state.change_state([])

    def check_readable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        if Property.READABLE not in node.properties:
            info.error('{} is not readable', 'invalid_action', {'obj':node, 'subtype': 'not readable'})
            return False
        hand_rel = _find_holding_hand(state, node)
        if hand_rel is None:
            info.error('{} is not holding {}', 'not_holding', {'character': _get_character_node(state), 'obj': node, 'subtype':None})
            return False
        return True


class TouchExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_reachable(state, node, info):
            yield state.change_state([])

    def check_reachable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character': _get_character_node(state), 'obj': node, 'subtype':None})
            return False
        if _is_inside(state, node):
            info.error('{} is inside other closed thing', 'internally_contained', {'obj': node, 'subtype':None})
            return False
        return True


class LieExecutor(ActionExecutor):

    _MAX_OCCUPANCIES = {
        'couch': 2,
        'bathtub': 2,
        'bed': 3,
        'loveseat': 2,
        'sofa': 2,
        'bench': 1
    }

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_lieable(state, node, info):
            char_node = _get_character_node(state)
            new_char_node = char_node.copy()
            new_char_node.states.discard(State.SITTING)
            new_char_node.states.add(State.LYING)
            yield state.change_state(
                [AddEdges(CharacterNode(), Relation.ON, NodeInstance(node)),
                 ChangeNode(new_char_node)]
            )

    def check_lieable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)
        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj': node, 'subtype':None})
            return False
        if State.LYING in char_node.states:
            info.error('{} is {}', 'unflipped_boolean_state', {'obj': char_node, 'state':'lying down', 'error_state':'lying down', 'subtype':None})
            return False
        if Property.LIEABLE not in node.properties:
            info.error('{} is not lieable', 'invalid_action', {'obj': node, 'subtype': 'lie down'})
            return False
        max_occupancy = self._MAX_OCCUPANCIES.get(node.class_name, 1)
        if state.evaluate(CountRelations(AnyNode(), Relation.ON, NodeInstanceFilter(node),
                                         min_value=max_occupancy)):
            info.error('Too many things on {}', 'max_occupancy', {'obj':node, 'subtype':None})
            return False
        return True


class PourExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        src_node = state.get_state_node(current_line.object())
        dest_node = state.get_state_node(current_line.subject())

        if src_node is None or dest_node is None:
            info.script_object_found_error(current_line.object() if src_node is None else current_line.subject())
        elif self._check_pourable(state, src_node, dest_node, info):
            changes = [AddEdges(NodeInstance(src_node), Relation.INSIDE, NodeInstance(dest_node))]
            if src_node.class_name == 'water':
                changes += [DeleteEdges(CharacterNode(), [Relation.HOLDS_LH, Relation.HOLDS_RH], NodeInstance(src_node))]
            yield state.change_state(changes)

    def _check_pourable(self, state: EnvironmentState, src_node: GraphNode, dest_node: GraphNode, info: ExecutionInfo):

        if Property.POURABLE not in src_node.properties and Property.DRINKABLE not in src_node.properties:
            info.error('{} is not pourable or drinkable', 'invalid_action', {'obj':src_node, 'subtype':'cannot be poured'})
            return False

        if Property.RECIPIENT not in dest_node.properties and dest_node.class_name not in ["hands_both", "sponge", "face"]:
            info.error('{} is not recipient', 'invalid_action', {'obj':dest_node, 'subtype': 'is not recipient'})
            return False

        hand_rel = _find_holding_hand(state, src_node)
        if hand_rel is None:
            info.error('{} is not holding {}', 'not_holding', {'character':_get_character_node(state), 'obj':src_node, 'subtype':None})
            return False

        char_node = _get_character_node(state)
        if not _is_character_close_to(state, dest_node):
            info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj':dest_node, 'subtype':None})
            return False
        
        return True


class TypeExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_typeable(state, node, info):
                yield state.change_state([])
        
    def check_typeable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)

        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj': node, 'subtype':None})
            return False
        
        if node.class_name == 'keyboard':
            return True

        if Property.HAS_SWITCH not in node.properties:
            info.error('{} does not have switch', 'invalid_action', {'obj': node, 'subtype':'does not have a switch'})
            return False
        
        return True


class WatchExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_watchable(state, node, info):
            yield state.change_state([])

    def check_watchable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        char_node = _get_character_node(state)
        char_room = _get_room_node(state, char_node)
        node_room = _get_room_node(state, node)
        
        if Property.LOOKABLE not in node.properties:
            info.error('{} not lookable', 'invalid_action', {'obj':node, 'subtype':'cannot be looked at'})
            return False
        if node_room.id != char_room.id:
            info.error('char room {} is not node room {}', 'invalid_room', {'char_room': char_room, 'target_room': node_room, 'subtype':None})
            return False
        if not _is_character_face_to(state, node):
            info.error('{} does not face {}', 'not_facing', {'character': char_node, 'obj':node, 'subtype':None})
            return False
        if node.class_name != 'computer' and (State.SITTING in char_node.states or State.LYING in char_node.states) and not state.evaluate(ExistsRelation(CharacterNode(), Relation.FACING, NodeInstanceFilter(node))):
            info.error('{} is not facing {} while sitting', 'not_facing', {'character':char_node, 'obj':node, 'subtype':None})
            return False
        if _is_inside(state, node):
            info.error('{} is inside other closed thing', 'internally_contained', {'obj':node, 'subtype':None})
        return True


class MoveExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        action_name = current_line.action.name.lower()
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        else:
            new_relation = self.check_movable(state, node, info, action_name)
            if new_relation is not None:
                yield state.change_state([])

    def check_movable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo, action_name: str) -> Optional[Relation]:

        if Property.MOVABLE not in node.properties and (action_name != 'push' and node.class_name != 'button') and node.class_name not in ['chair', 'curtain']:
            info.error('{} is not movable', 'invalid_action', {'obj':node,'subtype': 'not movable'})
            return None
        if not _is_character_close_to(state, node):
            char_node = _get_character_node(state)
            info.error('{} is not close to {}', 'proximity', {'character': char_node, 'obj':node, 'subtype':None})
            return None
        if _is_inside(state, node):
            info.error('{} is inside other closed thing', 'internally_contained', {'obj':node, 'subtype':None})
            return None
        new_relation = _find_free_hand(state)
        if new_relation is None:
            char_node = _get_character_node(state)
            info.error('{} does not have a free hand', 'hands_full', {'character':char_node, 'subtype':None})
            return None
        return new_relation


class WashExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_washable(state, node, info):
            new_node = node.copy()
            new_node.states.discard(State.DIRTY)
            new_node.states.add(State.CLEAN)
            yield state.change_state([ChangeNode(new_node)])

    def check_washable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        
        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character':_get_character_node(state), 'obj':node, 'subtype':None})
            return False

        return True


class SqueezeExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_squeezable(state, node, info):
            yield state.change_state([])

    def check_squeezable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        
        if _find_free_hand(state) is None:
            info.error('{} does not have a free hand', 'hands_full', {'character':_get_character_node(state), 'subtype':None})
            return False
        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity',{'character':_get_character_node(state), 'obj':node, 'subtype':None})
            return False

        squeezable_objects = ['cleaning_solution', 'tooth_paste', 'shampoo', 
                'food_peanut_butter', 'dish_soap', 'soap', 'towel', 'rag', 'paper', 'sponge', 
                'food_lemon', 'check']
        if Property.CLOTHES not in node.properties and node.class_name not in squeezable_objects:
            info.error('{} is not clothes', 'invalid_action', {'obj':node,'subtype':'not a piece of clothing'})
            return False

        return True


class PlugExecutor(ActionExecutor):

    def __init__(self, plug_in):
        """
            PlugInExecutor: True
            PlugOutExecutor: False
        """
        self.plug_in = plug_in

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_plugable(state, node, info):
            new_node = node.copy()
            new_node.states.discard(State.PLUGGED_OUT if self.plug_in else State.PLUGGED_IN)
            new_node.states.add(State.PLUGGED_IN if self.plug_in else State.PLUGGED_OUT)
            yield state.change_state([ChangeNode(new_node)])

    def check_plugable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):
        s = State.PLUGGED_OUT if self.plug_in else State.PLUGGED_IN
        s_err = State.PLUGGED_IN if self.plug_in else State.PLUGGED_OUT

        if Property.HAS_PLUG not in node.properties:
            info.error('{} does not have a plug', 'invalid_action', {'obj':node,'subtype':'does not have a plug'})
            return False
        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character':_get_character_node(state), 'obj':node, 'subtype':None})
            return False
        if _find_free_hand(state) is None:
            info.error('{} does not have a free hand', 'hands_full', {'character':_get_character_node(state), 'subtype':None})
            return False
        if s not in node.states:
            info.error('{} is {}', 'unflipped_boolean_state', {'obj':node, 'state': 'not' + s.name.lower(), 'error_state': s_err.name.lower(), 'subtype':None})
            return False
        if not self.plug_in and State.ON in node.states:
            info.error('{} is still {}', 'unflipped_boolean_state', {'obj':node, 'state':'on', 'error_state':'on', 'subtype':None})
        return True
    

class CutExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_cuttable(state, node, info):
            yield state.change_state([])

    def check_cuttable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):

        if _find_free_hand(state) is None:
            info.error('{} does not have a free hand', 'hands_full', {'character':_get_character_node(state), 'subtype':None})
            return False
        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character':_get_character_node(state), 'obj':node, 'subtype':None})
            return False
        if Property.EATABLE not in node.properties:
            info.error('{} is not eatable', 'invalid_action', {'obj':node, 'subtype':'not eatable'})
            return False
        if Property.CUTTABLE not in node.properties:
            info.error('{} is not cuttable', 'invalid_action', {'obj':node, 'subtype':'not cuttable'})
            return False

        char_node = _get_character_node(state)
        holding_nodes = _find_nodes_from(state, char_node, [Relation.HOLDS_LH, Relation.HOLDS_RH])
        if not any(['knife' in node.class_name for node in holding_nodes]):
            info.error('{} is not holding a {}', 'not_holding', {'character':_get_character_node(state), 'obj':'knife', 'subtype':None})
            return False

        return True


class EatExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):
        current_line = script[0]
        info.set_current_line(current_line)
        node = state.get_state_node(current_line.object())
        if node is None:
            info.object_found_error()
        elif self.check_eatable(state, node, info):
            yield state.change_state([])

    def check_eatable(self, state: EnvironmentState, node: GraphNode, info: ExecutionInfo):

        if not _is_character_close_to(state, node):
            info.error('{} is not close to {}', 'proximity', {'character':_get_character_node(state), 'obj':node, 'subtype':None})
            return False
            
        if Property.EATABLE in node.properties:
            return True
        else:
            nodes_in_objs = _find_nodes_to(state, node, relations=[Relation.ON])
            if len(nodes_in_objs) == 0:
                info.error('{} is not eatable', 'invalid_action', {'obj':node,'subtype':'not eatable'})
                return False
            elif any([Property.EATABLE in node.properties for node in nodes_in_objs]):
                return True
            else:
                info.error('none of object on {} is eatable', 'invalid_action',  {'obj':node, 'subtype':'not eatable'})
                return False


class SleepExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):

        info.set_current_line(script[0])
        char_node = _get_character_node(state)
        if State.LYING not in char_node.states and State.SITTING not in char_node.states:
            info.error('{} is not lying or sitting', 'missing_step', {'character': char_node, 'subtype': 'not lying down or sitting'})
        else:
            yield state.change_state([])


class WakeUpExecutor(ActionExecutor):

    def execute(self, script: Script, state: EnvironmentState, info: ExecutionInfo):

        info.set_current_line(script[0])
        char_node = _get_character_node(state)
        if State.LYING not in char_node.states and State.SITTING not in char_node.states:
            info.error('{} is not lying or sitting', 'missing_step', {'character': char_node, 'subtype':'not lying down or sitting'})
        else:
            yield state.change_state([])


PointAtExecutor = LookAtExecutor


# General checks and helpers


def _is_character_close_to(state: EnvironmentState, node: Node):
    if state.evaluate(ExistsRelation(CharacterNode(), Relation.CLOSE, NodeInstanceFilter(node))):
        return True
    # loose rule
    for close_node in state.get_nodes_from(_get_character_node(state), Relation.CLOSE):
        if state.evaluate(ExistsRelation(NodeInstance(close_node), Relation.CLOSE, NodeInstanceFilter(node))):
            return True
        if state.evaluate(ExistsRelation(NodeInstance(node), Relation.ON, NodeInstanceFilter(close_node))):
            return True
    return False


def _is_character_face_to(state: EnvironmentState, node: Node):
    if state.evaluate(ExistsRelation(CharacterNode(), Relation.FACING, NodeInstanceFilter(node))):
        return True
    for face_node in state.get_nodes_from(_get_character_node(state), Relation.FACING):
        if state.evaluate(ExistsRelation(NodeInstance(face_node), Relation.FACING, NodeInstanceFilter(node))):
            return True
    return False


def _get_character_node(state: EnvironmentState):
    chars = state.get_nodes_by_attr('class_name', 'character')
    return None if len(chars) == 0 else chars[0]


def _get_room_node(state: EnvironmentState, node: Node):
    if node.category == 'Rooms':
        return node
    for n in state.get_nodes_from(node, Relation.INSIDE):
        if n.category == 'Rooms':
            return n
    return None


def _find_nodes_to(state: EnvironmentState, node: Node, relations: List[Relation]):
    nodes = []
    for src_node in AnyNode().enumerate(state):
        for r in relations:
            nl = state.get_nodes_from(src_node, r)
            if node in nl:
                nodes.append(src_node)
    return nodes

def _find_nodes_from(state: EnvironmentState, node: Node, relations: List[Relation]):
    nodes = []
    for r in relations:
        nl = state.get_nodes_from(node, r)
        nodes += nl
    return nodes


def _find_first_node_from(state: EnvironmentState, node: Node, relations: List[Relation]):
    for r in relations:
        for n in state.get_nodes_from(node, r):
            if n.category != 'Rooms':
                return n, r
    return None, None


def _find_free_hand(state: EnvironmentState):
    if not state.evaluate(ExistsRelation(CharacterNode(), Relation.HOLDS_RH, AnyNodeFilter())):
        return Relation.HOLDS_RH
    if not state.evaluate(ExistsRelation(CharacterNode(), Relation.HOLDS_LH, AnyNodeFilter())):
        return Relation.HOLDS_LH
    return None


def _find_holding_hand(state: EnvironmentState, node: Node):
    if state.evaluate(ExistsRelation(CharacterNode(), Relation.HOLDS_RH, NodeInstanceFilter(node))):
        return Relation.HOLDS_RH
    if state.evaluate(ExistsRelation(CharacterNode(), Relation.HOLDS_LH, NodeInstanceFilter(node))):
        return Relation.HOLDS_LH
    return None


def _is_inside(state: EnvironmentState, node: Node):
    return state.evaluate(ExistsRelation(NodeInstance(node), Relation.INSIDE,
                                         NodeConditionFilter(And(NodeAttrIn(State.CLOSED, 'states'),
                                                                 Not(IsRoomNode())))))


def _create_walkable_graph(state: EnvironmentState):
    doors = state.get_nodes_by_attr('class_name', 'door')
    doorjambs = state.get_nodes_by_attr('class_name', 'doorjamb')
    adj_lists = {}
    for door_node in doors:
        door_rooms = state.get_nodes_from(door_node, Relation.BETWEEN)
        if len(door_rooms) > 1:
            adj_lists.setdefault(door_rooms[0].id, []).append((door_rooms[1].id, door_node.id))
            adj_lists.setdefault(door_rooms[1].id, []).append((door_rooms[0].id, door_node.id))
    for dj_node in doorjambs:
        dj_rooms = state.get_nodes_from(dj_node, Relation.BETWEEN)
        if len(dj_rooms) > 1:
            adj_lists.setdefault(dj_rooms[0].id, []).append((dj_rooms[1].id, dj_node.id))
            adj_lists.setdefault(dj_rooms[1].id, []).append((dj_rooms[0].id, dj_node.id))
    return adj_lists


def _check_closed_doors(state: EnvironmentState, room1: GraphNode, room2: GraphNode):
    graph_adj_lists = _create_walkable_graph(state)
    bfs_prev = BFS_check_closed(state, graph_adj_lists, room1.id)
    if room2.id in bfs_prev:
        return []
    bfs_prev = BFS(graph_adj_lists, room1.id)
    if room2.id not in bfs_prev:
        return None  # No path!
    closed_between = []
    current_id = room2.id
    while current_id != room1.id:
        next_id, door_id = bfs_prev[current_id]
        if next_id is None:
            break
        door_node = state.get_node(door_id)
        if State.CLOSED in door_node.states:
            closed_between.append(door_node)
        current_id = next_id
    return closed_between


def BFS_check_closed(state: EnvironmentState, adj_lists: dict, s):
    prev = {}
    prev[s] = None
    q = queue.Queue()
    q.put(s)
    while not q.empty():
        v = q.get()
        for u, d in adj_lists[v]:
            door_node = state.get_node(d)
            if State.CLOSED not in door_node.states and u not in prev:
                prev[u] = v
                q.put(u)
    return prev


def BFS(adj_lists: dict, s):
    prev = {}
    prev[s] = (None, None)
    q = queue.Queue()
    q.put(s)
    while not q.empty():
        v = q.get()
        for u, d in adj_lists[v]:
            if u not in prev:
                prev[u] = (v, d)
                q.put(u)
    return prev


# ScriptExecutor
###############################################################################


class ScriptExecutor(object):

    _action_executors = {
        Action.WALK: WalkExecutor(),
        Action.FIND: FindExecutor(),
        Action.SIT: SitExecutor(),
        Action.STANDUP: StandUpExecutor(),
        Action.GRAB: GrabExecutor(),
        Action.OPEN: OpenExecutor(False),
        Action.CLOSE: OpenExecutor(True),
        Action.PUTBACK: PutExecutor(Relation.ON),
        Action.PUTIN: PutExecutor(Relation.INSIDE),
        Action.SWITCHON: SwitchExecutor(True),
        Action.SWITCHOFF: SwitchExecutor(False),
        Action.DRINK: DrinkExecutor(), 
        Action.LOOKAT: LookAtExecutor(), 
        Action.TURNTO: TurnToExecutor(), 
        Action.WIPE: WipeExecutor(), 
        Action.RUN: WalkExecutor(),
        Action.PUTON: PutOnExecutor(), 
        Action.PUTOFF: PutOffExecutor(), 
        Action.GREET: GreetExecutor(), 
        Action.DROP: DropExecutor(), 
        Action.READ: ReadExecutor(), 
        Action.POINTAT: PointAtExecutor(), 
        Action.TOUCH: TouchExecutor(), 
        Action.LIE: LieExecutor(),
        Action.PUTOBJBACK: PutBackExecutor(), 
        Action.POUR: PourExecutor(), 
        Action.TYPE: TypeExecutor(), 
        Action.WATCH: WatchExecutor(), 
        Action.PUSH: MoveExecutor(), 
        Action.PULL: MoveExecutor(), 
        Action.MOVE: MoveExecutor(), 
        Action.RINSE: WashExecutor(),
        Action.WASH: WashExecutor(), 
        Action.SCRUB: WashExecutor(), 
        Action.SQUEEZE: SqueezeExecutor(), 
        Action.PLUGIN: PlugExecutor(True), 
        Action.PLUGOUT: PlugExecutor(False), 
        Action.CUT: CutExecutor(), 
        Action.EAT: EatExecutor(), 
        Action.SLEEP: SleepExecutor(), 
        Action.WAKEUP: WakeUpExecutor(), 
        Action.RELEASE: DropExecutor()
    }

    def __init__(self, graph: EnvironmentGraph, name_equivalence):
        self.graph = graph
        self.name_equivalence = name_equivalence
        self.processing_time_limit = 10  # 10 seconds
        self.processing_limit = 0
        self.info = ExecutionInfo()

    def find_solutions(self, script: Script, init_changers: List[StateChanger]=None):
        self.processing_limit = time.time() + self.processing_time_limit
        init_state = EnvironmentState(self.graph, self.name_equivalence)
        _apply_initial_changers(init_state, script, init_changers)
        return self.find_solutions_rec(script, 0, init_state)

    def find_solutions_rec(self, script: Script, script_index: int, state: EnvironmentState):
        if script_index >= len(script):
            yield state
        future_script = script.from_index(script_index)
        next_states = self.call_action_method(future_script, state, self.info)
        if next_states is not None:
            for next_state in next_states:
                for rec_state_list in self.find_solutions_rec(script, script_index + 1, next_state):
                    yield rec_state_list
                if time.time() > self.processing_limit:
                    break

    def execute(self, script: Script, init_changers: List[StateChanger]=None, w_graph_list: bool=True):

        info = self.info
        state = EnvironmentState(self.graph, self.name_equivalence, instance_selection=True)
        _apply_initial_changers(state, script, init_changers)
        graph_state_list = []
        for i in range(len(script)):
            prev_state = state
            if w_graph_list:
                graph_state_list.append(state.to_dict())
            
            future_script = script.from_index(i)
            state = next(self.call_action_method(future_script, state, info), None)
            if state is None:
                return False, prev_state, graph_state_list, i
                
        if w_graph_list:
            graph_state_list.append(state.to_dict())

        return True, state, graph_state_list, len(script)

    @classmethod
    def call_action_method(cls, script: Script, state: EnvironmentState, info: ExecutionInfo):
        executor = cls._action_executors.get(script[0].action, UnknownExecutor())
        return executor.execute(script, state, info)


def _apply_initial_changers(state: EnvironmentState, script: Script, changers: List[StateChanger]=None):
    if changers is not None:
        for changer in changers:
            changer.apply_changes(state, script=script)


# state preparation

_DEFAULT_PROPERTY_STATES = {Property.HAS_SWITCH: State.OFF,
                            Property.CAN_OPEN: State.CLOSED}


def _prepare_state(state: EnvironmentState, script: Script, name_equivalence, object_placing, properties_data):
    state_classes = {n.class_name for n in state.get_nodes()}
    script_classes = {so.name for sl in script for so in sl.parameters}
    missing_classes = set()
    for sc in script_classes:
        if sc not in state_classes and len(set(name_equivalence.get(sc, [])) & state_classes) == 0:
            missing_classes.add(sc)
    if len(missing_classes) > 0:
        for mc in missing_classes:
            if mc not in object_placing:
                raise ExecutionException('No placing information for "{0}"', mc)
            if mc not in properties_data:
                raise ExecutionException('No properties data for "{0}"', mc)
            new_node_id = state.get_max_node_id() + 1
            for pi in object_placing[mc]:
                dest = pi['destination']
                room_name = pi['room']
                properties = properties_data.get(mc, [])
                placed = False
                for dest_node in state.get_nodes_by_attr('class_name', dest):
                    if room_name is None:
                        new_node = _create_node(new_node_id, mc, properties)
                        _change_state(state, new_node, dest_node, [])
                        new_node_id += 1
                        placed = True
                        break
                    else:
                        room_node = _get_room_node(state, dest_node)
                        if room_node is not None and room_node.class_name == room_name:
                            new_node = _create_node(new_node_id, mc, properties)
                            _change_state(state, new_node, dest_node,
                                          [AddEdges(NodeInstance(new_node), Relation.INSIDE, NodeInstance(room_node))])
                            new_node_id += 1
                            placed = True
                            break
                if placed:
                    break


def _create_node(node_id: int, class_name: str, properties):
    states = [_DEFAULT_PROPERTY_STATES[p] for p in properties if p in _DEFAULT_PROPERTY_STATES]
    return GraphNode(node_id, class_name, None, properties, states, None, None)


def _change_state(state: EnvironmentState, new_node: GraphNode, dest_node: Node, add_changers: List[StateChanger]):
    changers = [AddNode(new_node),
                AddEdges(NodeInstance(new_node), Relation.ON, NodeInstance(dest_node)),
                AddEdges(NodeInstance(new_node), Relation.CLOSE, NodeInstance(dest_node), add_reverse=True)]
    changers.extend(add_changers)
    state.apply_changes(changers)


# Exception
###############################################################################

class ExecutionException(common.Error):
    pass
