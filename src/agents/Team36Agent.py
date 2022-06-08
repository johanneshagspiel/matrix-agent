import time
from typing import Dict
import numpy as np  # type: ignore
import random  # type: ignore
import json

from matrx.agents import StateTracker, Navigator
from matrx.agents.agent_types.patrolling_agent import PatrollingAgentBrain  # type: ignore
from matrx.actions import MoveNorth, OpenDoorAction, CloseDoorAction, GrabObject, DropObject  # type: ignore
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest, MoveNorthWest  # type: ignore
from matrx.agents.agent_utils.state import State  # type: ignore

from matrx.messages import Message
from matrx.objects import AgentBody

from src.bw4t.BW4TBlocks import CollectableBlock
from src.bw4t.BW4TBrain import BW4TBrain


class Team36Agent(BW4TBrain):
    '''
    This agent makes random walks and opens any doors it hits upon
    '''

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._moves = [MoveNorth.__name__, MoveEast.__name__, MoveSouth.__name__, MoveWest.__name__]

        self.initialized = False # for the first run
        self.voted = False # for the votation term
        self.phase = "announcing_phase" # set to "announce_phase" or "vote_phase" or "command_phase" accordingly
        # self.announcePhase = True # start out in announce phase
        self.N = 1 # the number of times to get a check from everyobe before voting phase
        self.allAgents = {} # contains all known members
        self.received_message_ids = set() # contains all known messages
        self.sent_message_ids = set()

        self.sent_cluster_messages_ids = set()
        self.received_cluster_messages_ids = set()

        self.leader_id = None

        self.UNIQUE_CLUSTER_ID = '87432986489'

        self.cluster_msg_types = {
            "hand_shake" : "hand_shake",
            "room_explore" : "room_explore",
            "pick_block" : "pick_block",
            "drop_block" : "drop_block",
            "goal_block_information" : "goal_block_information",
            "possible_block" : "possible_block",
            "check_out_block" : "check_out_block"
        }

        self.msg_types = {
            "announce": "announce", # those should only contain your ID, whether you're shapeblind, colorblind, and your speed
            "goal_block_information" : "goal_block_information",
            "vote": "vote", # these are sent when a vote is being sent, they contain just the random number
            "clusters": "clusters",
            "assignments": "assignments",
            "reply" : ["clusters_reply", "assignment_reply", "goal_block_information_reply"],
            "update" : "update",
            "wait_for_pick_up" : "wait_for_pick_up",
            "pick_up_instructions" : "pick_up_instructions",
            "go_to_open_room" : "go_to_open_room",
            "go_to_closed_room" : "go_to_closed_room",
            "empty_inventory" : "empty_inventory",
            "finish_sequence" : "finish_sequence",
            "wait_task" : "wait_task"
        }

        # Boolean to check if this is the first tick and so some information should be extracted from the state object
        self._first_tick = True

        self.decided_protocol = False

        # Attributes about the capabilities of the agent
        self.color_blind = False
        self.shape_blind = False
        # 1 implies normal speed of 1 action per tick. 3 givs 1 allowed action every 3 ticks
        self.speed = 1

        for (Key, Value) in settings.items():
                if Key == "colorblind":
                    self.color_blind = Value
                elif Key == "shapeblind":
                    self.shape_blind = Value
                elif Key == "slowdown":
                    self.speed = Value

        self.location = (0, 0)

        #
        self.agent_sense_range = 2
        self.block_sense_range = 2

        #
        self.goal_block_dict = {}
        self.goal_block_location_list = []
        self.goal_block_drop_list = []

        #
        self.inventory_emptying_under_way = False
        self.inventory_should_be_emptied = False

        #
        self.goal_blocks_identified = False
        self.all_goal_blocks_found = False
        self.finished = False

        #
        self.found_goal_block_dic = {}
        self.found_goal_block_list = []
        self.found_goal_block_location_list = []
        self.to_open_door_location_dic = {}

        #
        self.room_assignments = {}

        self.potential_goal_blocks_dic = {}

        self.cluster_end_blocks = {}

        #
        self.assigned_door_location_dic = {}

        #
        self.block_dictionary = {}

        #
        self.allAgents = {}

        #
        self.state_tracker = None
        self.navigator = None

        #
        self.current_task = None
        self.task_queue = []

        #
        self.interruptable = True

        #
        self.room_exploration_moves = []

        #
        self.inventory_dic = {}

        #
        self.clusters = []
        self.cluster_members = []  # not including ourself
        self.assignments = []

        #
        self.phase_reply_dictionnary = {}

        #
        self.alone = True  # to check if we're alone

        #
        self.cluster_mode = False
        self.group_mode = False

        #
        self.current_room_being_explored = ""

        #
        self.our_cluster_drop_list = {}

        #
        self.total_drop_list = {}
        self.cluster_us_drop_dic = {}

        self.current_to_drop = 0

        self.to_drop_item = None

        self.to_send = None

        self.leader = False

        self.assignment = None

        self.state = None

        #
        self.finish_sequence_under_way = False

        #
        self.agents_who_currently_empty_their_inventory = {}

        #
        self.incapable = False

        #
        self.cluster_potential_goal_block_dic = {}
        self.cluster_info_not_send = True
        self.cluster_rooms_to_check_dic = {}

        self.cluster_current_check_out_room = ""
        self.cluster_rooms_no_longer_needed_to_check = []

        self.seen_blocks = {}

    def log(self, s):
        DEBUG = True
        if(DEBUG):
            print("-> ", self.agent_id, ":", s)

    def initialize(self):
        super().initialize()

    def filter_bw4t_observations(self, state) -> State:

        self.state = state

        if (len(self.cluster_end_blocks) > 0 and self.cluster_mode and self.cluster_info_not_send):
            self._send_cluster_goal_information_message()
            self.cluster_info_not_send = False

        elif self._first_tick:
            self._sensing_world_first_time(state)
        else:
            self._environment_check(state)

        self._message_behavior(state)

        return state  # Why need to returning state

    def _message_behavior(self, state):

        self._receive_message_behavior()
        self._send_message_behavior()

    def _send_message_behavior(self):
        # this is the announce phase
        # ---- INITIALIZATION ----

        if(self.incapable == False):

            if (self.phase == "announcing_phase"):
                self._send_announce_message()
                #self._send_cluster_announce_message()

            if(self.alone == False):
                if (self.phase == "goal_information_phase"):
                    self._send_goal_block_information()

                #elif (self.phase == "goal")

                # ----GENERAL BEHAVIOR ----
                # this is the voting phase
                elif (self.phase == "vote_phase"):
                    if(not self.voted):
                        self._send_vote()
                        # self.log(str("vote made with score: " + str(self.allAgents[self.agent_id]["vote"])))
                        self.voted = True

                    # if we're the leader
                if (self.leader):
                    if (self.phase == "command_phase"):
                        # clustering...
                        agents = [agent for agent in self.allAgents.values() if agent.id != self.agent_id]  # get all agents except ours
                        agents_with_score = [(self._compute_disability_score(agent), agent) for agent in agents]
                        agents_with_score.sort(key=lambda x: x[0])

                        self._assign_tasks()

                    elif (self.phase == "wait_for_pick_up"):
                        self._send_wait_for_pick_up()

                #else:
                elif (self.phase == "wait_for_pick_up"):
                    self._send_wait_for_pick_up()

    def _receive_message_behavior(self):

        if(self.incapable == False):

            if (self._message_received()):
                for (index, message) in enumerate(self.received_messages):
                    # msg_protocol, msg_type, msg = self._parse_message_json(message)
                    parsed_message = self._parse_message_json(message)
                    # if the message is new, from our cluster and we're not alone
                    if (isinstance(parsed_message, list)):

                        index_to_delete = -1

                        for message in parsed_message:

                            msg_type = message[0]
                            msg_content = message[1]

                            # if (msg_type and not msg["message_id"] in self.received_cluster_messages_ids and not msg["message_id"] in self.sent_cluster_messages_ids):
                            if (msg_type):

                                    self.alone = False

                                    index_to_delete = index

                                    if("agent" in msg_content):
                                        if(msg_content["agent"] != self.agent_id):

                                            if (msg_type in self.cluster_msg_types["hand_shake"]):
                                                self._receive_cluster_handshake(msg_content)

                                            elif (msg_type in self.cluster_msg_types["room_explore"]):
                                                self._receive_cluster_room_exploration(msg_content)

                                            elif (msg_type in self.cluster_msg_types["pick_block"]):
                                                self._receive_cluster_pick_block(msg_content)

                                            elif (msg_type in self.cluster_msg_types["goal_block_information"]):
                                                self._receive_cluster_goal_block_information(msg_content)

                                            elif (msg_type in self.cluster_msg_types["possible_block"]):
                                                self._receive_cluster_possible_block_information(msg_content)

                                            elif (msg_type in self.cluster_msg_types["check_out_block"]):
                                                self._receive_check_out_block(msg_content)

                                    elif (msg_type in self.cluster_msg_types["drop_block"]):
                                        self._receive_cluster_drop_block(msg_content)

                        if (index_to_delete != -1):
                            self.received_messages.pop(index_to_delete)

                    else:

                        msg_type, msg = parsed_message

                        if(self.cluster_mode == False):

                            if (msg_type and not msg["message_id"] in self.received_message_ids):

                                if (msg["message_id"] not in self.sent_message_ids):
                                #if (msg["message_id"] not in self.sent_message_ids):
                                    self.alone = False

                                msg_content = msg["content"]
                                self.received_message_ids.add(msg["message_id"])  # add to known messages

                                if (msg_type in self.msg_types["reply"]):
                                    self._receive_reply_message(msg_type, msg_content)

                                elif (msg_type == self.msg_types["update"]):
                                    self._receive_update_message(msg_content)

                                elif (msg_type == self.msg_types["announce"]):
                                    self._receive_announce_message(msg_content)

                                elif (msg_type == self.msg_types["goal_block_information"]):
                                    self._receive_goal_information_message(msg_content)

                                elif (msg_type == self.msg_types["vote"]):
                                    self._receive_vote_message(msg_content)

                                elif (msg_type == self.msg_types["clusters"]):
                                    self._receive_clusters_message(msg_content)

                                elif (msg_type == self.msg_types["assignments"]):
                                    self._receive_assignments(msg_content)

                                elif (msg_type == self.msg_types["wait_for_pick_up"]):
                                    self._receive_wait_for_pick_up(msg_content)

                                elif (msg_type == self.msg_types["pick_up_instructions"]):
                                    self._receive_pick_up_instructions(msg_content)

                                elif (msg_type == self.msg_types["go_to_open_room"]):
                                    self._receive_go_to_open_room(msg_content)

                                elif (msg_type == self.msg_types["go_to_closed_room"]):
                                    self._receive_go_to_closed_room(msg_content)

                                elif (msg_type == self.msg_types["empty_inventory"]):
                                    self._receive_empty_inventory(msg_content)

                                elif (msg_type == self.msg_types["finish_sequence"]):
                                    self._receive_finish_sequence(msg_content)

                                elif (msg_type == self.msg_types["wait_task"]):
                                    self._receive_wait_task(msg_content)


    def _message_received(self):
        return (len(self.received_messages) > 0)

    def _new_phase_enter(self, phase):

        awaited_reply_list = []

        for agent_id in self.allAgents.keys():
            if agent_id != self.agent_id:
                awaited_reply_list.append((agent_id, False))

        self.phase_reply_dictionnary[phase] = awaited_reply_list

        self.phase = phase
        awaited_reply_list = None

    def _receive_wait_task(self, msg_content):
        self.log("Received wait_task")
        self.assignment = ("wait_task", msg_content['wait_location'])
        self.phase = "received_wait_task"
        self._clean_task_queue()

    def _receive_finish_sequence(self, msg_content):
        self.log("Received finish_sequence")
        self.assignment = ("finish_sequence", msg_content['finish_sequence'])
        self.phase = "received_finish_sequence"
        self._clean_task_queue()

    def _receive_empty_inventory(self, msg_content):
        self.log("Received empty_inventory")
        self.assignment = ("empty_inventory", msg_content['drop_sequence'])
        self.phase = "received_empty_inventory_instructions"
        self._clean_task_queue()

    def _receive_go_to_closed_room(self, msg_content):
        self.log("Received go_to_closed_room")
        self.assignment = ("go_to_closed_room", msg_content['location'], msg_content['door_id'])
        self.phase = "received_go_to_closed_room_instructions"
        self._clean_task_queue()

    def _receive_go_to_open_room(self, msg_content):
        self.log("Received go_to_open_room")
        self.assignment = ("go_to_open_room", msg_content['location'])
        self.phase = "received_go_to_open_room_instructions"
        self._clean_task_queue()

    def _receive_pick_up_instructions(self, msg_content):

        self.log("Received pick_up instructions")
        self.assignment = ("pick_up_instructions", msg_content['location_block'], msg_content['block_id'])
        self.phase = "received_pick_up_instructions"
        self._clean_task_queue()

    def _receive_reply_message(self, msg_type, msg_content):

        self.log("received " + str(msg_type))

        reply_phase_list = self.phase_reply_dictionnary[msg_type]
        new_reply_phase_list = []
        for (agent_id, reply_received) in reply_phase_list:
            if agent_id == msg_content["id"]:
                new_reply_phase_list.append((agent_id, True))
            else:
                new_reply_phase_list.append((agent_id, reply_received))
        self.phase_reply_dictionnary[msg_type] = new_reply_phase_list

        self._check_if_all_reply_received(msg_type)

    def _check_if_all_reply_received(self, phase):
        current_phase = self.phase_reply_dictionnary[phase]
        all_received = True

        for (agent_id, reply_received) in current_phase:
            if (reply_received == False):
                all_received = False
                break

        if all_received:
            self._advance_next_phase_group_protocol()

    def _decided_on_protocol(self):

        if (self.group_mode):
            self.log("Protocol used: group protocol")
            self.phase = "goal_information_phase"
        else:
            self.log("Protocol used: cluster protocol")
            self.phase = "room_exploration"

            self._clean_task_queue()

        self.decided_protocol = True

    def _advance_next_phase_group_protocol(self):

        if (self.phase == "goal_block_information_reply"):
            self.phase = "vote_phase"
            self.log("reached vote phase")
        elif (self.phase == "clusters_reply"):
            self.phase = "assign_tasks"
            self.log("reached assign_tasks phase")

    def _update_goal_drop_list(self):

        new_goal_drop_dic = {}

        for goal_block in self.goal_block_dict.values():
            new_goal_drop_dic[goal_block.id] = goal_block

        new_goal_drop_list = []

        for new_goal_drop in new_goal_drop_dic.values():
            new_goal_drop_list.append(new_goal_drop)

        new_goal_drop_list.sort(reverse=False,key=self._sort_drop_list)

        self.goal_block_drop_list = new_goal_drop_list


    def _combine_end_goal_block_information(self, other_person_goal_block):
        new_goal_block_dic = {}

        #for (id, location, color, shape, picked_up, finished, stored) in other_person_goal_block:
        for goal_block in other_person_goal_block:
            (id, location_x, location_y, color, shape, picked_up, finished, stored) = goal_block.split(",")

            for goal_block in self.goal_block_dict.values():
                if goal_block.id == id:
                    new_goal_block = GoalBlock()
                    new_goal_block.id = goal_block.id
                    new_goal_block.location = goal_block.location

                    if(self.color_blind and color != "#empty"):
                        new_goal_block.color = color
                    else:
                        new_goal_block.color = goal_block.color
                    if(self.shape_blind and shape != "-1"):
                        new_goal_block.shape = int(shape)
                    else:
                        new_goal_block.shape = goal_block.shape

                    new_goal_block.picked_up = goal_block.picked_up
                    new_goal_block.finished = goal_block.finished
                    new_goal_block.stored = goal_block.stored

                    new_goal_block_dic[new_goal_block.id] = new_goal_block
                    break

        self.goal_block_dict = new_goal_block_dic

    def _receive_goal_information_message(self, msg_content):
        other_person_goal_block = json.loads(msg_content)

        self._combine_end_goal_block_information(other_person_goal_block)
        self._update_goal_drop_list()

        self._send_reply_message("goal_block_information_reply")

    def _receive_wait_for_pick_up(self, msg_content):

        location_other_person = (msg_content['location'][0], msg_content['location'][1])
        id_other_person = msg_content['id']

        if id_other_person in self.agents_who_currently_empty_their_inventory:
            self.agents_who_currently_empty_their_inventory.pop(id_other_person)

        if (self._check_if_agents_inventory_is_full(id_other_person)):

            drop_sequence = []

            new_found_goal_block_location_list = []

            for (location, block) in self.found_goal_block_location_list:
                if (location[0] == id_other_person):

                    corresponding_goal_location = self._find_corresponding_goal_location_to_store(block)
                    drop_sequence.append((block.id, corresponding_goal_location))
                    new_found_goal_block_location_list.append((corresponding_goal_location, block))
                else:
                    new_found_goal_block_location_list.append((location, block))

            self.found_goal_block_location_list = new_found_goal_block_location_list

            self.agents_who_currently_empty_their_inventory[id_other_person] = True

            self._send_emtpy_inventory_message(drop_sequence, id_other_person)

        elif (self._get_if_all_goal_blocks_are_found()):

            if (self._check_if_all_blocks_dropped()):

                if (self.finish_sequence_under_way == False and self._no_one_still_under_way()):

                    finish_sequence = []

                    self._update_goal_drop_list()

                    for block in self.goal_block_drop_list:
                        (pick_up_location, block_id) = self._find_location_of_closest_found_block(block)

                        finish_sequence.append((pick_up_location, block_id, block.location))

                    self._send_finish_sequence(finish_sequence, id_other_person)

                    self.finish_sequence_under_way = True

                else:
                    closest_end_goal_location = self._determine_closest_end_goal_location(location_other_person)

                    self._send_wait_task(closest_end_goal_location, id_other_person)

                    self._print_inventory(id_other_person)

            else:

                if (self._check_if_agent_has_inventory(id_other_person)):

                        drop_sequence =[]

                        new_found_goal_block_location_list = []

                        for (location, block) in self.found_goal_block_location_list:
                            if (location[0] == id_other_person):

                                corresponding_goal_location = self._find_corresponding_goal_location_to_store(block)
                                drop_sequence.append((block.id, corresponding_goal_location))
                                new_found_goal_block_location_list.append((corresponding_goal_location, block))
                            else:
                                new_found_goal_block_location_list.append((location, block))

                        self.found_goal_block_location_list = new_found_goal_block_location_list

                        self.agents_who_currently_empty_their_inventory[id_other_person] = True

                        self._send_emtpy_inventory_message(drop_sequence, id_other_person)
                else:
                    closest_end_goal_location = self._determine_closest_end_goal_location(location_other_person)

                    self._send_wait_task(closest_end_goal_location, id_other_person)

                    self._print_inventory(id_other_person)

        elif (self._determine_if_block_to_pick_up_found()):

            reachable_blocks = []

            for block_id in self.found_goal_block_dic.keys():
                if (self._is_reachable(block_id)):
                    reachable_blocks.append(block_id)

            if (len(reachable_blocks) > 0):
                (location_block, block_id, result_block) = self._determine_closest_reachable_goal_block_location_and_id_with_agent_location(reachable_blocks, location_other_person)
                self._send_pick_up_instructions(id_other_person, location_block, block_id)

                self.found_goal_block_location_list.append(((id_other_person, id_other_person), result_block))
                self.found_goal_block_list.append(block_id)

                for goal_block in self.goal_block_dict.values():
                    if (goal_block.shape == result_block.shape and goal_block.color == result_block.color and goal_block.picked_up == False):
                        goal_block.picked_up = True
                        break

                new_found_goal_block_dic = {}
                for goal_block in self.found_goal_block_dic.values():
                    # check if block is not weird case where already picked up blocks pop up again
                    if (goal_block.id not in self.found_goal_block_list):

                        # check if we still need this block
                        for end_block in self.goal_block_dict.values():
                            if (goal_block.shape == end_block.shape and goal_block.color == end_block.color and end_block.picked_up == False):
                                new_found_goal_block_dic[goal_block.id] = goal_block
                                break;

                self.found_goal_block_dic = new_found_goal_block_dic
            else:
                #possible with full capable agent
                (door_location, door_id) = self._determine_closest_door_with_found_block()

                assigned_room = self._check_if_room_is_assigned(door_location)

                if (assigned_room == False):

                    other_person = self.allAgents[id_other_person]

                    if (other_person.color_blind == True and other_person.shape_blind == False):

                        self.to_open_door_location_dic.pop(door_id)
                        self.assigned_door_location_dic[door_id] = (door_location, None, id_other_person)

                    elif (other_person.color_blind == False and other_person.shape_blind == True):

                        self.to_open_door_location_dic.pop(door_id)
                        self.assigned_door_location_dic[door_id] = (door_location, id_other_person, None)

                    elif (other_person.color_blind == False and other_person.shape_blind == False):

                        self.to_open_door_location_dic.pop(door_id)
                        self.assigned_door_location_dic[door_id] = (door_location, id_other_person, id_other_person)

                    self._send_target_go_to_closed_room(id_other_person=id_other_person, door_location=door_location, door_id=door_id)

                else:
                    self._assign_new_room(location_other_person, id_other_person)
        else:
            self._assign_new_room(location_other_person, id_other_person)

    def _assign_new_room(self, location_other_person, id_other_person):

        other_person = self.allAgents[id_other_person]

        # location, color_seer_id, shape_seer_id
        if (other_person.color_blind == True and other_person.shape_blind == False):

            if (self._determine_if_room_not_visited_by_shape_seer()):
                (door_location, door_id) = self._determine_closest_room_not_visited_by_shape_seer(location_other_person)

                old_entry = self.assigned_door_location_dic[door_id]
                new_entry = (old_entry[0], old_entry[1], id_other_person)
                self.assigned_door_location_dic[door_id] = new_entry

                # was open here
                self._send_target_go_to_closed_room(id_other_person=id_other_person, door_location=door_location,
                                                    door_id=door_id)

            else:
                if (len(self.to_open_door_location_dic.values()) > 0):

                    (door_location, door_id) = self._determine_closest_door_location_and_id()
                    self.to_open_door_location_dic.pop(door_id)

                    self.assigned_door_location_dic[door_id] = (door_location, None, id_other_person)

                    self._send_target_go_to_closed_room(id_other_person, door_location, door_id)

                else:
                    self._what_to_do_if_nothing_is_left_to_do(id_other_person, location_other_person)

        # location, color_seer_id, shape_seer_id
        elif (other_person.color_blind == False and other_person.shape_blind == True):

            if (self._determine_if_room_not_visited_by_color_seer()):
                (door_location, door_id) = self._determine_closest_room_not_visited_by_color_seer(location_other_person)

                old_entry = self.assigned_door_location_dic[door_id]
                new_entry = (old_entry[0], id_other_person, old_entry[2])
                self.assigned_door_location_dic[door_id] = new_entry

                # was open here
                self._send_target_go_to_closed_room(id_other_person=id_other_person, door_location=door_location,
                                                    door_id=door_id)

            else:
                if (len(self.to_open_door_location_dic.values()) > 0):

                    (door_location, door_id) = self._determine_closest_door_location_and_id()
                    self.to_open_door_location_dic.pop(door_id)

                    self.assigned_door_location_dic[door_id] = (door_location, id_other_person, None)

                    self._send_target_go_to_closed_room(id_other_person, door_location, door_id)

                else:
                    self._what_to_do_if_nothing_is_left_to_do(id_other_person, location_other_person)

        elif (other_person.color_blind == False and other_person.shape_blind == False):

            if (self._determine_if_room_not_visited_by_either_color_or_shape_seer()):

                (door_location, door_id) = self._determine_closest_room_not_visited_by_fully_capable(
                    location_other_person)

                old_entry = self.assigned_door_location_dic[door_id]
                new_entry = (old_entry[0], id_other_person, id_other_person)
                self.assigned_door_location_dic[door_id] = new_entry

                # was open here
                self._send_target_go_to_closed_room(id_other_person=id_other_person, door_location=door_location,
                                                    door_id=door_id)

            else:
                if (len(self.to_open_door_location_dic.values()) > 0):

                    (door_location, door_id) = self._determine_closest_door_location_and_id()
                    self.to_open_door_location_dic.pop(door_id)

                    self.assigned_door_location_dic[door_id] = (door_location, id_other_person, id_other_person)

                    self._send_target_go_to_closed_room(id_other_person, door_location, door_id)

                else:
                    self._what_to_do_if_nothing_is_left_to_do(id_other_person, location_other_person)

    def _what_to_do_if_nothing_is_left_to_do(self, id_other_person, location_other_person):

        if (self._check_if_agent_has_inventory(id_other_person)):

            drop_sequence = []

            new_found_goal_block_location_list = []

            for (location, block) in self.found_goal_block_location_list:
                if (location[0] == id_other_person):

                    corresponding_goal_location = self._find_corresponding_goal_location_to_store(block)
                    drop_sequence.append((block.id, corresponding_goal_location))
                    new_found_goal_block_location_list.append((corresponding_goal_location, block))
                else:
                    new_found_goal_block_location_list.append((location, block))

            self.found_goal_block_location_list = new_found_goal_block_location_list

            self.agents_who_currently_empty_their_inventory[id_other_person] = True

            self._send_emtpy_inventory_message(drop_sequence, id_other_person)
        else:
            closest_end_goal_location = self._determine_closest_end_goal_location(location_other_person)

            self._send_wait_task(closest_end_goal_location, id_other_person)

            self._print_inventory(id_other_person)

    def _receive_assignments(self, msg_content):
        test = json.loads(msg_content)

        for (agent_id, location_1, door_id) in test:
            if (agent_id == self.agent_id):
                self.assignment = ("room_assignment", location_1, door_id)
                break

        self.phase = "assignment_start"
        self.log("reached assignment_start phase " + str(location_1))
        self._send_reply_message("assignments_reply")
        self._clean_task_queue()

    def _receive_update_message(self, msg_content):
        if msg_content["id"] in self.allAgents:
            agent = self.allAgents[msg_content["id"]]
            agent.location = msg_content["location"]
            self.allAgents[msg_content["id"]] = agent

            if (len(msg_content['potential_goal_blocks']) > 0):
                self._update_potential_goal_blocks_and_found_block_dic(msg_content['potential_goal_blocks'])

            if (len(msg_content['found_goal_blocks']) > 0):
                self._update_with_other_found_goal_blocks(msg_content['found_goal_blocks'])

        else:
            self.log("_receive_update_message: agent not found")

    def _receive_clusters_message(self, msg_content):
        for cluster in msg_content[-1:-1]:
            for agent in self.allAgents.values():
                if (agent.id == cluster[1]) or (agent.id == cluster[2]):
                    agent.cluster = (cluster[0], cluster[1], cluster[2])

        self.log("receive_clusters_message")
        self._send_reply_message("clusters_reply")

    def _receive_cluster_drop_block(self, msg_content):
        self.current_to_drop = self.current_to_drop + 1

        self._update_phase_cluster_mode()
        self._clean_task_queue()

    def _receive_check_out_block(self, msg_content):
        other_agent = self.allAgents[msg_content['agent']]
        room_door_location = None

        if (self.phase == "check_out_block" and self.cluster_current_check_out_room == msg_content["room_id"]):
            our_id = self.agent_id
            their_id = msg_content["agent"]

            their_id_is_smaller_than_ours = their_id < our_id

            if (their_id_is_smaller_than_ours):
                # self.cluster_rooms_to_check_dic.pop(msg_content["room_id"])
                #
                # self.phase == ""
                self._update_phase_cluster_mode()
                self._clean_task_queue()

        else:

            if msg_content["room_id"] in self.cluster_rooms_to_check_dic:
                self.cluster_rooms_to_check_dic.pop(msg_content["room_id"])

            if (self.color_blind and other_agent.color_blind):
            #     self.cluster_rooms_to_check_dic.pop(msg_content["room_id"])
            #
            # elif (self.shape_blind and other_agent.shape_blind):
            #     self.cluster_rooms_to_check_dic.pop(msg_content["room_id"])
            #
            # elif (self.shape_blind == False and self.color_blind == False):
            #     self.cluster_rooms_to_check_dic.pop(msg_content["room_id"])

                self.cluster_rooms_no_longer_needed_to_check.append(msg_content["room_id"])
                self._cluster_clean_potential_block_list()

    def _receive_cluster_possible_block_information(self, msg_content):
        other_agent = self.allAgents[msg_content['agent']]
        room_door_location = None

        door_tiles = self.state[{'class_inheritance': 'Door'}]
        for door in door_tiles:
            room_name = door['obj_id'].split("_-_")[0]

            if (msg_content["room_id"] == room_name):
                room_door_location = door['location']
                break

        if (msg_content["room_id"] not in self.cluster_rooms_no_longer_needed_to_check):

            if (self.color_blind and other_agent.shape_blind):
                self.cluster_rooms_to_check_dic[msg_content["room_id"]] = (room_door_location, msg_content['block'])

            elif (self.shape_blind and other_agent.color_blind):
                self.cluster_rooms_to_check_dic[msg_content["room_id"]] = (room_door_location, msg_content['block'])

            elif (self.shape_blind == False and self.color_blind == False):
                self.cluster_rooms_to_check_dic[msg_content["room_id"]] = (room_door_location, msg_content['block'])

        else:
            print("hi")

    def _receive_cluster_goal_block_information(self, msg_content):

        other_person_goal_block_list_as_string = []

        send_other_person = msg_content['goal_block_from_state']

        for goal_block in send_other_person:
            color = "#empty"
            shape =-1

            if "colour" in goal_block['visualization']:
                color = goal_block['visualization']['colour']

            if "shape" in goal_block['visualization']:
                shape = goal_block['visualization']['shape']

            other_person_goal_block_list_as_string.append(str(goal_block["obj_id"]) + "," + str(goal_block['location'][0]) + "," + str(goal_block['location'][1]) + "," + str(color) + "," + str(shape) + "," + "False" + "," + "False" + "," + "False")

        self._combine_end_goal_block_information(other_person_goal_block_list_as_string)
        self._update_goal_drop_list()

    def _cluster_get_block_at_drop_order(self, drop_order):

        test = self.goal_block_drop_list[::-1]

        for index, block in enumerate(test):
            if index == drop_order:
                return block

    def _receive_cluster_pick_block(self, msg_content):

        drop_order = int(msg_content["drop_order"])
        block_at_goal_drop_list = self._cluster_get_block_at_drop_order(drop_order)

        if(msg_content["drop_order"] in self.cluster_us_drop_dic):
            their_number = int(msg_content["agent"].split("_")[1])
            our_number = int(self.agent_id.split("_")[1])

            if (their_number > our_number):

                block_at_drop_order = self.cluster_us_drop_dic[msg_content["drop_order"]]

                block_is_needed_again = self._block_needed_multiple_times(block_at_goal_drop_list)

                if (block_is_needed_again):
                    next_drop_order_needed = self._find_drop_order(block_at_drop_order)

                    self.total_drop_list[msg_content["drop_order"]] = msg_content["block"]
                    self.total_drop_list[next_drop_order_needed] = block_at_drop_order

                    block_state_obj = self.cluster_us_drop_dic.pop(msg_content["drop_order"])
                    self.cluster_us_drop_dic[next_drop_order_needed] = block_state_obj

                    #self._update_list_based_on_found_block(msg_content["block"])
                    self._cluster_update_list_based_on_found_block(block_at_goal_drop_list)

                    self._send_cluster_pick_block(block_state_obj, next_drop_order_needed)

                elif block_at_drop_order['obj_id'] in self.inventory_dic:
                #if (self._check_if_block_in_found_goal_location_list(block_id)):

                    # self.to_drop_item = msg_content["block"]
                    # self.phase = "drop_unneeded_item"
                    #
                    # self._clean_task_queue()
                    # self._decide_on_task()
                    self.cluster_us_drop_dic.pop(msg_content["drop_order"])

                #this only happens if we have already picked up a block and stored due to full inventory
                #then the other person should drop their block
                else:
                    self._send_cluster_pick_block(msg_content["block"], msg_content["drop_order"])
            # else:
            #     block_name = msg_content["block"]["obj_id"]
            #
            #     for (location, block_id) in self.found_goal_block_location_list:
            #         if (block_name == block_id):
            #             self.to_drop_item = msg_content["block"]
            #             self.phase = "drop_unneeded_item"
            #             self._decide_on_task()
            #             break
        else:
            self.total_drop_list[msg_content["drop_order"]] = msg_content["block"]

            #self._update_list_based_on_found_block(msg_content["block"])
            self._cluster_update_list_based_on_found_block(block_at_goal_drop_list)

    def _receive_cluster_room_exploration(self, msg_content):

        if (self.phase == "room_exploration" and self.current_room_being_explored == msg_content["room_id"]):
            our_id = self.agent_id
            their_id = msg_content["agent"]

            their_id_is_smaller_than_ours = their_id < our_id

            if (their_id_is_smaller_than_ours):
                self._clean_task_queue()
        else:

            door_tiles = self.state[{'class_inheritance': 'Door'}]
            for door in door_tiles:
                room_name = door['obj_id'].split("_-_")[0]

                if (msg_content["room_id"] == room_name):
                    door_id = door['obj_id']

                    if(door_id in self.to_open_door_location_dic):
                        self.to_open_door_location_dic.pop(door_id)

                    break

    def _receive_cluster_handshake(self, msg_content):

        newAgentId = msg_content["agent"]
        #self.log(str("Handshake message received from: " + str(newAgentId)))

        if newAgentId != self.agent_id:

            if (not newAgentId in self.allAgents):
                new_agent = Agent()
                new_agent.id = msg_content["agent"]
                new_agent.speed = int(msg_content["speed"])
                new_agent.color_blind = msg_content["color_blind"]
                new_agent.shape_blind = msg_content["shape_blind"]
                new_agent.count = 0
                new_agent.cluster_protocol_count = 1

                self.allAgents[newAgentId] = new_agent

            else:
                self.allAgents[newAgentId].cluster_protocol_count = self.allAgents[newAgentId].cluster_protocol_count + 1

            if (self.decided_protocol == False):
                self._check_if_decided_on_protocol()

    def _receive_announce_message(self, msg_content):
        newAgentId = msg_content["id"]
        #self.log(str("announce message received from: " + str(newAgentId)))

        if (not newAgentId in self.allAgents):
            new_agent = Agent()
            new_agent.id = msg_content["id"]
            new_agent.speed = int(msg_content["speed"])
            new_agent.color_blind = msg_content["colorblind"]
            new_agent.shape_blind = msg_content["shapeblind"]
            new_agent.count = 1
            new_agent.cluster_protocol_count = 0

            self.allAgents[newAgentId] = new_agent

        else:
            self.allAgents[newAgentId].count = self.allAgents[newAgentId].count + 1

        if (self.decided_protocol == False):
            self._check_if_decided_on_protocol()

    def _receive_vote_message(self, msg_content):
        voteAgentId = msg_content["id"]
        vote = msg_content["vote"]
        #self.log(str("vote received from: " + str(voteAgentId) + " with score " + str(vote)))
        self.allAgents[voteAgentId].vote = vote
        if (self._check_all_agents_voted()):
            agents_by_vote_score = [(agent.vote, agent_id) for agent_id, agent in self.allAgents.items()]
            agents_by_vote_score.sort(reverse=True)
            #self.log(str("all agents have voted! " + str(agents_by_vote_score)))

            self.leader_id = agents_by_vote_score[0][1]

            if (self.leader_id == self.agent_id):
                self.leader = True

            #self.log(str("LEADER ELECTED: " + str(self.leader_id)))
            #self.log("command phase start...")
            self.phase = "command_phase"


    def _assign_tasks(self):

        to_check_assignments = []

        for (agent_id, agent) in self.allAgents.items():
            (door_location, door_id) = self._determine_closest_door_location_and_id()
            self.to_open_door_location_dic.pop(door_id)
            self.assignments.append((agent_id, door_location, door_id))

            if (agent.color_blind == True and agent.shape_blind == False):
                to_check_assignments.append((door_id, (door_location, None, agent_id)))
            elif (agent.color_blind == False and agent.shape_blind == True):
                to_check_assignments.append((door_id, (door_location, agent_id, None)))
            elif (agent.color_blind == False and agent.shape_blind == False):
                to_check_assignments.append((door_id, (door_location, agent_id, agent_id)))

        for (door_id, entry) in to_check_assignments:
            self.assigned_door_location_dic[door_id] = entry

        for (id, location_1, door_id_1) in self.assignments:
            if (id == self.agent_id):
                self.assignment = ("room_assignment", location_1, door_id_1)
                self.phase = "assignment_start"
                break
        self._clean_task_queue()

        self._send_assignments()

    def _block_needed_multiple_times(self, item_at_drop_order):

        test = self.goal_block_drop_list.copy()
        test.reverse()

        for (index, block) in enumerate(test):
            if block.color == item_at_drop_order.color and block.shape == item_at_drop_order.color and index not in self.total_drop_list:
                return True

        return False

    def _update_with_other_found_goal_blocks(self, other_found_goal_blocks):

        for other_goal_block in other_found_goal_blocks:

            other_goal_block_list = other_goal_block.split(",")

            new_goal_block = Block()
            new_goal_block.id = other_goal_block_list[0]
            new_goal_block.location = (int(other_goal_block_list[1]), int(other_goal_block_list[2]))
            new_goal_block.color = other_goal_block_list[3]
            new_goal_block.shape = int(other_goal_block_list[4])

            for goal_block in self.goal_block_dict.values():
                if new_goal_block.color == goal_block.color and new_goal_block.shape == goal_block.shape and goal_block.picked_up == False:
                    if new_goal_block.id not in self.found_goal_block_dic:
                        self.found_goal_block_dic[new_goal_block.id] = new_goal_block
                        break

            if new_goal_block.id in self.potential_goal_blocks_dic:
                self.potential_goal_blocks_dic.pop(new_goal_block.id)

    def _update_potential_goal_blocks_and_found_block_dic(self, other_user_potential_goal_blocks):

        to_delete = []


        for other_user_potential_goal_block in other_user_potential_goal_blocks:
            other_block = other_user_potential_goal_block.split(",")

            have_we_seen_this_block_before = False

            for potential_goal_block in self.potential_goal_blocks_dic.values():
                #update value already in our self.potential_goal_blocks_dic
                if (potential_goal_block.id == other_block[0]):

                    have_we_seen_this_block_before = True

                    last_seen_location = (int(other_block[1]), int(other_block[2]))
                    potential_goal_block.location = last_seen_location

                    if (other_block[3] != "#empty"):
                        potential_goal_block.color = other_block[3]

                    elif (int(other_block[4]) != -1):
                        potential_goal_block.shape = int(other_block[4])

                    if (potential_goal_block.color != "#empty" and potential_goal_block.shape != -1):

                            not_needed = True

                            for goal_block in self.goal_block_dict.values():
                                if (potential_goal_block.color == goal_block.color and potential_goal_block.shape == goal_block.shape and goal_block.picked_up == False):
                                    self.found_goal_block_dic[potential_goal_block.id] = potential_goal_block
                                    not_needed = False
                                    break

                            if (not_needed):
                                if potential_goal_block.id in self.potential_goal_blocks_dic:
                                    to_delete.append(potential_goal_block.id)

                    else:
                        self.potential_goal_blocks_dic[potential_goal_block.id] = potential_goal_block

            #add new value to the goal blocks dick
            if (have_we_seen_this_block_before == False):
                new_Block = Block()
                new_Block.id = other_block[0]
                new_Block.location = (int(other_block[1]), int(other_block[2]))
                new_Block.color = other_block[3]
                new_Block.shape = int(other_block[4])

                if(new_Block.shape != -1 and new_Block.color != "#empty"):
                    for goal_block in self.goal_block_dict.values():
                        if (new_Block.color == goal_block.color and new_Block.shape == goal_block.shape and goal_block.picked_up == False):
                            self.found_goal_block_dic[new_Block.id] = new_Block
                            break
                elif(new_Block.shape != -1):
                    for goal_block in self.goal_block_dict.values():
                        if (new_Block.shape == goal_block.shape and goal_block.picked_up == False):
                            self.potential_goal_blocks_dic[new_Block.id] = new_Block
                            break
                elif(new_Block.color != "#empty"):
                    for goal_block in self.goal_block_dict.values():
                        if (new_Block.color == goal_block.color and goal_block.picked_up == False):
                            self.potential_goal_blocks_dic[new_Block.id] = new_Block
                            break

        for entry in to_delete:
            self.potential_goal_blocks_dic.pop(entry)



    def _print_inventory(self, id_other_person):
        None

        # items_in_inventory = 0
        #
        # for (location, block) in self.found_goal_block_location_list:
        #     if (location[0] == id_other_person):
        #         items_in_inventory = items_in_inventory + 1
        #
        # missing_blocks = 0
        #
        # for end_goal in self.goal_block_dict.values():
        #     if end_goal.picked_up == False:
        #         missing_blocks = missing_blocks + 1
        #
        #
        # print(" ")
        # print("ID: " + str(id_other_person) + " inventory: " + str(items_in_inventory))
        # print("Missing Blocks: " + str(missing_blocks))
        # print(" ")

    def _check_if_block_in_found_goal_location_list(self, to_check_block_id):

        for (location, block_id) in self.found_goal_block_location_list:
            if(block_id == to_check_block_id):
                return True

        return False

    def _create_clusters(self):
        other_agent = None
        for agent in self.allAgents.values():
            if agent.id != self.agent_id:
                other_agent = agent

        cluster_with_id = [(0, other_agent.id, self.agent_id)]
        self.clusters = cluster_with_id

        for (cluster_id, sees_color_id, sees_shape_id) in self.clusters:
            for agent in self.allAgents.values():
                if (agent.id == sees_color_id) or (agent.id == sees_shape_id):
                    agent.cluster = (cluster_id, sees_color_id, sees_shape_id)

    def _update_clusters(self, agent, clusters):

        new_clusters = []

        if (len(clusters) != 0):
            found = False
            if (agent.color_blind) == False:
                for (sees_color_id, sees_shape_id) in clusters:
                    if (sees_color_id == None):
                        sees_color_id = agent.id
                        found == True
                        new_clusters.append((sees_color_id, agent.id))
                    else:
                        new_clusters.append((sees_color_id, sees_shape_id))
                if (found == False):
                    new_clusters.append((agent.id, None))
            elif (agent.shape_blind) == False:
                for (sees_color_id, sees_shape_id) in clusters:
                    if (sees_shape_id == None):
                        for (sees_color_id, sees_shape_id) in clusters:
                            if (sees_shape_id == None):
                                sees_shape_id = agent.id
                                found == True
                                new_clusters.append((agent.id, sees_shape_id))
                            else:
                                new_clusters.append((sees_color_id, sees_shape_id))
                        if (found == False):
                            new_clusters.append((None, agent.id))
        else:
            if (agent.shape_blind == False):
                new_clusters = [(agent.id, None)]
            elif (agent.color_blind == False):
                new_clusters = [(None, agent.id)]

        return new_clusters

    def _parse_message_json(self, message_content_list):
        """
        returns a tuple of the message type and the message content
        or false if it's not a json message
        """
        try:
            content = json.loads(message_content_list)
            # message is indeed a json

            if (isinstance(content, list)):
                message_content = content

                message_list = []

                for entry in message_content:
                    message_list.append((entry["type"], entry["content"]))

                return message_list

            elif (self.UNIQUE_CLUSTER_ID in content):
                # it's from our cluster
                #return "group_message", content[self.UNIQUE_CLUSTER_ID]["type"], content[self.UNIQUE_CLUSTER_ID]
                return content[self.UNIQUE_CLUSTER_ID]["type"], content[self.UNIQUE_CLUSTER_ID]
            else:
                return "error", False, content
        except:
            return "error", False, message_content_list

    def _construct_cluster_message(self, msg_type, msg_content, target_id=None):

        message = {
            "type": msg_type,
            "content": msg_content,
        }
        message = [message]

        #self.sent_cluster_messages_ids.add(messageid)

        m = Message(content=json.dumps(message), from_id=self.agent_id, to_id=target_id)
        return m

    def _construct_message(self, msg_type, msg_content, target_id=None):
        messageid = hash(hash(self.agent_id) + hash(time.time()) + hash(random.randint(1, 100000)))
        message = {
            self.UNIQUE_CLUSTER_ID: {
                "type": msg_type,
                "content": msg_content,
                "message_id": messageid
            }
        }
        self.sent_message_ids.add(messageid)

        m = Message(content=json.dumps(message), from_id=self.agent_id, to_id=target_id)
        return m

    def _send_wait_task(self, furthest_end_goal_location, id_other_person):

        content = {
            "wait_location": furthest_end_goal_location
        }
        #self.log("send wait_task to " + str(id_other_person))

        wait_task_message = self._construct_message(self.msg_types["wait_task"], content, target_id=id_other_person)

        self.send_message(wait_task_message)

    def _send_finish_sequence(self, finish_sequence, id_other_person):
        content = {
            "finish_sequence": finish_sequence
        }
        #self.log("send finish_sequence to " + str(id_other_person))

        finish_sequence_message = self._construct_message(self.msg_types["finish_sequence"], content,
                                                          target_id=id_other_person)
        self.send_message(finish_sequence_message)

    def _send_emtpy_inventory_message(self, drop_sequence, id_other_person):
        content = {
            "drop_sequence" : drop_sequence
        }
        #self.log("send empty_inventory to " + str(id_other_person))

        empty_invenentory_message = self._construct_message(self.msg_types["empty_inventory"], content, target_id=id_other_person)
        self.send_message(empty_invenentory_message)

    def _send_target_go_to_open_room(self, id_other_person, door_location):
        content = {
            "id": self.agent_id,
            "location": door_location
        }
        #self.log("send go_to_open_room to " + str(id_other_person))
        target_go_to_open_room_message = self._construct_message(self.msg_types["go_to_open_room"], content, target_id=id_other_person)
        self.send_message(target_go_to_open_room_message)

    def _send_target_go_to_closed_room(self, id_other_person, door_location, door_id):
        content = {
            "id": self.agent_id,
            "location": door_location,
            "door_id" : door_id
        }
        #self.log("send go_to_closed_room to " + str(id_other_person))
        target_go_to_open_room_message = self._construct_message(self.msg_types["go_to_closed_room"], content, target_id=id_other_person)
        self.send_message(target_go_to_open_room_message)

    def _send_wait_for_pick_up(self):
        content = {
            "id": self.agent_id,
            "location": self.location
        }
        #self.log("send wait_for_pick_up to " + str(self.leader_id))
        self.phase = "wait_for_pick_up_response"
        update_message = self._construct_message(self.msg_types["wait_for_pick_up"], content, target_id=self.leader_id)

        self.send_message(update_message)

    def _send_assignments(self):
        content = json.dumps(self.assignments)

        message = self._construct_message(self.msg_types["assignments"], content)
        self.send_message(message)
        #self.log("send assignments")

        self.phase = "assignment_start"

        #self.log("reached assignment_start phase")

    def _send_round_update(self):

        potential_goal_blocks_json = []

        for potential_goal_block in self.potential_goal_blocks_dic.values():
            potential_goal_blocks_json.append(potential_goal_block.toJSON())

        found_goal_block_json = []

        for found_goal_block in self.found_goal_block_dic.values():
            found_goal_block_json.append(found_goal_block.toJSON())

        content = {
            "id" : self.agent_id,
            "location" : self.location,
            "potential_goal_blocks" : potential_goal_blocks_json,
            "found_goal_blocks" : found_goal_block_json
        }

        if (self.leader_id != None):
            update_message = self._construct_message(self.msg_types["update"], content, target_id=self.leader_id)
        else:
            update_message = self._construct_message(self.msg_types["update"], content)

        self.send_message(update_message)

    def _send_cluster_drop_block(self):
        content = {
            "drop_order" : self.current_to_drop
        }
        cluster_announce_msg = self._construct_cluster_message(self.cluster_msg_types["drop_block"], content)
        self.send_message(cluster_announce_msg)

    def _send_cluster_announce_message(self):

        content = {
            "agent": self.agent_id,
            "color_blind" : self.color_blind,
            "shape_blind" : self.shape_blind,
            "speed" : self.speed
        }
        cluster_announce_msg = self._construct_cluster_message(self.cluster_msg_types["hand_shake"], content)
        self.send_message(cluster_announce_msg)

    def _send_cluster_goal_information_message(self):

        list = []

        for value in self.cluster_end_blocks.values():
            list.append(value)

        content = {
            "goal_block_from_state" : list,
            "agent" : self.agent_id
        }

        goal_information_msg = self._construct_cluster_message(self.cluster_msg_types["goal_block_information"], content)
        self.send_message(goal_information_msg)

    def _send_cluster_pick_block(self, block_id, drop_order, target_id=None):

        content = {
            "agent" : self.agent_id,
            "block": block_id,
            "drop_order": drop_order
        }
        cluster_pick_msg = self._construct_cluster_message(self.cluster_msg_types["pick_block"], content, target_id=target_id)
        self.send_message(cluster_pick_msg)

    def _send_cluster_found_potential_blocks(self, room):

        list = []

        for value in room.values():
            list.append(value)

        closest_room_door = self.state.get_closest_room_door()[0]
        closest_room_door_name = closest_room_door['obj_id']
        room_name = closest_room_door_name.split("_-_")[0]

        content = {
            "room_id" : room_name,
            "block" : list,
            "agent" : self.agent_id
        }

        #self.log("Potential Room " + str(room_name))
        #self.log("Blocks: " + str(list))

        possible_block_msg = self._construct_cluster_message(self.cluster_msg_types["possible_block"], content)
        self.send_message(possible_block_msg)

    def _send_cluster_check_out_block_message(self, room_id, block_obj):
        #self.log("Check out " + str(room_id))
        content = {
            "room_id" : room_id,
            "block" : block_obj,
            "agent" : self.agent_id
        }

        check_out_block_msg = self._construct_cluster_message(self.cluster_msg_types["check_out_block"], content)
        self.send_message(check_out_block_msg)

    def _send_cluster_room_exploration(self, room_id):
        #self.log("Exploring " + str(room_id))
        content = {
            "room_id" : room_id,
            "agent" : self.agent_id
        }
        room_exploration_msg = self._construct_cluster_message(self.cluster_msg_types["room_explore"], content)
        self.send_message(room_exploration_msg)

    def _send_announce_message(self):
        aemg = {
            "id": self.agent_id,
            "shapeblind": self.shape_blind,
            "colorblind": self.color_blind,
            "speed": self.speed
        }  # TODO get the actual information
        announce_existance_msg = self._construct_message(self.msg_types["announce"], aemg)
        self.send_message(announce_existance_msg)

        self.phase = "announcing_phase"
        #self.log("initial announcement made")

    def _send_goal_block_information(self):
        content = []

        for goal_block in self.goal_block_dict.values():
            content.append(goal_block.toJSON())

        message = self._construct_message(self.msg_types["goal_block_information"], json.dumps(content))
        self.send_message(message)
        self._new_phase_enter("goal_block_information_reply")
        #self.log("send goal_information")
        #self.log("reached goal_block_information_reply")

    def _send_clusters(self):
        content = json.dumps(self.clusters)

        message = self._construct_message(self.msg_types["clusters"], content)
        self.send_message(message)
        self._new_phase_enter("clusters_reply")

    def _send_vote(self):
        vote_num = self._generate_vote_number()
        content = {
            "id": self.agent_id,
            "vote": vote_num
        }
        self.allAgents[self.agent_id].vote = vote_num  # sets own vote number for election

        message = self._construct_message(self.msg_types["vote"], content)
        self.send_message(message)

    def _send_pick_up_instructions(self, id_other_person, location_block, block_id):
        content = {
            "location_block" : location_block,
            "block_id" : block_id
        }
        #self.log("Send pick_up to " + id_other_person)
        message = self._construct_message(self.msg_types["pick_up_instructions"], msg_content=content, target_id=id_other_person)
        self.send_message(message)

    def _send_reply_message(self, topic):
        content = {
            "id": self.agent_id,
            "reply": topic
        }
        message = self._construct_message(topic, content)
        self.send_message(message)
        #self.log("send goal_block_information_reply")

    def _check_if_decided_on_protocol(self):

        all_agents_agree_to_our_protocol = True
        number_agents_send_our_protocol = 0

        for (agent_id, agent) in self.allAgents.items():
            if(agent_id != self.agent_id):
                if(agent.count > 0):
                    number_agents_send_our_protocol = number_agents_send_our_protocol + 1
                if (agent.count < self.N ):
                    all_agents_agree_to_our_protocol = False

        all_agents_agree_to_cluster_protocol = True
        number_agents_send_cluster_protocol = 0

        for (agent_id, agent) in self.allAgents.items():
            if(agent_id != self.agent_id):
                if (agent.cluster_protocol_count > 0):
                    number_agents_send_cluster_protocol = number_agents_send_cluster_protocol + 1
                if (agent.cluster_protocol_count < self.N):
                    all_agents_agree_to_cluster_protocol = False

        if (self._first_tick == False):
        #if(number_agents_send_cluster_protocol > 0 or number_agents_send_our_protocol > 0):
            if (all_agents_agree_to_our_protocol or all_agents_agree_to_cluster_protocol):
                # if (number_agents_send_cluster_protocol > number_agents_send_our_protocol):
                if (number_agents_send_cluster_protocol > number_agents_send_our_protocol):
                    if (all_agents_agree_to_cluster_protocol):
                        self.cluster_mode = True
                        self.group_mode = False

                        self._decided_on_protocol()

                else:
                    if (all_agents_agree_to_our_protocol):
                        self.cluster_mode = False
                        self.group_mode = True

                        self._decided_on_protocol()

    def _check_all_known_agents_announced_twice(self) -> bool:
        """
        returns true if all agents in self.allAgentsCount have their counter set >= 2
        """
        c = True
        # print("-----")
        # print(self.allAgents)
        for (agent_id, agent) in self.allAgents.items():
            if (agent.count < self.N):
                c = False
        return c

    def _check_all_agents_voted(self):
        # returns true if all agents have a vote number
        c = True
        for (agent_id, agent) in self.allAgents.items():
            if (agent.vote == None):
                c = False
        return c

    def _generate_vote_number(self) -> int:
        base = random.randint(0, 1000000000)  # zero and a billion
        offset = 0
        if True:  # TODO replace with colorblindness
            offset += 1
        if True:  # TODO replace with shapeblindness
            offset += 1
        offset += 2  # TODO replace with speed
        offset = 1000000000 * offset
        return base + offset

    def _compute_disability_score(self, agent) -> int:
        """
        you get:
        "id": self.agent_id,
        "shapeblind": False,
        "colorblind": False,
        "speed": 1
        and with that you create the score (lower means more diabled)
        note: score of 3 means perfectly healthy
        """
        score = 3
        if (agent.shape_blind):
            score -= 1
        if (agent.color_blind):
            score -= 1
        return score

    def _environment_check(self, state):
        self.state_tracker.update(state)

        if(self.goal_blocks_identified == False):
            self._sense_goal_blocks(state)
            self._check_if_goal_blocks_are_identified()

        elif(self.all_goal_blocks_found == False):
            self._check_if_all_goal_blocks_are_found()
            self._check_if_finished()

            if(self.inventory_emptying_under_way == False):
                self._check_if_inventory_is_full()

        #check if we have a current task
        if(self.current_task != None):
            self.current_task.environment_check(state)
        else:
            self._decide_on_task()

        # regular update
        #update agents
        self._sense_other_agents(state)

        #update blocks
        self._sense_collectable_blocks(state)

        if(self.interruptable):
            if(self._determine_if_block_to_pick_up_found() or self.all_goal_blocks_found):
                self.current_task = None
                self.task_queue = []
                self._decide_on_task()

        #update location of agent
        self.location = self.state_tracker.get_memorized_state()[self.agent_id]['location']
        self.allAgents[self.agent_id].location = self.location

    def decide_on_bw4t_action(self, state: State):
        action = self.current_task.get_action(state)

        if (self.incapable == False and self.group_mode):
        #if (self.incapable == False):
            self._send_round_update()

        return action

    def _sensing_world_first_time(self, state):
        self.state_tracker = StateTracker(agent_id=self.agent_id)
        self.navigator = Navigator(agent_id=self.agent_id,
                                   action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

        self.state_tracker.update(state)
        self.location = self.state_tracker.get_memorized_state()[self.agent_id]['location']

        # end goal tiles
        area_tiles = state[{'class_inheritance': 'AreaTile'}]
        for area in area_tiles:
            for (key, value) in area.items():
                if (key == "is_drop_zone" and value == True):
                    self.goal_block_location_list.append(area["location"])

        # end goal blocks
        area_tiles = state[{'class_inheritance': 'CollectableBlock'}]
        if (area_tiles != None):
            for area in area_tiles:
                for (key, value) in area.items():
                    if (key == "location" and (value in self.goal_block_location_list)):
                        new_goal_block = GoalBlock()
                        new_goal_block.location = area["location"]
                        new_goal_block.id = area['obj_id']

                        if (self.color_blind == False):
                            new_goal_block.color = area['visualization']['colour']
                        if (self.shape_blind == False):
                            new_goal_block.shape = area['visualization']['shape']

                        self.goal_block_dict[new_goal_block.id] = new_goal_block

                        #self.goal_block_drop_list.append(new_goal_block)

        # door locations
        door_tiles = state[{'class_inheritance': 'Door'}]
        for door in door_tiles:
            new_door_location = Door_Location()
            new_door_location.door_id = door["obj_id"]
            old_location = door["location"]
            old_y = old_location[1]
            new_location = (old_location[0], old_y + 1)
            new_door_location.location = new_location
            self.to_open_door_location_dic[new_door_location.door_id] = new_door_location

        for (key, value) in self.sense_capability.get_capabilities().items():
            if isinstance(key, AgentBody):
                self.agent_sense_range = value
            if isinstance(key, CollectableBlock):
                self.block_sense_range = value

        myself = Agent()
        myself.id = self.agent_id
        myself.speed = self.speed
        myself.color_blind = self.color_blind
        myself.shape_blind = self.shape_blind

        if (myself.color_blind == True and myself.shape_blind == True):
            self.incapable = True
            wait_task = WaitTask(self,10000)
            self.current_task = wait_task

        else:
            self._send_cluster_announce_message()
            self._send_announce_message()

        self.allAgents[self.agent_id] = myself

        self._create_room_exploration_moves_2(state)
        self._first_tick = False
        self._second_tick = True

    def _update_task_queue(self):
        if(len(self.task_queue) == 0):
            if(self.group_mode):
                self._update_phase_group_mode()
            elif(self.cluster_mode):
                self._update_phase_cluster_mode()

            self._decide_on_task()
        else:
            self.current_task = self.task_queue.pop()

    def _update_phase_cluster_mode(self):

        self._update_found_goal_dic()

        if(self.phase == "pick_block"):
            self._send_cluster_pick_block(self.to_send[0], self.to_send[1])
            self._update_found_goal_dic()
        elif (self.phase == "drop_block"):
            self._send_cluster_drop_block()
            self.current_to_drop = self.current_to_drop + 1
        elif(self.phase == "drop_unneeded_item"):
            return
        elif(self.phase == "check_out_block"):
            checked_out_room = self.cluster_current_check_out_room
            self.cluster_current_check_out_room = ""
            self._cluster_update_found_goal_blocks(checked_out_room)
            self.cluster_rooms_to_check_dic.pop(checked_out_room)
            self._cluster_clean_potential_block_list()

        self._cluster_clean_potential_block_list()
        closest_room_door = self.state.get_closest_room_door()
        current_room = closest_room_door[0]['room_name']

        if (current_room in self.cluster_potential_goal_block_dic and current_room not in self.cluster_rooms_no_longer_needed_to_check and current_room):
            if (len(self.cluster_potential_goal_block_dic[current_room]) > 0):
                self._send_cluster_found_potential_blocks(self.cluster_potential_goal_block_dic[current_room])
            #self.cluster_potential_goal_block_dic = {}

        #self._check_if_inventory_is_full()
        blooks_in_room = self._determine_if_goal_blocks_in_current_room()
        if (self._determine_cluster_if_all_blocks_found()):
            if (self.current_to_drop in self.cluster_us_drop_dic):
                self.phase = "drop_block"
            else:
                self.phase = "go_to_closest_drop_location"

        # elif (self.inventory_should_be_emptied):
        #     self.phase = "cluster_empty_inventory"

        elif (blooks_in_room):
            self._update_found_goal_dic()
            self.phase = "pick_block"

        elif (len(self.cluster_rooms_to_check_dic) > 0):
            self.phase = "check_out_block"

        elif (self._if_no_rooms_to_explore_left()):
            self.phase = "go_to_closest_drop_location"

        else:
            self.phase = "room_exploration"

    def _update_phase_group_mode(self):
        if self.phase == "assignment_start":
            self.phase = "wait_for_pick_up"
        elif self.phase == "received_pick_up_instructions":
            self.phase = "wait_for_pick_up"
        elif self.phase == "received_go_to_open_room_instructions":
            self.phase = "wait_for_pick_up"
        elif self.phase == "received_go_to_closed_room_instructions":
            self.phase = "wait_for_pick_up"
        elif self.phase == "received_empty_inventory_instructions":
            self.phase = "wait_for_pick_up"
        elif self.phase == "received_wait_task":
            self.phase = "wait_for_pick_up"

    def _clean_task_queue(self):
        self.current_task = None
        self.task_queue = []
        self._decide_on_task()

    def _create_room_exploration_moves_2(self, state: State):

        room_0 = state[{'room_name': 'room_0', 'class_inheritance': 'Wall'}]
        door_location = state[{'room_name': 'room_0', 'class_inheritance': 'Door'}]['location']

        move_list = [(MoveNorth.__name__, {}), (MoveNorth.__name__, {})]

        current_location = (door_location[0], door_location[1])
        last_location = current_location
        return_position = current_location

        finished = False

        wall_map = {}
        for entry in room_0:
            wall_map[entry["location"]] = True
        wall_map[door_location] = True

        if (len(wall_map) == 8):
            move_list.append((MoveNorthWest.__name__, {}))

        else:

            test_location = (last_location[0], last_location[1] - 1)
            good = self._check_direction(test_location, wall_map, self.block_sense_range, "North")
            if good:
                move_list.append((MoveNorth.__name__, {}))

                last_location = test_location
                return_position = last_location

            while (finished == False):
                test_location = (last_location[0] + 1, last_location[1])
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "East")
                if good:
                    move_list.append((MoveEast.__name__, {}))
                    last_location = test_location
                else:
                    finished = True
            finished = False

            # go west as much as possible
            while (finished == False):
                test_location = (last_location[0] - 1, last_location[1])
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "West")
                if good:
                    move_list.append((MoveWest.__name__, {}))
                    last_location = test_location
                else:
                    finished = True
            finished = False

            still_move = True
            iterate = False

            while(still_move):

                one_move = (last_location[0], last_location[1] - 1)
                one_move_good = self._check_direction(test_location, wall_map, self.block_sense_range, "North")

                two_move = (last_location[0], last_location[1] - 2)
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "North")

                if (good == False):
                    still_move = False
                    break

                else:

                    move_list.append((MoveNorth.__name__, {}))
                    move_list.append((MoveNorth.__name__, {}))

                    last_location = (last_location[0], last_location[1] - 2)

                    if (iterate):

                        while (finished == False):
                            test_location = (last_location[0] + 1, last_location[1])
                            good = self._check_direction(test_location, wall_map, self.block_sense_range, "East")
                            if good:
                                move_list.append((MoveEast.__name__, {}))
                                last_location = test_location
                            else:
                                finished = True
                        finished = False

                        # go west as much as possible
                        while (finished == False):
                            test_location = (last_location[0] - 1, last_location[1])
                            good = self._check_direction(test_location, wall_map, self.block_sense_range, "West")
                            if good:
                                move_list.append((MoveWest.__name__, {}))
                                last_location = test_location
                            else:
                                finished = True
                        finished = False

                        iterate = False

                    else:
                        # go west as much as possible
                        while (finished == False):
                            test_location = (last_location[0] - 1, last_location[1])
                            good = self._check_direction(test_location, wall_map, self.block_sense_range, "West")
                            if good:
                                move_list.append((MoveWest.__name__, {}))
                                last_location = test_location
                            else:
                                finished = True
                        finished = False

                        while (finished == False):
                            test_location = (last_location[0] + 1, last_location[1])
                            good = self._check_direction(test_location, wall_map, self.block_sense_range, "East")
                            if good:
                                move_list.append((MoveEast.__name__, {}))
                                last_location = test_location
                            else:
                                finished = True
                        finished = False

                        iterate = True

        move_list.reverse()
        self.room_exploration_moves = move_list


    def _create_room_exploration_moves(self, state: State):

        room_0 = state[{'room_name': 'room_0', 'class_inheritance': 'Wall'}]
        door_location = state[{'room_name': 'room_0', 'class_inheritance': 'Door'}]['location']

        move_list = [(MoveNorth.__name__, {}), (MoveNorth.__name__, {})]

        current_location = (door_location[0], door_location[1])
        last_location = current_location
        return_position = current_location

        finished = False

        wall_map = {}
        for entry in room_0:
            wall_map[entry["location"]] = True
        wall_map[door_location] = True

        if (len(wall_map) == 8):
            move_list.append((MoveNorthWest.__name__, {}))

        else:

            test_location = (last_location[0], last_location[1] - 1)
            good = self._check_direction(test_location, wall_map, self.block_sense_range, "North")

            if good:
                move_list.append((MoveNorth.__name__, {}))

                last_location = test_location
                return_position = last_location

                test_location = (last_location[0], last_location[1] - 1)
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "North")

                if good:
                    move_list.append((MoveNorth.__name__, {}))

                    last_location = test_location
                    return_position = last_location

            #move east as much as possible
            while (finished == False):
                test_location = (last_location[0] + 1, last_location[1])
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "East")
                if good:
                    move_list.append((MoveEast.__name__, {}))
                    last_location = test_location
                else:
                    finished = True
            finished = False

            #go north as much as possible
            while (finished == False):
                test_location = (last_location[0], last_location[1] - 1)
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "North")
                if good:
                    move_list.append((MoveNorth.__name__, {}))
                    last_location = test_location
                else:
                    finished = True
            finished = False

            #go west as much as possible
            while (finished == False):
                test_location = (last_location[0] - 1, last_location[1])
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "West")
                if good:
                    move_list.append((MoveWest.__name__, {}))
                    last_location = test_location
                else:
                    finished = True
            finished = False

            #go south as much as possible
            while (finished == False):
                test_location = (last_location[0], last_location[1] + 1)
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "South")
                if good:
                    move_list.append((MoveSouth.__name__, {}))
                    last_location = test_location
                else:
                    finished = True
            finished = False

            #go east to the return position
            while (finished == False):
                test_location = (last_location[0] + 1, last_location[1])
                good = self._check_direction(test_location, wall_map, self.block_sense_range, "East")
                if good:
                    move_list.append((MoveEast.__name__, {}))
                    last_location = test_location
                else:
                    finished = True
            finished = False

        move_list.reverse()
        self.room_exploration_moves = move_list

    def _check_direction(self, test_location, wall_map, range, direction):
        run = True
        good = True
        check = 0

        while (run and (check < range)):
            if (direction == "North"):
                if ((test_location[0], test_location[1] - check) in wall_map):
                    good = False
                    run = False
                else:
                    check = check + 1
            elif (direction == "East"):
                if ((test_location[0] + check, test_location[1]) in wall_map):
                    good = False
                    run = False
                else:
                    check = check + 1
            elif (direction == "South"):
                if ((test_location[0], test_location[1] + check) in wall_map):
                    good = False
                    run = False
                else:
                    check = check + 1
            elif (direction == "West"):
                if ((test_location[0] - check, test_location[1]) in wall_map):
                    good = False
                    run = False
                else:
                    check = check + 1

        return good

    def _check_start_top(self, test_location, wall_map, range):
        run = True
        good = True
        top_check = 0

        while(run):
            while(top_check < range):
                if((test_location[0], test_location[1] - top_check) in wall_map):
                    good = False
                    run = False
                else:
                    top_check = top_check + 1
            top_check = range + 1

            if((test_location[0], test_location[1] + top_check) in wall_map):
                good = False
            run = False

        return good

    def _if_no_rooms_to_explore_left(self):
        return (len(self.to_open_door_location_dic) == 0)

    def _sense_goal_blocks(self, state: State):

        at_end_location = []

        for entry in state.values():
            for (key, value) in entry.items():
                if (key == "location" and (value in self.goal_block_location_list)):
                    at_end_location.append(entry)

        if(at_end_location != None):
            for entry in at_end_location:
                for(key, value) in entry.items():
                    if(key == "is_goal_block" and value == True):
                        if entry['obj_id'] in self.goal_block_dict:

                            self.cluster_end_blocks[entry['obj_id']] = entry

                            goal_block = self.goal_block_dict[entry['obj_id']]

                            if (self.color_blind == False):
                                goal_block.color = entry['visualization']['colour']
                            if (self.shape_blind == False):
                                goal_block.shape = int(entry['visualization']['shape'])

                            self.goal_block_dict[entry['obj_id']] = goal_block

                            self.goal_block_drop_list.append(goal_block)

                        else:

                            self.cluster_end_blocks[entry['obj_id']] = entry

                            goal_block = GoalBlock();
                            goal_block.id = entry['obj_id']
                            goal_block.location = entry['location']

                            if (self.color_blind == False):
                                goal_block.color = entry['visualization']['colour']
                            if (self.shape_blind == False):
                                goal_block.shape = int(entry['visualization']['shape'])

                            self.goal_block_dict[goal_block.id] = goal_block

                            self.goal_block_drop_list.append(goal_block)

        self.goal_block_drop_list.sort(reverse=False,key=self._sort_drop_list)

        self._update_goal_drop_list()

        #potential problem: maybe call self._update_goal_drop_list()

    def _sort_drop_list(self, entry):
        return entry.location[1]

    def _cluster_update_found_goal_blocks(self, checked_out_room):

        if checked_out_room in self.cluster_potential_goal_block_dic:

            potential_goal_blocks_in_room = self.cluster_potential_goal_block_dic[checked_out_room]

            other_person_potential_goals_list = self.cluster_rooms_to_check_dic[checked_out_room][1]
            other_person_potential_goals_list_dic = {}

            for entry in other_person_potential_goals_list:
                other_person_potential_goals_list_dic[entry['obj_id']] = entry

            for (block_id, block_obj) in potential_goal_blocks_in_room.items():
                if block_id in other_person_potential_goals_list_dic:
                    other_block_obj = other_person_potential_goals_list_dic[block_id]

                    new_block = Block()
                    new_block.id = block_id
                    new_block.location = block_obj['location']

                    if "colour" in other_block_obj["visualization"]:
                        new_block.color = other_block_obj["visualization"]["colour"]
                    elif "colour" in block_obj["visualization"]:
                        new_block.color = block_obj["visualization"]["colour"]

                    if "shape" in other_block_obj["visualization"]:
                        new_block.shape = other_block_obj["visualization"]["shape"]
                    elif "shape" in block_obj["visualization"]:
                        new_block.shape = block_obj["visualization"]["shape"]

                    block_needed = self._cluster_check_if_block_with_shape_and_clour_needed(new_block)

                    if block_needed:
                        self._check_if_block_is_goal_block(new_block)

            self.cluster_potential_goal_block_dic.pop(checked_out_room)


    def _sense_other_agents(self, state: State):
        None
        # seen_agents = state[{'class_inheritance': 'AgentBrain'}]
        # if(isinstance(seen_agents, list)):
        #     for agent in seen_agents:
        #         if (agent['obj_id'] != self.agent_id):
        #             seen_agent = self.allAgents.get(agent['obj_id'])
        #             seen_agent.last_seen_location = agent['location']
        #             self.allAgents[seen_agent.id] = seen_agent
        # else: #case if there is only one agent around
        #     if (seen_agents['obj_id'] != self.agent_id):
        #         seen_agent = self.allAgents.get(seen_agents['obj_id'])
        #         seen_agent.last_seen_location = seen_agents['location']
        #         self.allAgents[seen_agent.id] = seen_agent

    def _sense_collectable_blocks(self, state: State):

        #Potentially does not work with the one person case due to check if potential goal block

        seen_blocks = state[{'class_inheritance': 'CollectableBlock'}]
        if (seen_blocks != None):
            if(isinstance(seen_blocks, list)):
                for block in seen_blocks:

                    self.seen_blocks[block['obj_id']] = block

                    if block['obj_id'] in self.block_dictionary:
                        seen_block = self.block_dictionary[block['obj_id']]
                        seen_block.location = block['location']
                        self.block_dictionary[block['obj_id']] = seen_block

                        self._check_if_block_is_goal_block(seen_block)
                        self._check_if_block_is_possible_goal_block(seen_block, block)

                        if(self.cluster_mode):
                            self._cluster_clean_potential_block_list()

                    else:
                        new_block = self._create_block_from_dic(block)
                        self.block_dictionary[new_block.id] = new_block

                        self._check_if_block_is_goal_block(new_block)
                        self._check_if_block_is_possible_goal_block(new_block, block)

                        if(self.cluster_mode):
                            self._cluster_clean_potential_block_list()
            else:

                self.seen_blocks[seen_blocks['obj_id']] = seen_blocks

                if seen_blocks['obj_id'] in self.block_dictionary:
                    seen_block = self.block_dictionary[seen_blocks['obj_id']]
                    seen_block.location = seen_blocks['location']
                    self.block_dictionary[seen_blocks['obj_id']] = seen_block

                    self._check_if_block_is_goal_block(seen_block)
                    self._check_if_block_is_possible_goal_block(seen_block, seen_blocks)

                    if (self.cluster_mode):
                        self._cluster_clean_potential_block_list()
                else:
                    new_block = self._create_block_from_dic(seen_blocks)
                    self.block_dictionary[new_block.id] = new_block

                    self._check_if_block_is_goal_block(new_block)
                    self._check_if_block_is_possible_goal_block(new_block, seen_blocks)

                    if (self.cluster_mode):
                        self._cluster_clean_potential_block_list()

        self._update_found_goal_dic()

    def _cluster_check_if_block_with_shape_and_clour_needed(self, block):
        if(block not in self.found_goal_block_list):

            for goal_block in self.goal_block_dict.values():
                if (block.color == goal_block.color and block.shape == goal_block.shape and goal_block.picked_up == False):
                    return True

        return False

    def _check_if_block_is_goal_block(self, block):
        #Possible error here
        if(block not in self.found_goal_block_list):

            for goal_block in self.goal_block_dict.values():
                if (block.color == goal_block.color and block.shape == goal_block.shape and goal_block.picked_up == False):
                    self.found_goal_block_dic[block.id] = block

    def _cluster_clean_potential_block_list(self):

        for room in self.cluster_rooms_no_longer_needed_to_check:
            if room in self.cluster_potential_goal_block_dic:
                self.cluster_potential_goal_block_dic.pop(room)

    def _check_if_block_is_possible_goal_block(self, block, block_state_obj):

        for goal_block in self.goal_block_dict.values():
            if (self.color_blind == False and self.shape_blind == False):
                if (block.color == goal_block.color and block.shape == goal_block.shape and goal_block.picked_up == False):
                    self.found_goal_block_dic[block.id] = block
                    break
            elif (self.color_blind == False):
                if (block.color == goal_block.color and goal_block.picked_up == False):
                    self.potential_goal_blocks_dic[block.id] = block

                    room_number = block.id.split("_")[3]
                    room_id = "room_" + str(room_number)

                    if (room_id in self.cluster_potential_goal_block_dic):
                        old_entry = self.cluster_potential_goal_block_dic[room_id]
                        old_entry[block.id] = block_state_obj
                        self.cluster_potential_goal_block_dic[room_id] = old_entry
                    else:
                        temp_dic = {}
                        temp_dic[block.id] = block_state_obj
                        self.cluster_potential_goal_block_dic[room_id] = temp_dic

                    break
            elif (self.shape_blind == False):
                if (block.shape == goal_block.shape and goal_block.picked_up == False):
                    self.potential_goal_blocks_dic[block.id] = block

                    room_number = block.id.split("_")[3]
                    room_id = "room_" + str(room_number)

                    if (room_id in self.cluster_potential_goal_block_dic):
                        old_entry = self.cluster_potential_goal_block_dic[room_id]
                        old_entry[block.id] = block_state_obj
                        self.cluster_potential_goal_block_dic[room_id] = old_entry
                    else:
                        temp_dic = {}
                        temp_dic[block.id] = block_state_obj
                        self.cluster_potential_goal_block_dic[room_id] = temp_dic

                    break

    def _check_if_finished(self):
        finished = True

        for goal_block in self.goal_block_dict.values():
            if (goal_block.finished == False):
                finished = False
                break

        self.finished = finished

    def _check_if_goal_blocks_are_identified(self):
        one_missing = False

        if(len(self.goal_block_dict) == len(self.goal_block_location_list)):
            for end_goal_block in self.goal_block_dict.values():
                if (end_goal_block.color == "#empty" or end_goal_block.shape == -1):
                    one_missing = True
                    break
        else:
            one_missing = True

        if (one_missing == False):
            self.goal_blocks_identified = True

    def _get_if_all_goal_blocks_are_found(self):
        all_found = True

        for goal_block in self.goal_block_dict.values():
            if goal_block.picked_up == False:
                all_found = False
                break

        return all_found

    def _check_if_all_goal_blocks_are_found(self):
        all_found = True

        for goal_block in self.goal_block_dict.values():
            if goal_block.picked_up == False:
                all_found = False
                break

        if (all_found == True):
            self.all_goal_blocks_found = True

    def _create_block_from_dic(self, block_as_dic):
        new_Block = Block()
        new_Block.id = block_as_dic['obj_id']
        new_Block.location = block_as_dic['location']

        if (self.color_blind == False):
            new_Block.color = block_as_dic['visualization']['colour']
        if (self.shape_blind == False):
            new_Block.shape = block_as_dic['visualization']['shape']

        return new_Block

    def _decide_on_task(self):

        if(self.alone):

            if (self.goal_blocks_identified == False):
                furthest_goal_block_location = self._determine_furthest_end_goal_location()
                new_task = GoToLocationTask(agent=self, target_location=furthest_goal_block_location)
                self.current_task = new_task
            elif (self.all_goal_blocks_found == False):

                if(self.inventory_should_be_emptied):

                    (closest_end_location, block_id) = self._determine_closest_found_end_goal_location()
                    go_to_closest_store_location_task = GoToLocationTask(agent=self, target_location=closest_end_location)
                    store_item_task = DropBlockTask(agent=self,block_id=block_id)
                    self.task_queue.append(store_item_task)
                    self.current_task = go_to_closest_store_location_task

                elif(self._determine_if_block_to_pick_up_found()):

                    reachable_blocks = []

                    for block_id in self.found_goal_block_dic.keys():
                        if (self._is_reachable(block_id)):
                            reachable_blocks.append(block_id)

                    if (len(reachable_blocks) > 0):
                        (location, block_id) = self._determine_closest_reachable_goal_block_location_and_id(reachable_blocks)

                        to_add_list = []

                        if(location == self.location):
                            to_add_list.append(GoToLocationTask(self, (location[0], location[1] + 1)))

                        go_to_block_task = GoToLocationTask(agent=self, target_location=location)
                        pick_up_block_task = PickUpBlockTask(agent=self, block_id=block_id)

                        to_add_list.append(go_to_block_task)
                        to_add_list.append(pick_up_block_task)

                        to_add_list.reverse()
                        for task in to_add_list:
                            self.task_queue.append(task)

                        self.current_task = self.task_queue.pop()

                    else:
                        (door_location, door_id) = self._determine_closest_door_with_found_block()

                        go_to_door_task = GoToLocationTask(agent=self, target_location=door_location)
                        open_door_task = OpenDoorTask(agent=self, door_id=door_id)
                        explore_room_task = ExploreRoomTask(agent=self)

                        self.task_queue.append(explore_room_task)
                        self.task_queue.append(open_door_task)
                        self.current_task = go_to_door_task
                else:
                    (location, door_id) = self._determine_closest_door_location_and_id()
                    go_to_door_task = GoToLocationTask(agent=self, target_location=location)
                    open_door_task = OpenDoorTask(agent=self, door_id=door_id)
                    explore_room_task = ExploreRoomTask(agent=self)

                    self.task_queue.append(explore_room_task)
                    self.task_queue.append(open_door_task)
                    self.current_task = go_to_door_task

            elif (self.all_goal_blocks_found):
                self._check_if_finished()

                if(self.finished):
                    temp_wait = WaitTask(self, duration=100)
                    self.current_task = temp_wait

                else:
                    self.interruptable = False

                    if (len(self.goal_block_drop_list) == 0):
                        temp_wait = WaitTask(self, duration=100)
                        self.current_task = temp_wait
                    else:

                        top_block = self.goal_block_drop_list.pop()
                        (found_block_location, found_block_id) = self._find_location_of_closest_found_block(top_block)

                        to_add_tasks = []

                        if(found_block_location[0] != "inventory"):
                            go_to_block_location = GoToLocationTask(agent=self, target_location=found_block_location)
                            pick_up_block_task = PickUpBlockTask(agent=self, block_id=found_block_id)
                            to_add_tasks.append(go_to_block_location)
                            to_add_tasks.append(pick_up_block_task)

                        go_to_end_block_location = GoToLocationTask(agent=self, target_location=top_block.location)
                        drop_found_block = DropBlockTask(agent=self, block_id=found_block_id)
                        to_add_tasks.append(go_to_end_block_location)
                        to_add_tasks.append(drop_found_block)

                        to_add_tasks.reverse()
                        for task in to_add_tasks:
                            self.task_queue.append(task)

                        self.current_task = self.task_queue.pop()

        elif(self.group_mode):
            self.interruptable = False
            if(self.assignment != None):
                if (self.phase == "assignment_start"):
                    agent = self
                    location = (self.assignment[1][0], self.assignment[1][1])
                    door_id = self.assignment[2]

                    go_to_door_task_1 = GoToLocationTask(agent=agent, target_location=location)
                    open_door_task = OpenDoorTask(agent=self, door_id=door_id)
                    explore_room_task_1 = ExploreRoomTask(agent=self)

                    self.task_queue.append(explore_room_task_1)
                    self.task_queue.append(open_door_task)

                    self.current_task = go_to_door_task_1

                elif (self.phase == "received_wait_task"):
                    wait_location = (self.assignment[1][0], self.assignment[1][1])
                    go_to_wait_location = GoToLocationTask(agent=self, target_location=wait_location)

                    self.current_task = go_to_wait_location

                elif (self.phase == "received_finish_sequence"):

                    for finish_sequence_task in self.assignment[1]:
                        pick_up_location = (finish_sequence_task[0][0], finish_sequence_task[0][1])
                        block_id = finish_sequence_task[1]
                        drop_off_location = (finish_sequence_task[2][0], finish_sequence_task[2][1])

                        go_to_stored_block = GoToLocationTask(agent=self, target_location=pick_up_location)
                        pick_up_block = PickUpBlockTask(agent=self, block_id=block_id)
                        go_to_goal_location = GoToLocationTask(agent=self, target_location=drop_off_location)
                        drop_block_task = DropBlockTask(agent=self, block_id=block_id)

                        self.task_queue.append(drop_block_task)
                        self.task_queue.append(go_to_goal_location)
                        self.task_queue.append(pick_up_block)
                        self.task_queue.append(go_to_stored_block)

                    self.current_task = self.task_queue.pop()

                elif (self.phase == "received_empty_inventory_instructions" ):

                    for drop_task in self.assignment[1]:

                        block_id = drop_task[0]
                        location = (drop_task[1][0], drop_task[1][1])
                        #location_2 = (drop_task[1][0] - 1, drop_task[1][1])

                        #go_to_location_task_test = GoToLocationTask(agent=self, target_location=location_2)
                        go_to_location_task = GoToLocationTask(agent=self, target_location=location)
                        drop_block_task = DropBlockTask(agent=self, block_id=block_id)

                        #self.task_queue.append(go_to_location_task_test)
                        self.task_queue.append(drop_block_task)
                        self.task_queue.append(go_to_location_task)

                    self.current_task = self.task_queue.pop()

                elif (self.phase == "received_go_to_open_room_instructions" and self.assignment[0] == "go_to_open_room"):
                    location = (self.assignment[1][0], self.assignment[1][1])

                    go_to_door_task = GoToLocationTask(agent=self, target_location=location)
                    explore_room_task = ExploreRoomTask(agent=self)

                    self.task_queue.append(explore_room_task)

                    self.current_task = go_to_door_task

                elif (self.phase == "received_go_to_closed_room_instructions" and self.assignment[0] == "go_to_closed_room"):
                    location = (self.assignment[1][0], self.assignment[1][1])
                    door_id = self.assignment[2]

                    go_to_door_task = GoToLocationTask(agent=self, target_location=location)
                    open_door_task = OpenDoorTask(agent=self, door_id=door_id)
                    explore_room_task = ExploreRoomTask(agent=self)

                    self.task_queue.append(explore_room_task)
                    self.task_queue.append(open_door_task)

                    self.current_task = go_to_door_task

                elif (self.phase == "wait_for_pick_up" or self.phase == "wait_for_pick_up_response"):
                    wait_task = WaitTask(self, 1000)

                    self.current_task = wait_task

                elif (self.phase == "received_pick_up_instructions" and self.assignment[0] == "pick_up_instructions"):
                    location = (self.assignment[1][0], self.assignment[1][1])
                    block_id = self.assignment[2]

                    go_to_location_task = GoToLocationTask(self,location)
                    pick_up_block_task = PickUpBlockTask(self, block_id)

                    self.task_queue.append(pick_up_block_task)

                    self.current_task = go_to_location_task


        elif (self.cluster_mode):

            if(self.phase == "drop_block"):

                block_to_drop = self.cluster_us_drop_dic[self.current_to_drop]
                block_to_drop_id = block_to_drop["obj_id"]

                loation_to_go_to = None

                for location, block_id in self.found_goal_block_location_list:
                    if block_id == block_to_drop_id:
                        loation_to_go_to = location
                        break

                to_add_tasks = []

                if (loation_to_go_to[0] != "inventory"):
                    go_to_block_task = GoToLocationTask(self, loation_to_go_to)
                    pick_up_task = PickUpBlockTask(self, block_to_drop_id)

                    to_add_tasks.append(go_to_block_task)
                    to_add_tasks.append(pick_up_task)

                test = self.goal_block_drop_list.copy()
                test.reverse()

                location_to_drop = test[self.current_to_drop].location

                go_to_drop_location_task = GoToLocationTask(self, location_to_drop)
                drop_block_task = DropBlockTask(self, block_to_drop["obj_id"])

                to_add_tasks.append(go_to_drop_location_task)
                to_add_tasks.append(drop_block_task)

                to_add_tasks.reverse()
                for task in to_add_tasks:
                    self.task_queue.append(task)

                self.current_task = self.task_queue.pop()

            elif (self.phase == "check_out_block"):

                (room_id, closest_to_check_out_room_location, block_obj) = self._cluster_find_closest_to_check_out_room_location()

                go_to_closest_check_out_room_task = GoToLocationTask(self, closest_to_check_out_room_location)
                explore_room_task = ExploreRoomTask(self)

                self.task_queue.append(explore_room_task)
                self.current_task = go_to_closest_check_out_room_task

                #self.cluster_rooms_to_check_dic.pop(room_id)
                self.cluster_current_check_out_room = room_id

                self._send_cluster_check_out_block_message(room_id, block_obj)

            elif(self.phase == "go_to_closest_drop_location"):

                if(len(self.cluster_us_drop_dic) > 0):
                    closest_drop_location = self._determine_cluster_closest_end_goal_location()
                    go_to_closest_drop_location = GoToLocationTask(self, (closest_drop_location[0] + 1, closest_drop_location[1]))
                    wait_task = WaitTask(self, 3)

                    self.task_queue.append(wait_task)
                    self.current_task = go_to_closest_drop_location

                else:
                    wait_task = WaitTask(self, 10000)

                    self.current_task = wait_task


            elif(self.phase == "cluster_empty_inventory"):

                entries_to_update_in_inventory = []

                for block in self.inventory_dic.values():

                    corresponding_goal_location = self._find_cluster_corresponding_goal_location_to_store(block)

                    go_to_location_task = GoToLocationTask(agent=self, target_location=corresponding_goal_location)
                    drop_block_task = DropBlockTask(agent=self, block_id=block.id)

                    self.task_queue.append(drop_block_task)
                    self.task_queue.append(go_to_location_task)

                    entries_to_update_in_inventory.append((block.id, corresponding_goal_location))

                self.current_task = self.task_queue.pop()

                for entry in entries_to_update_in_inventory:
                    self.inventory_dic.pop(entry[0])
                    index_to_delete = self._delete_entry_from_block_location_list(entry[0])
                    self.found_goal_block_location_list.pop(index_to_delete)
                    self.found_goal_block_location_list.append((entry[1], entry[0]))

            elif(self.phase == "room_exploration"):
                (target_location, door_id) = self._determine_closest_door_location_and_id()


                if(len(self.to_open_door_location_dic) == 0):
                    print("hi")

                self.to_open_door_location_dic.pop(door_id)

                go_to_door_task = GoToLocationTask(agent=self, target_location=target_location)
                open_door_task = OpenDoorTask(agent=self, door_id=door_id)
                explore_room_task = ExploreRoomTask(agent=self)

                self.task_queue.append(explore_room_task)
                self.task_queue.append(open_door_task)

                self.current_task = go_to_door_task

                room_id = door_id.split("_-_")[0]

                self.current_room_being_explored = room_id

                self._send_cluster_room_exploration(room_id)

            elif(self.phase == "pick_block"):

                self._update_found_goal_dic()

                closest_room_door = self.state.get_closest_room_door()
                current_room = closest_room_door[0]['room_name']
                current_room_number = current_room.split("_")[1]

                goal_blocks_in_current_room = self._find_blocks_in_room(current_room_number)

                (target_location, block_id) = self._determine_closest__goal_block_location_and_id_from_list(goal_blocks_in_current_room)

                to_add_tasks = []

                if(self.location == target_location):
                    move_one_down_task = GoToLocationTask(self, (target_location[0], target_location[1] + 1))
                    to_add_tasks.append(move_one_down_task)

                found_block = self.found_goal_block_dic.pop(block_id)
                self.found_goal_block_list.append(block_id)
                self.inventory_dic[block_id] = found_block
                self.found_goal_block_location_list.append((("inventory", "inventory"), block_id))

                found_block_state_obj = self.seen_blocks[block_id]


                # blocks_in_room = self.state.get_of_type("CollectableBlock")
                # found_block_state_obj = None
                # for block_in_room in blocks_in_room:
                #     if(block_in_room["obj_id"] == block_id):
                #         found_block_state_obj = block_in_room
                #         break

                #self._update_list_based_on_found_block(found_block_state_obj)
                self._update_list_based_on_found_block_class(found_block)

                #drop_order = self._find_drop_order(found_block_state_obj)
                drop_order = self._find_drop_order_class(found_block)

                self.total_drop_list[drop_order] = found_block_state_obj

                self.to_send = (found_block_state_obj, drop_order)

                self.cluster_us_drop_dic[drop_order] = found_block_state_obj

                go_to_block_task = GoToLocationTask(self, target_location)
                pick_up_task = PickUpBlockTask(self, block_id)

                to_add_tasks.append(go_to_block_task)
                to_add_tasks.append(pick_up_task)

                to_add_tasks.reverse()
                for task in to_add_tasks:
                    self.task_queue.append(task)

                self.current_task = self.task_queue.pop()

            elif (self.phase == "drop_unneeded_item"):
                to_drop_id = None

                to_find_shape = int(self.to_drop_item['visualization']['shape'])
                to_find_colour = self.to_drop_item['visualization']['colour']

                for(block_id, block) in self.inventory_dic.items():
                    if (block.color == to_find_colour and block.shape == to_find_shape):
                        to_drop_id = block_id
                        break

                self.inventory_dic.pop(to_drop_id)

                found_drop_order = None

                for (drop_order, block_as_state_obj) in self.cluster_us_drop_dic.items():
                    if block_as_state_obj["obj_id"] == to_drop_id:
                        found_drop_order = drop_order
                        break

                self.cluster_us_drop_dic.pop(found_drop_order)

                blocks_in_room = self.state.get_of_type("CollectableBlock")
                block_still_in_room = False

                for block_in_room in blocks_in_room:
                    if(block_in_room["obj_id"] == to_drop_id):
                        block_still_in_room = True
                        break

                if (block_still_in_room == False):

                    drop_unecessary_item = DropBlockTask(self, block_id=to_drop_id)
                    self.current_task = drop_unecessary_item
                    self.phase = ""

                else:
                    self.phase = ""
                    self._update_phase_cluster_mode()
                    self._decide_on_task()

    def _cluster_find_closest_to_check_out_room_location(self):

        min_route_length = 10000000000000

        target_location = (0, 0)
        room_id = ""
        block_obj = None

        for key, value in self.cluster_rooms_to_check_dic.items():

            self.navigator.reset_full()
            self.navigator.add_waypoint(value[0])
            route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

            if route_length < min_route_length:
                min_route_length = route_length

                target_location = value[0]
                block_obj = value[1]
                room_id = key

        return (room_id, target_location, block_obj)

    def _determine_cluster_closest_end_goal_location(self):

        back_up_location = self.goal_block_drop_list[0].location

        if (len(self.cluster_us_drop_dic) == 0):
            return back_up_location

        else:

            earliest_drop = 100000000000

            for key in self.cluster_us_drop_dic.keys():
                if key < earliest_drop:
                    earliest_drop = key

            return self.goal_block_drop_list[earliest_drop].location

    def _delete_entry_from_block_location_list(self, to_find_block_id):

        for index, entry in enumerate(self.found_goal_block_location_list):
            if entry[1] == to_find_block_id:
                return index

    def _determine_cluster_if_all_blocks_found(self):
        return (len(self.total_drop_list.keys()) >= len(self.goal_block_drop_list))

    def _update_list_based_on_found_block_class(self, found_block):

        for goal_block in self.goal_block_dict.values():
            if (goal_block.shape == found_block.shape and goal_block.color == found_block.color and goal_block.picked_up == False):
                goal_block.picked_up = True
                break

        self._update_found_goal_dic()

    def _cluster_update_list_based_on_found_block(self, block_at_goal_drop_list):

        old_value = self.goal_block_dict[block_at_goal_drop_list.id]
        old_value.picked_up = True
        self.goal_block_dict[block_at_goal_drop_list.id] = old_value

        # for goal_block in self.goal_block_dict.values():
        #     if (goal_block.shape == block_at_goal_drop_list.shape and goal_block.color == block_at_goal_drop_list.color
        #             and goal_block.picked_up == False):
        #         goal_block.picked_up = True
        #         break

        self._update_found_goal_dic()

    def _update_list_based_on_found_block(self, found_block_state_obj):

        for goal_block in self.goal_block_dict.values():
            if (goal_block.shape == found_block_state_obj['visualization']['shape'] and goal_block.color == found_block_state_obj['visualization']['colour']
                    and goal_block.picked_up == False):
                goal_block.picked_up = True
                break

        self._update_found_goal_dic()

    def _update_found_goal_dic(self):

        new_found_goal_block_dic = {}
        for goal_block in self.found_goal_block_dic.values():
            # check if block is not weird case where already picked up blocks pop up again
            if (goal_block.id not in self.found_goal_block_list):

                # check if we still need this block
                for end_block in self.goal_block_dict.values():
                    if (goal_block.shape == end_block.shape and goal_block.color == end_block.color and end_block.picked_up == False):
                        new_found_goal_block_dic[goal_block.id] = goal_block
                        break;

        new_found_goal_block_dic_2 = {}
        for new_block in new_found_goal_block_dic.values():
            result = self._find_drop_order_class(new_block)

            if result != -1:
                new_found_goal_block_dic_2[new_block.id] = new_block

        self.found_goal_block_dic = new_found_goal_block_dic_2

    def _find_drop_order_class(self, found_block):

        result = -1
        test = self.goal_block_drop_list.copy()
        test.reverse()

        for (index, block) in enumerate(self.goal_block_drop_list[::-1]):
            if block.color == found_block.color and block.shape == found_block.shape and index not in self.total_drop_list:
                return index

        return result

    def _find_drop_order(self, found_block):

        result = -1
        test = self.goal_block_drop_list.copy()
        test.reverse()

        for (index, block) in enumerate(test):
            if block.color == found_block['visualization']['colour'] and block.shape == found_block['visualization']['shape'] and index not in self.total_drop_list:
                return index

        return result

    def _find_blocks_in_room(self, current_room_number):

        blocks_in_current_room = []

        for (block_name, block) in self.found_goal_block_dic.items():
            block_room_number = block_name.split("_")[3]

            if (block_room_number == current_room_number):
                blocks_in_current_room.append(block)

        return blocks_in_current_room


    def _determine_if_goal_blocks_in_current_room(self):
        closest_room_door = self.state.get_closest_room_door()
        current_room = closest_room_door[0]['room_name']
        current_room_number = current_room.split("_")[1]

        #self._sense_collectable_blocks()

        for (block_name, block) in self.found_goal_block_dic.items():
            block_room_number = block_name.split("_")[3]

            if (block_room_number == current_room_number):
                return True

        return False

    def _determine_door_location_of_room_with_block(self, block_id):
        room = block_id.split("Block_in_")[1]
        room_number = room.split("_")[1]

        for (room_id, door) in self.to_open_door_location_dic.items():
            room_number_in_dic = room_id.split("_")[1]

            if room_number == room_number_in_dic:
                return (door.location, door.door_id)
                break

    def _is_reachable(self, block_id):
        room = block_id.split("Block_in_")[1]
        room_number = room.split("_")[1]
        reachable = True

        for room_id in self.to_open_door_location_dic.keys():
            room_number_in_dic = room_id.split("_")[1]

            if room_number == room_number_in_dic:
                reachable = False
                break

        return reachable

    def _find_closest_found_block_in_room(self):
        closest_room_door = self.state.get_closest_room_door()
        current_room = closest_room_door[0]['room_name']
        current_room_number = current_room.split("_")[1]

        target_location = (0, 0)
        min_route_length = 10000000000000
        block_id = ""

        for (block_name, block) in self.found_goal_block_dic.items():
            block_room_number = block_name.split("_")[3]

            if (block_room_number == current_room_number):
                self.navigator.reset_full()
                block_location = block.location
                self.navigator.add_waypoint(block_location)
                route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

                if route_length < min_route_length:
                    target_location = block_location
                    min_route_length = route_length
                    block_id = block.id

        return (target_location, block_id)

    def _find_cluster_corresponding_goal_location_to_store(self, block):

        self._update_goal_drop_list()

        test = self.goal_block_drop_list.copy()
        test.reverse()

        for goal_block_drop in test:

            goal_block_dic_entry = self.goal_block_dict[goal_block_drop.id]

            if block.color == goal_block_dic_entry.color and block.shape == goal_block_dic_entry.shape and goal_block_dic_entry.stored == False:

                goal_block_dic_entry.stored = True
                self.goal_block_dict[goal_block_drop.id] = goal_block_dic_entry

                return (goal_block_dic_entry.location[0] - 1, goal_block_dic_entry.location[1])


    def _find_corresponding_goal_location_to_store(self, block):

        self._update_goal_drop_list()

        for goal_block_drop in self.goal_block_drop_list:

            goal_block_dic_entry = self.goal_block_dict[goal_block_drop.id]

            if block.color == goal_block_dic_entry.color and block.shape == goal_block_dic_entry.shape and goal_block_dic_entry.stored == False:

                goal_block_dic_entry.stored = True
                self.goal_block_dict[goal_block_drop.id] = goal_block_dic_entry

                return (goal_block_dic_entry.location[0] - 1, goal_block_dic_entry.location[1])

    def _check_if_agents_inventory_is_full(self, id_other_person):

        full_inventory = False
        items_in_inventory = 0

        for (location, block) in self.found_goal_block_location_list:
            if (location[0] == id_other_person):
                items_in_inventory = items_in_inventory + 1

                if (items_in_inventory == 3):
                    full_inventory = True
                    break

        return full_inventory

    def _check_if_agent_has_inventory(self, id_other_person):

        agent_has_inventory = False

        for (location, block) in self.found_goal_block_location_list:
            if (location[0] == id_other_person):
                agent_has_inventory = True
                break

        return agent_has_inventory

    def _check_if_all_blocks_dropped(self):

        all_dropped = True

        for (location, block) in self.found_goal_block_location_list:
            if (isinstance(location[0], int) == False):
                all_dropped = False
                break

        return all_dropped

        #self.agent.found_goal_block_location_list.append((("inventory", "inventory"), new_found_goal_block))

    def _find_location_of_closest_found_block(self, drop_block):

        min_route_length = 10000000000000
        target_location = ()
        block_id = None

        for (location, block) in self.found_goal_block_location_list:
            if (block.color == drop_block.color and block.shape == drop_block.shape):

                if(location[0] == "inventory"):
                    min_route_length = 0
                    target_location = location
                    block_id = block.id

                elif(((location[0] + 1) == drop_block.location[0]) and (location[1] == drop_block.location[1])):
                    return (location, block.id)

                else:
                    self.navigator.reset_full()
                    new_location = location
                    self.navigator.add_waypoint(new_location)
                    route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

                    if route_length < min_route_length:
                        target_location = new_location
                        min_route_length = route_length
                        block_id = block.id

        return (target_location, block_id)


    def _find_location_of_found_block(self, drop_block):

        for (location, block) in self.found_goal_block_location_list:
            if (block.color == drop_block.color and block.shape == drop_block.shape):
                return (location, block.id)

    def _determine_furthest_end_goal_location(self):
        target_location = (0, 0)
        max_route_length = -100

        for location in self.goal_block_location_list:
            self.navigator.reset_full()
            new_location = location
            self.navigator.add_waypoint(new_location)
            route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

            if route_length > max_route_length:
                target_location = new_location
                max_route_length = route_length

        return target_location

    def _determine_closest_end_goal_location(self, location_other_person):
        target_location = (0, 0)
        min_route_length = 10000000000000

        for location in self.goal_block_location_list:

            if(location_other_person == location):
                return(location_other_person)

            else:

                self.navigator.reset_full()
                new_location = location
                self.navigator.add_waypoint(new_location)
                route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

                if route_length < min_route_length:
                    target_location = new_location
                    min_route_length = route_length

        return target_location

    def _determine_closest_found_end_goal_location(self):
        target_location = (0, 0)
        min_route_length = 10000000000000
        block_id = ""

        for goal_block in self.inventory_dic.values():

            for end_goal in self.goal_block_dict.values():
                if (end_goal.shape == goal_block.shape and end_goal.color == goal_block.color and end_goal.stored == False):
                    new_location = end_goal.location

                    self.navigator.reset_full()
                    self.navigator.add_waypoint(new_location)
                    route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

                    if route_length < min_route_length:
                        target_location = new_location
                        min_route_length = route_length
                        block_id = goal_block.id

        return ((target_location[0] - 1, target_location[1]), block_id)


    def _determine_if_room_not_visited_by_either_color_or_shape_seer(self):
        exists = False

        # door_location, shape_blind_id, color_blind_id
        for entry in self.assigned_door_location_dic.values():
            if (entry[1] == None or entry[2] == None):
                exists = True
                break

        return exists

    # location, color_seer_id, shape_seer_id
    def _determine_if_room_not_visited_by_color_seer(self):

        exists = False

        # door_location, shape_blind_id, color_blind_id
        for entry in self.assigned_door_location_dic.values():
            if (entry[1] == None):
                exists = True
                break

        return exists

    def _check_if_room_is_assigned(self, door_location):
        assigned = False

        # door_location, shape_blind_id, color_blind_id
        for entry in self.assigned_door_location_dic.values():
            if (entry[0] == door_location):
                assigned = True
                break

        return assigned

    # location, color_seer_id, shape_seer_id
    def _determine_if_room_not_visited_by_shape_seer(self):

        exists = False

        # door_location, shape_blind_id, color_blind_id
        for entry in self.assigned_door_location_dic.values():
            if (entry[2] == None):
                exists = True
                break

        return exists

    # location, color_seer_id, shape_seer_id
    def _determine_if_room_not_visited_by_one_kind_of_seer(self):

        exists = False

        # door_location, agent_id, "shape_blind", None, ""
        for entry in self.assigned_door_location_dic.values():
            if (entry[1] == None or entry[2] == None):
                exists = True
                break

        return exists


    def _determine_closest_door_location_and_id(self):
        target_location = (0,0)
        min_route_length = 10000000000000
        door_id = ""

        for door_location in self.to_open_door_location_dic.values():
            #TODO here we might override any other move commands
            self.navigator.reset_full()
            new_location = door_location.location
            self.navigator.add_waypoint(new_location)
            route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

            if route_length < min_route_length:
                target_location = new_location
                min_route_length = route_length
                door_id = door_location.door_id

        return (target_location, door_id)

    def _determine_closest_door_location_and_id_specific_agent(self, agent_id):
        navigator = Navigator(agent_id=agent_id, action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        #TODO: This might now work if the leader is far away from the other agents - the StateTracker might not know their locations
        #state_tracker = StateTracker(agent_id=agent_id)

        target_location = (0, 0)
        min_route_length = 10000000000000
        door_id = ""

        for door_location in self.to_open_door_location_dic.values():
            # TODO here we might override any other move commands
            navigator.reset_full()
            new_location = door_location.location
            navigator.add_waypoint(new_location)
            route_length = len(navigator._Navigator__get_route(self.state_tracker))

            if route_length < min_route_length:
                target_location = new_location
                min_route_length = route_length
                door_id = door_location.door_id

        return (target_location, door_id, min_route_length)

    def _determine_if_block_to_pick_up_found(self):

        new_dictionary = {}
        for (key, found_goal) in self.found_goal_block_dic.items():
            if key not in self.found_goal_block_list:
                new_dictionary[key] = found_goal

        self.found_goal_block_dic = new_dictionary
        result = (len(self.found_goal_block_dic.values()) > 0)

        return result

    def _determine_if_potential_block_to_check_out_found(self):

        return (len(self.potential_goal_blocks_dic.items()) > 0)


    def _determine_closest_door_with_found_block(self):
        target_location = (0, 0)
        min_route_length = 10000000000000
        door_id = ""

        for found_block in self.found_goal_block_dic.values():
            room = found_block.id.split("Block_in_")[1]
            room_number_of_block = room.split("_")[1]

            for (room_id, door) in self.to_open_door_location_dic.items():
                room_number_in_dic = room_id.split("_")[1]

                if room_number_of_block == room_number_in_dic:
                    self.navigator.reset_full()
                    door_location = door.location
                    self.navigator.add_waypoint(door_location)
                    route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

                    if route_length < min_route_length:
                        target_location = door_location
                        min_route_length = route_length
                        door_id = room_id

        return (target_location, door_id)

    def _determine_closest_room_not_visited_by_fully_capable(self, agent_location):

        target_location = (0, 0)
        target_door_id = ""
        min_route_length = 10000000000000

        # door_location, agent_id, "shape_blind", None, ""
        for (door_id, entry) in self.assigned_door_location_dic.items():
            if (entry[1] == None or entry[2] == None):

                block_location = entry[0]
                x_difference = abs(agent_location[0] - block_location[0])
                y_difference = abs(agent_location[1] - block_location[1])
                route_length = x_difference + y_difference

                if route_length < min_route_length:
                    target_location = block_location
                    min_route_length = route_length
                    target_door_id = door_id

        return (target_location, target_door_id)

    # location, color_seer_id, shape_seer_id
    def _determine_closest_room_not_visited_by_color_seer(self, agent_location):

        target_location = (0, 0)
        target_door_id = ""
        min_route_length = 10000000000000

        # door_location, agent_id, "shape_blind", None, ""
        for (door_id, entry) in self.assigned_door_location_dic.items():
            if (entry[1] == None):

                block_location = entry[0]
                x_difference = abs(agent_location[0] - block_location[0])
                y_difference = abs(agent_location[1] - block_location[1])
                route_length = x_difference + y_difference

                if route_length < min_route_length:
                    target_location = block_location
                    min_route_length = route_length
                    target_door_id = door_id

        return (target_location, target_door_id)

    # location, color_seer_id, shape_seer_id
    def _determine_closest_room_not_visited_by_shape_seer(self, agent_location):

        target_location = (0, 0)
        target_door_id = ""
        min_route_length = 10000000000000

        # door_location, agent_id, "shape_blind", None, ""
        for (door_id, entry) in self.assigned_door_location_dic.items():
            if (entry[2] == None):

                block_location = entry[0]
                x_difference = abs(agent_location[0] - block_location[0])
                y_difference = abs(agent_location[1] - block_location[1])
                route_length = x_difference + y_difference

                if route_length < min_route_length:
                    target_location = block_location
                    min_route_length = route_length
                    target_door_id = door_id

        return (target_location, target_door_id)

    def _no_one_still_under_way(self):
        return (len(self.agents_who_currently_empty_their_inventory.values()) == 0)

    def _determine_location_stored_block(self, block):

        pick_up_location = ()
        stored_block_id = ""
        to_pop_entry = 0

        for (location, stored_block) in self.found_goal_block_location_list:
            if stored_block.color == block.color and stored_block.shape == block.shape:
                pick_up_location = location
                stored_block_id = stored_block.id
                break
            else:
                to_pop_entry = to_pop_entry + 1

        self.found_goal_block_location_list.pop(to_pop_entry)

        return(pick_up_location, stored_block_id)



    def _determine_closest_reachable_goal_block_location_and_id_with_agent_location(self, reachable_blocks, agent_location):

        target_location = (0, 0)
        min_route_length = 10000000000000
        block_id = ""
        result_block = None

        for reachable_block_id in reachable_blocks:
            for found_block in self.found_goal_block_dic.values():
                if (found_block.id == reachable_block_id):

                    block_location = found_block.location
                    x_difference = abs(agent_location[0] - block_location[0])
                    y_difference = abs(agent_location[1] - block_location[1])
                    route_length = x_difference + y_difference

                    if route_length < min_route_length:
                        target_location = block_location
                        min_route_length = route_length
                        block_id = found_block.id
                        result_block = found_block

                    break

        return (target_location, block_id, result_block)

    def _determine_closest__goal_block_location_and_id_from_list(self, block_list):
        target_location = (0, 0)
        min_route_length = 10000000000000
        block_id = ""

        for possible_block in block_list:
            for found_block in self.found_goal_block_dic.values():
                if (found_block.id == possible_block.id):
                    self.navigator.reset_full()
                    block_location = found_block.location
                    self.navigator.add_waypoint(block_location)
                    route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

                    if route_length < min_route_length:
                        target_location = block_location
                        min_route_length = route_length
                        block_id = found_block.id

        return (target_location, block_id)


    def _determine_closest_reachable_goal_block_location_and_id(self, reachable_blocks):
        target_location = (0, 0)
        min_route_length = 10000000000000
        block_id = ""

        for reachable_block_id in reachable_blocks:
            for found_block in self.found_goal_block_dic.values():
                if (found_block.id == reachable_block_id):
                    self.navigator.reset_full()
                    block_location = found_block.location
                    self.navigator.add_waypoint(block_location)
                    route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

                    if route_length < min_route_length:
                        target_location = block_location
                        min_route_length = route_length
                        block_id = found_block.id

        return (target_location, block_id)

    def _determine_closest_goal_block_location_and_id(self):

        target_location = (0, 0)
        min_route_length = 10000000000000
        block_id = ""

        for found_block in self.found_goal_block_dic.values():
            self.navigator.reset_full()
            block_location = found_block.location
            self.navigator.add_waypoint(block_location)
            route_length = len(self.navigator._Navigator__get_route(self.state_tracker))

            if route_length < min_route_length:
                target_location = block_location
                min_route_length = route_length
                block_id = found_block.id

        return (target_location, block_id)

    def _clean_room_exploration_list(self):

        new_room_exploration_list = []

        for entry in self.room_exploration_moves:
            new_room_exploration_list.append((entry[0], {}))

        self.room_exploration_moves = new_room_exploration_list

    def _check_if_inventory_is_full(self):
        self.inventory_should_be_emptied = (len(self.inventory_dic.values()) == 3)

class Block:
    id = ""
    location = (0, 0)

    color = "#empty"
    shape = -1

    still_needed = False

    def __init__(self):
        None

    def toJSON(self):
        return str(self.id) + "," + str(self.location[0]) + "," + str(self.location[1]) + ","+ str(self.color) + "," + str(self.shape)

class GoalBlock(Block):
    picked_up = False
    finished = False
    stored = False

    def __init__(self):
        super(GoalBlock, self).__init__()

    def toJSON(self):
        return str(self.id) + "," + str(self.location )+ "," + str(self.color) + "," + str(self.shape) + "," + str(self.picked_up) + "," + str(self.finished) + "," + str(self.stored)

class Door_Location:
    location = (0, 0)
    door_id = ""

    def __init__(self):
        None

class Agent:
    id = ""
    last_seen_location = (0,0)

    color_blind = False
    shape_blind = False
    speed = 1

    vote = None
    count = 0

    cluster_protocol_count = 0

    cluster = None

    def __init__(self):
        None

class Task:
    agent = None
    last_position = (0,0)
    success = False

    def __init__(self, agent: Team36Agent):
        self.agent = agent

    def environment_check(self, state):
        None

    def get_action(self):
        None

#completly overrides every other move command
class GoToLocationTask(Task):
    target_location = ()
    first_time = True

    def __init__(self, agent: Team36Agent, target_location):
        super().__init__(agent)
        self.target_location = target_location
        self.initialize()

    def initialize(self):

        self.navigator = Navigator(agent_id=self.agent.agent_id, action_set=self.agent.action_set, algorithm=Navigator.A_STAR_ALGORITHM)
        self.navigator.reset_full()
        self.navigator.add_waypoint(self.target_location)

        self.last_position = self.target_location

    def environment_check(self, state):
        if (self.agent.location == self.target_location):
            self.success = True
            self.agent._update_task_queue()

    def get_action(self, state):

        if self.first_time:
            self.initialize()
            self.first_time = False

        return (self.navigator.get_move_action(self.agent.state_tracker), {})

#assumes you are standing right in front of the door i.e. current location = (doorLocation[0], doorLocation[1] - 1)
class OpenDoorTask(Task):
    door_id = ""

    def __init__(self, agent: Team36Agent, door_id):
        super().__init__(agent)
        self.door_id = door_id

    def environment_check(self, state):
        door_tiles = state[{'class_inheritance': 'Door'}]
        for door in door_tiles:
            if(door['obj_id'] == self.door_id):
                if(door['is_open'] == True):

                    #for the one agent case
                    if(self.door_id in self.agent.to_open_door_location_dic):
                        self.agent.to_open_door_location_dic.pop(self.door_id)

                    self.success = True
                    self.agent._update_task_queue()

    def get_action(self, state):
        return (OpenDoorAction.__name__, {'object_id': self.door_id})

#assumes you are standing on the block
class PickUpBlockTask(Task):
    block_id = ""

    def __init__(self, agent: Team36Agent, block_id):
        super().__init__(agent)
        self.block_id = block_id

        if (self.agent.alone):
            self.agent.found_goal_block_list.append(block_id)
            self.agent.interruptable = False

    def environment_check(self, state):
        self.success = True
        self.agent._update_task_queue()

    def get_action(self, state):

        seen_blocks = state[{'class_inheritance': 'CollectableBlock'}]

        if (self.agent.alone):
            if (seen_blocks != None):
                if (self.agent.color_blind == False and self.agent.shape_blind == False):
                    if (isinstance(seen_blocks, list)):
                        for block in seen_blocks:
                            if ((block['obj_id'] == self.block_id)):
                                new_found_goal_block = Block()
                                new_found_goal_block.shape = block['visualization']['shape']
                                new_found_goal_block.color = block['visualization']['colour']
                                new_found_goal_block.id = self.block_id

                                self.agent.inventory_dic[self.block_id] = new_found_goal_block
                                self.agent.found_goal_block_location_list.append((("inventory", "inventory"), new_found_goal_block))

                                for goal_block in self.agent.goal_block_dict.values():
                                    if (goal_block.shape == new_found_goal_block.shape and goal_block.color == new_found_goal_block.color
                                            and goal_block.picked_up == False):
                                        goal_block.picked_up = True
                                        break

                                new_found_goal_block_dic = {}
                                for goal_block in self.agent.found_goal_block_dic.values():
                                    #check if block is not weird case where already picked up blocks pop up again
                                    if (goal_block.id not in self.agent.found_goal_block_list):

                                        #check if we still need this block
                                        for end_block in self.agent.goal_block_dict.values():
                                            if (goal_block.shape == end_block.shape and goal_block.color == end_block.color and end_block.picked_up == False):
                                                new_found_goal_block_dic[goal_block.id] = goal_block
                                                break;

                                self.agent.found_goal_block_dic = new_found_goal_block_dic

                    else:
                        if ((seen_blocks['obj_id'] == self.block_id)):
                            new_found_goal_block = Block()
                            new_found_goal_block.shape = seen_blocks['visualization']['shape']
                            new_found_goal_block.color = seen_blocks['visualization']['colour']
                            new_found_goal_block.id = self.block_id

                            self.agent.inventory_dic[self.block_id] = new_found_goal_block
                            self.agent.found_goal_block_location_list.append((("inventory", "inventory"), new_found_goal_block))

                            for goal_block in self.agent.goal_block_dict.values():
                                if (goal_block.shape == new_found_goal_block.shape and goal_block.color == new_found_goal_block.color and goal_block.picked_up == False):
                                    goal_block.picked_up = True
                                    break

                            new_found_goal_block_dic = {}
                            for goal_block in self.agent.found_goal_block_dic.values():
                                # check if block is not weird case where already picked up blocks pop up again
                                if (goal_block.id not in self.agent.found_goal_block_list):

                                    # check if we still need this block
                                    for end_block in self.agent.goal_block_dict.values():
                                        if (goal_block.shape == end_block.shape and goal_block.color == end_block.color):
                                            new_found_goal_block_dic[goal_block.id] = goal_block
                                            break;

                            self.agent.found_goal_block_dic = new_found_goal_block_dic

                else:
                    if (isinstance(seen_blocks, list)):
                        for block in seen_blocks:
                            if ((block['obj_id'] == self.block_id)):
                                new_found_goal_block = Block()

                                if (self.agent.shape_blind):
                                    new_found_goal_block.shape = self.agent.found_goal_block_dic[self.block_id].shape
                                else:
                                    new_found_goal_block.shape = block['visualization']['shape']
                                if (self.agent.color_blind):
                                    new_found_goal_block.color = self.agent.found_goal_block_dic[self.block_id].color
                                else:
                                    new_found_goal_block.color = block['visualization']['colour']

                                new_found_goal_block.id = self.block_id

                                self.agent.inventory_dic[self.block_id] = new_found_goal_block
                                self.agent.found_goal_block_location_list.append(
                                    (("inventory", "inventory"), new_found_goal_block))

                                for goal_block in self.agent.goal_block_dict.values():
                                    if (
                                            goal_block.shape == new_found_goal_block.shape and goal_block.color == new_found_goal_block.color
                                            and goal_block.picked_up == False):
                                        goal_block.picked_up = True
                                        break

                                new_found_goal_block_dic = {}
                                for goal_block in self.agent.found_goal_block_dic.values():
                                    # check if block is not weird case where already picked up blocks pop up again
                                    if (goal_block.id not in self.agent.found_goal_block_list):

                                        # check if we still need this block
                                        for end_block in self.agent.goal_block_dict.values():
                                            if (
                                                    goal_block.shape == end_block.shape and goal_block.color == end_block.color and end_block.picked_up == False):
                                                new_found_goal_block_dic[goal_block.id] = goal_block
                                                break;

                                self.agent.found_goal_block_dic = new_found_goal_block_dic

                    else:
                        if ((seen_blocks['obj_id'] == self.block_id)):
                            new_found_goal_block = Block()

                            if (self.agent.shape_blind):
                                new_found_goal_block.shape = self.agent.found_goal_block_dic[self.block_id].shape
                            else:
                                new_found_goal_block.shape = seen_blocks['visualization']['shape']
                            if (self.agent.color_blind):
                                new_found_goal_block.color = self.agent.found_goal_block_dic[self.block_id].color
                            else:
                                new_found_goal_block.color = seen_blocks['visualization']['colour']

                            new_found_goal_block.id = self.block_id

                            self.agent.inventory_dic[self.block_id] = new_found_goal_block
                            self.agent.found_goal_block_location_list.append(
                                (("inventory", "inventory"), new_found_goal_block))

                            for goal_block in self.agent.goal_block_dict.values():
                                if (
                                        goal_block.shape == new_found_goal_block.shape and goal_block.color == new_found_goal_block.color and goal_block.picked_up == False):
                                    goal_block.picked_up = True
                                    break

                            new_found_goal_block_dic = {}
                            for goal_block in self.agent.found_goal_block_dic.values():
                                # check if block is not weird case where already picked up blocks pop up again
                                if (goal_block.id not in self.agent.found_goal_block_list):

                                    # check if we still need this block
                                    for end_block in self.agent.goal_block_dict.values():
                                        if (goal_block.shape == end_block.shape and goal_block.color == end_block.color):
                                            new_found_goal_block_dic[goal_block.id] = goal_block
                                            break;

                            self.agent.found_goal_block_dic = new_found_goal_block_dic


        return (GrabObject.__name__, {'object_id': self.block_id})

#assumes you are standing right in front of the door i.e. current location = (doorLocation[0], doorLocation[1] - 1)
class ExploreRoomTask(Task):
    room_exploration_moves = []
    first_time = True

    def __init__(self, agent: Team36Agent):
        super().__init__(agent)
        self.room_exploration_moves = agent.room_exploration_moves.copy()
        self.agent.interruptable = False

    def environment_check(self, state):
        if (len(self.room_exploration_moves) == 0):

            self.agent._clean_room_exploration_list()
            self.success = True

            if (self.agent.alone):
                self.agent.interruptable = True
            self.agent._update_task_queue()

    def get_action(self, state):
        if (self.first_time):
            self.agent._clean_room_exploration_list()
            self.first_time = False

        move = self.room_exploration_moves.pop()
        cleaned_move = ((move[0], {}))

        return cleaned_move

class DropBlockTask(Task):
    block_id = ""

    def __init__(self, agent: Team36Agent, block_id):
        super().__init__(agent)
        self.block_id = block_id

    def environment_check(self, state):
        self.success = True
        self.agent._update_task_queue()

    def get_action(self, state):

        if (self.agent.alone):

            current_location = self.agent.state_tracker.get_memorized_state()[self.agent.agent_id]['location']

            old_location = None
            old_block = None

            for (location, block) in self.agent.found_goal_block_location_list:
                if (block.id == self.block_id):
                    old_location = location
                    old_block = block
                    if (location[0] == "inventory"):
                        self.agent.inventory_dic.pop(block.id)
                    #why?

                    # if current_location in self.agent.goal_block_location_list:
                    #     self.agent.found_goal_block_location_list.remove((location, block))

                    # to_update_entry = None
                    # for goal_block in self.agent.goal_block_dict.values():
                    #     if (goal_block.location == current_location):
                    #         to_update_entry = goal_block
                    #         to_update_entry.finished = True
                    #         break
                    #     elif (goal_block.color == old_block.color and goal_block.shape == old_block.shape and goal_block.stored == False):
                    #         to_update_entry = goal_block
                    #         to_update_entry.stored = True
                    #         break
                    #
                    # if(to_update_entry != None):
                    #     self.agent.goal_block_dict[to_update_entry.id] = to_update_entry
                    #
                    # break

                    for goal_block in self.agent.goal_block_dict.values():
                        if (goal_block.location == current_location):
                            goal_block.finished = True
                            break
                        elif (goal_block.color == old_block.color and goal_block.shape == old_block.shape and goal_block.stored == False):
                            goal_block.stored = True
                            break


            self.agent.found_goal_block_location_list.remove((old_location, old_block))
            self.agent.found_goal_block_location_list.append((current_location, old_block))

            still_in_inventory = 0
            for (location, block) in self.agent.found_goal_block_location_list:
                if (location[0] == "inventory"):
                    still_in_inventory = still_in_inventory + 1
            space_in_inventory = 3 - still_in_inventory

            still_to_be_found = 0
            for goal_block in self.agent.goal_block_dict.values():
                if (goal_block.picked_up == False):
                    still_to_be_found = still_to_be_found + 1

            if((still_to_be_found > space_in_inventory) and (still_in_inventory > 0)):
                self.agent.inventory_should_be_emptied = True
                self.agent.inventory_emptying_under_way = True
            else:
                self.agent.inventory_should_be_emptied = False
                self.agent.inventory_emptying_under_way = False


        return (DropObject.__name__, {'object_id': self.block_id})

class WaitTask(Task):
    duration = 0

    def __init__(self, agent: Team36Agent, duration):
        super().__init__(agent)
        self.duration = duration

    def environment_check(self, state):

            if (self.duration == 0):
                self.success = True
                self.agent.interruptable = False
                self.agent._update_task_queue()
            else:
                self.duration = self.duration - 1

    def get_action(self, state):
        return (None ,{})
