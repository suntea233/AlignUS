import copy
from convlab.nlg.template.multiwoz.nlg import slot2word

from convlab.dialog_agent import Agent
from convlab.policy.policy import Policy
from convlab.task.multiwoz.goal_generator import GoalGenerator
# from convlab.policy.genTUS.unify.Goal import Goal
from convlab2.policy.user_sim.multiwoz.goal import Goal
from convlab.policy.AlignUS.stepGenTUS import UserPolicy
# from convlab.policy.genTUS.stepGenTUS import UserPolicy
from convlab.policy.AlignUS.base_model import FlanT5, ChatGPT, ChatGLM2, LLAMA, LLAMA2


word2slot = {value.capitalize(): key.capitalize() for key, value in slot2word.items()}
old_slot_map = {
    'addr': "address",
    'post': "postcode",
    'price': "price range",
    'arrive': "arrive by",
    'leave': "leave at",
    'depart': "departure",
    'dest': "destination",
    'fee': "entrance fee",
    'open': 'open hours',
    # 'car': "type",
    'ticket': 'price',
    'id': 'train id',
    'people': 'book people',
    'stay': 'book stay',
    'time': 'duration',
    'none': '',
}

slot_map = {value.capitalize(): key.capitalize() for key, value in old_slot_map.items()}


class AlignUS(Agent):
    def __init__(self,print_details=True,large_name='chatgpt'):
        self.reward_func = lambda goal, dialog, completed: 40 if completed else -20
        self.opponent_name = 'sys'
        self.name = 'usr'
        self.large_name = large_name
        self.domain_intent = ['inform', 'request']
        self.general_intent = ['thank', 'bye']
        self.policy = SimulatorPolicy(print_details=print_details)
        self.small = UserPolicy(mode="semantic",dataset='multiwoz21',only_action=False)

        if large_name == "flant5":
            self.large = FlanT5()

        elif large_name == "chatgpt":
            self.large = ChatGPT()

        elif large_name == "chatglm2":
            self.large = ChatGLM2()

        elif large_name == "llama":
            self.large = LLAMA()

        elif large_name == 'llama2':
            self.large = LLAMA2()

        else:
            raise NotImplementedError


    def init_session(self, **kwargs):
        self.policy.init_session(**kwargs)
        self.history = []

    def parse_ans(self, text, choice, raw_list=None):
        try:
            if raw_list:
                if text:
                    raw_text = text[0].upper()
                    raw_text = raw_text.replace(".", "")
                    return [raw_list[ord(raw_text) - 65]]
                else:
                    return [['thank','general','none','none']]
            else:
                import re
                text = text[0].upper()
                text = text.replace(".", "")
                pattern = r'{}\.(\w+)'.format(text)
                matches = re.findall(pattern, choice)
                return matches
        except:
            for i in range(len(raw_list)):
                if f'{chr(65 + i)}.'in text:
                    return [raw_list[i]]

    def convert_old_2_slot(self, observation):
        temp = copy.deepcopy(observation)
        for i in range(len(temp)):
            if temp[i][1].lower() == 'taxi' and temp[i][2].lower() == 'type':
                temp[i][2] = 'Car'
            if temp[i][2] in slot_map:
                temp[i][2] = slot_map[temp[i][2]]
        return temp


    def response(self,observation):
        self.input_action = observation
        self.goal_status = self.policy.goal.get_goal_list()

        goal_status = [[item.replace('info', 'inform').replace('reqt', 'request') for item in inner_list] for
                            inner_list in self.goal_status]

        complete = [[status[0],status[1],status[2],status[3]] for status in goal_status if status[-1] != "fulfilled"]

        dialog_action, utterance = self.small.predict(observation,goal_status)

        self.output_action = dialog_action

        judge = self.large.generate('A = {}\nB = {}\nIs B in A?\nJUST ANSWER YES OR NO.'.format(complete,dialog_action))

        if "no" in judge.lower():
            multi_choice = " ".join([f"{chr(65 + i)}.{item}" for i, item in enumerate(complete)])

            alphabet = ' or '.join([f"{chr(65 + i)}" for i in range(len(complete))])

            instruction = "Choose one from the following candidates that matches the observation:\n{}\n".format(complete) + multi_choice + "\nJUST ANSWER {}.\n".format(alphabet)

            answer = self.large.generate(instruction)

            try:
                ans = self.parse_ans(text=answer, choice=multi_choice, raw_list=complete)
                if ans != None:
                    self.output_action = ans
            except:
                pass

        if self.output_action == None:
            self.output_action == [['thank','general','none','none']]

        self.output_action = self.convert_old_2_slot(self.output_action)

        # self.natural_language = self.large.generate("Please act as a CUSTOMER and rewrite this sentence and make it smoother, more advanced, and more conversational: {}".format(self.output_action,utterance))

        self.natural_language = utterance

        self.policy.predict(
            sys_act=self.input_action, usr_act=self.output_action)

        return self.output_action

    def get_in_da(self):
        return self.input_action

    def get_out_da(self):
        return self.output_action

    def is_terminated(self):
        if hasattr(self.policy, 'is_terminated'):
            return self.policy.is_terminated()
        return None


class SimulatorPolicy(Policy):
    def __init__(self, print_details=True):
        super().__init__()
        self.print_details = print_details
        self.goal_generator = GoalGenerator()
        self.max_turn = 40
        self.max_initiative = 4
        self.sys_acts = []


    def get_goal(self):
        return self.domain_goals


    def reset_turn(self):
        self.__turn = 0


    def init_session(self, goal=None):
        self.reset_turn()
        if not goal:
            self.goal = Goal(
                goal_generator=self.goal_generator,
                print_details=self.print_details
            )
            self.goal_in_natural_language = self.goal_generator.build_message(
                self.goal.goal_to_transform)
        else:
            self.goal = Goal(goal.domain_goals)
            self.goal_in_natural_language = ''

        self.domain_goals = self.goal.domain_goals
        self.sys_acts = []
        self.usr_acts = []
        self.terminated = False
        self.mode = "semantic"
        self.time_step = 0
        self.max_history = 4


    def _no_offer(self, system_in):
        for intent, domain, slot, value in system_in:
            if intent.lower() == "nooffer":
                self.terminated = True
                return True
            else:
                return False

    # from tus
    def predict(self, sys_act, usr_act):
        # allow_general_intent = False
        # self.model.eval()

        # if not self.add_sys_from_reward:
        self.goal.update_user_goal(action=sys_act, char="sys")
        self.sys_acts.append(sys_act)  # for terminate conversation

        # update constraint
        self.time_step += 2

        self.semantic_action = usr_act
        # TODO
        if self.is_finish():
            self.semantic_action, self.utterance = self._good_bye()

        self.goal.update_user_goal(action=self.semantic_action, char="usr")
        # self.vector.update_mentioned_domain(self.semantic_action)
        self.usr_acts.append(self.semantic_action)

        if self.mode == "language":
            # print("in", sys_act)
            # print("out", self.utterance)
            return self.utterance
        else:
            return self.semantic_action


    def is_finish(self):
        # stop by model generation?
        if self._finish_conversation_rule():
            self.terminated = True
            return True
        elif self._usr_terminate():
            self.terminated = True
            return True
        self.terminated = False
        return False


    def is_success(self):
        task_complete = self.goal.task_complete()
        # goal_status = self.goal.all_mentioned()
        # should mentioned all slots
        if task_complete:  # and goal_status["complete"] > 0.6:
            return True
        return False


    def _good_bye(self):
        if self.is_success():
            return [['thank', 'general', 'none', 'none']], "thank you. bye"
        else:
            return [["bye", "general", "None", "None"]], "bye"


    def _finish_conversation_rule(self):
        if self.is_success():
            return True

        if self.time_step > self.max_turn:
            return True

        if (len(self.sys_acts) > 4) and (
                self.sys_acts[-1] == self.sys_acts[-2]) and (
                self.sys_acts[-2] == self.sys_acts[-3]):  # noqa
            return True
        return False


    def is_terminated(self):
        # Is there any action to say?
        self.is_finish()
        return self.terminated


    def _usr_terminate(self):
        for act in self.semantic_action:
            if act[0] in ['thank', 'bye']:
                return True
        return False