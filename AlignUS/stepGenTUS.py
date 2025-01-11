import json
import os
import sys
import logging
import torch
from transformers import BartTokenizer,AutoTokenizer

import zipfile
from convlab.policy.BlendUS.ppo.vector import stepGenTUSVector
from convlab.policy.BlendUS.stepGenTUSmodel import stepGenTUSmodel
from convlab.policy.BlendUS.token_map import tokenMap
from convlab.policy.BlendUS.unify.knowledge_graph import KnowledgeGraph
from convlab.policy.policy import Policy
from convlab.util.file_util import cached_path


def model_downloader(download_dir, model_path):
    """
    Function to download models from web url
    :param download_dir: Directory where models should be downloaded
    :param model_path: URL/Path of the model
    """
    logging.info('Load from model_file param')
    model_path = cached_path(model_path)
    archive = zipfile.ZipFile(model_path, 'r')
    archive.extractall(download_dir)
    archive.close()




DEBUG = False


class UserActionPolicy(Policy):
    def __init__(self, model_checkpoint, mode="semantic", only_action=True, max_turn=40, **kwargs):
        self.mode = mode
        # if mode == "semantic" and only_action:
        #     # only generate semantic action in prediction
        print("model_checkpoint", model_checkpoint)
        self.only_action = only_action
        if self.only_action:
            print("change mode to semantic because only_action=True")
            self.mode = "semantic"
        self.max_in_len = 500
        self.max_out_len = 100 if only_action else 200
        max_act_len = kwargs.get("max_act_len", 2)
        # print("max_act_len", max_act_len)
        self.max_action_len = max_act_len
        if "max_act_len" in kwargs:
            self.max_out_len = 30 * self.max_action_len
            print("max_act_len", self.max_out_len)
        self.max_turn = max_turn
        if mode not in ["semantic", "language"]:
            print("Unknown user mode")

        self.reward = {"success":  self.max_turn*2,
                       "fail": self.max_turn*-1}
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        train_whole_model = kwargs.get("whole_model", True)
        self.model = stepGenTUSmodel(
            model_checkpoint, train_whole_model=train_whole_model)
        self.model.eval()
        self.model.to(self.device)
        self.model.share_memory()

        self.turn_level_reward = kwargs.get("turn_level_reward", True)
        self.cooperative = kwargs.get("cooperative", True)

        dataset = kwargs.get("dataset", "")
        self.kg = KnowledgeGraph(
            tokenizer=self.tokenizer,
            dataset=dataset)


        self.vector = stepGenTUSVector(
            model_checkpoint, self.max_in_len, self.max_out_len)
        self.norm_reward = False

        self.action_penalty = kwargs.get("action_penalty", False)
        self.usr_act_penalize = kwargs.get("usr_act_penalize", 0)
        self.update_mode = kwargs.get("update_mode", "normal")
        self.max_history = kwargs.get("max_history", 3)
        self.init_session()

    def load(self, model_path):
        self.model.load_state_dict(torch.load(
            model_path, map_location=self.device))

    def _update_seq(self, sub_seq: list, pos: int):
        for x in sub_seq:
            self.seq[0, pos] = x
            pos += 1

        return pos

    def _generate_action(self, raw_inputs, mode="max", allow_general_intent=True):
        # TODO no duplicate
        self.kg.parse_input(raw_inputs)
        model_input = self.vector.encode(raw_inputs, self.max_in_len)
        # start token
        self.seq = torch.zeros(1, self.max_out_len, device=self.device).long()
        pos = self._update_seq([0], 0)
        pos = self._update_seq(self.token_map.get_id('start_json'), pos)
        pos = self._update_seq(self.token_map.get_id('start_act'), pos)

        # get semantic actions
        for act_len in range(self.max_action_len):
            pos = self._get_semantic_action(
                model_input, pos, mode, allow_general_intent)

            terminate, token_name = self._stop_semantic(
                model_input, pos, act_len)
            pos = self._update_seq(self.token_map.get_id(token_name), pos)

            if terminate:
                break

        if self.only_action:
            # return semantic action. Don't need to generate text
            return self.vector.decode(self.seq[0, :pos])

        # TODO remove illegal action here?

        # get text output
        pos = self._update_seq(self.token_map.get_id("start_text"), pos)

        text = self._get_text(model_input, pos, mode)

        return text


    def _get_text(self, model_input, pos, mode="max"):
        s_pos = pos
        mode = "sample"
        for i in range(s_pos, self.max_out_len):
            next_token_logits = self.model.get_next_token_logits(
                model_input, self.seq[:1, :pos])
            if mode == "sample":
                s = torch.multinomial(torch.softmax(
                    next_token_logits, dim=-1), 1)
                next_token = s
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
            if self._stop_text(next_token):
                # text = self.vector.decode(self.seq[0, s_pos:pos])
                # text = self._norm_str(text)
                # return self.vector.decode(self.seq[0, :s_pos]) + text + '"}'
                break

            pos = self._update_seq([next_token], pos)
        text = self.vector.decode(self.seq[0, s_pos:pos])
        text = self._norm_str(text)
        return self.vector.decode(self.seq[0, :s_pos]) + text + '"}'
        # TODO return None

    def _stop_text(self, next_token):
        if next_token == self.token_map.get_id("end_json")[0]:
            return True
        elif next_token == self.token_map.get_id("end_json_2")[0]:
            return True

        return False

    @staticmethod
    def _norm_str(text: str):
        text = text.strip('"')
        text = text.replace('"', "'")
        text = text.replace('\\', "")
        return text

    def _stop_semantic(self, model_input, pos, act_length=0):

        outputs = self.model.get_next_token_logits(
            model_input, self.seq[:1, :pos])
        tokens = {}
        for token_name in ['sep_act', 'end_act']:
            tokens[token_name] = {
                "token_id": self.token_map.get_id(token_name)}
            hash_id = tokens[token_name]["token_id"][0]
            tokens[token_name]["score"] = outputs[:, hash_id].item()

        if tokens['end_act']["score"] > tokens['sep_act']["score"]:
            terminate = True
        else:
            terminate = False

        if act_length >= self.max_action_len - 1:
            terminate = True

        token_name = "end_act" if terminate else "sep_act"

        return terminate, token_name

    def _get_semantic_action(self, model_input, pos, mode="max", allow_general_intent=True):

        intent = self._get_intent(
            model_input, self.seq[:1, :pos], mode, allow_general_intent)
        pos = self._update_seq(intent["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get domain
        domain = self._get_domain(
            model_input, self.seq[:1, :pos], intent["token_name"], mode)
        pos = self._update_seq(domain["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get slot
        slot = self._get_slot(
            model_input, self.seq[:1, :pos], intent["token_name"], domain["token_name"], mode)
        if "book" in slot["token_name"]:
            pos = self._update_seq(self.token_map.get_id('book'), pos)
            slot = self._get_book_slot(
                model_input, self.seq[:1, :pos], intent["token_name"], domain["token_name"], mode)
            slot["token_name"] = "book" + slot["token_name"]
        pos = self._update_seq(slot["token_id"], pos)
        pos = self._update_seq(self.token_map.get_id('sep_token'), pos)

        # get value

        value = self._get_value(
            model_input, self.seq[:1, :pos], intent["token_name"], domain["token_name"], slot["token_name"], mode)
        pos = self._update_seq(value["token_id"], pos)

        return pos

    def _get_intent(self, model_input, generated_so_far, mode="max", allow_general_intent=True):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_intent(next_token_logits, mode, allow_general_intent)

    def _get_domain(self, model_input, generated_so_far, intent, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_domain(next_token_logits, intent, mode)

    def _get_slot(self, model_input, generated_so_far, intent, domain, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        is_mentioned = self.vector.is_mentioned(domain)
        return self.kg.get_slot(next_token_logits, intent, domain, mode, is_mentioned)

    def _get_book_slot(self, model_input, generated_so_far, intent, domain, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)
        is_mentioned = self.vector.is_mentioned(domain)
        return self.kg.get_book_slot(next_token_logits, intent, domain, mode, is_mentioned)

    def _get_value(self, model_input, generated_so_far, intent, domain, slot, mode="max"):
        next_token_logits = self.model.get_next_token_logits(
            model_input, generated_so_far)

        return self.kg.get_value(next_token_logits, intent, domain, slot, mode)

    def _remove_illegal_action(self, action):
        # Transform illegal action to legal action
        new_action = []
        for act in action:
            if len(act) == 4:
                if "<?>" in act[-1]:
                    act = [act[0], act[1], act[2], "?"]
                if act not in new_action:
                    new_action.append(act)
            else:
                print("illegal action:", action)
        return new_action

    def _parse_output(self, in_str):
        in_str = str(in_str)
        in_str = in_str.replace('<s>', '').replace(
            '<\\s>', '').replace('o"clock', "o'clock")
        action = {"action": [], "text": ""}
        try:
            action = json.loads(in_str)
            action['action'] = eval(str(action['action']))
        except:
            print("invalid action:", in_str)
            print("-"*20)
        return action



    def predict(self, sys_act,mode='max',goal=None):
        # TODO emotion
        # raw_sys_act = sys_act
        # sys_act = sys_act[:5]
        # update goal
        # TODO
        allow_general_intent = False
        self.model.eval()

        if not self.add_sys_from_reward:
            self.sys_acts.append(sys_act)  # for terminate conversation

        # update constraint
        self.time_step += 2

        history = []
        if self.usr_acts:
            if self.max_history == 1:
                history = self.usr_acts[-1]
            else:
                history = self.usr_acts[-1*self.max_history:]
        inputs = json.dumps({"system": sys_act,
                             "goal": goal,
                             "history": history,
                             "turn": str(int(self.time_step/2))})
        with torch.no_grad():
            raw_output = self._generate_action(
                raw_inputs=inputs, mode=mode, allow_general_intent=allow_general_intent)
        output = self._parse_output(raw_output)
        self.semantic_action = self._remove_illegal_action(output["action"])

        if not self.only_action:
            self.utterance = output["text"]


        self.vector.update_mentioned_domain(self.semantic_action)
        self.usr_acts.append(self.semantic_action)


        del inputs

        if self.mode == "language":

            return self.utterance
        else:
            return self.semantic_action,self.utterance


    def init_session(self, goal=None):
        self.token_map = tokenMap(tokenizer=self.tokenizer)
        self.token_map.default(only_action=self.only_action)
        self.time_step = 0

        self.terminated = False
        self.add_sys_from_reward = False
        self.sys_acts = []
        self.usr_acts = []
        self.semantic_action = []
        self.utterance = ""


class UserPolicy(Policy):
    def __init__(self,
                 # model_checkpoint=r"D:\AlignUS\unify\experiments\multiwoz21_0_1\llama2",
                 # model_checkpoint=r"D:\AlignUS\unify\experiments\multiwoz21_0_1\24-03-24-20-18",
                 model_checkpoint=r"convlab\policy\genTUS\unify\experiments\multiwoz21_0_1\multiwoz21-exp",
                 # model_checkpoint=r"D:\AlignUS\convlab\policy\genTUS\unify\experiments\multiwoz21_0_1\wo_system",
                 # model_checkpoint=r"D:\AlignUS\convlab\policy\genTUS\unify\experiments\multiwoz21_0_1\wo_goal",

                 mode="semantic",
                 only_action=True,
                 sample=False,
                 action_penalty=False,
                 **kwargs):
        if not os.path.exists(os.path.dirname(model_checkpoint)):
            os.makedirs(os.path.dirname(model_checkpoint))
            model_downloader(os.path.dirname(model_checkpoint),
                             "https://zenodo.org/record/7372442/files/multiwoz21-exp.zip")

        self.policy = UserActionPolicy(
            model_checkpoint,
            mode=mode,
            only_action=only_action,
            action_penalty=action_penalty,
            **kwargs)
        self.policy.load(os.path.join(
            model_checkpoint, "pytorch_model.bin"))
        self.sample = sample


    def predict(self, sys_act, goal=None):
        if self.sample:
            mode = "sample"
        else:
            mode = "max"

        response = self.policy.predict(sys_act, mode,goal)
        self.semantic_action = self.policy.semantic_action

        return response


    def init_session(self, goal=None):
        self.policy.init_session(goal)
        self.semantic_action = []


