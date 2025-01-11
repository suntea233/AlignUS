from convlab.dst.rule.multiwoz import RuleDST
from convlab.policy.ppo import PPO
from convlab.util.analysis_tool.analyzer import Analyzer
from convlab.dialog_agent import PipelineAgent
from convlab.util.custom_util import set_seed
from convlab.policy.AlignUS.AlignUS import AlignUS
from convlab.policy.genTUS.stepGenTUS import UserPolicy
from convlab.policy.rule.multiwoz import RulePolicy


import random
import numpy as np
import torch


def set_seed(r_seed):
    random.seed(r_seed)
    np.random.seed(r_seed)
    torch.manual_seed(r_seed)


def test_end2end():
    # go to README.md of each model for more information
    # BERT nlu
    # sys_nlu = BERTNLU()
    sys_nlu = None
    # simple rule DST
    sys_dst = RuleDST()
    # sys_dst = None
    # rule policy
    # sys_policy = None
    sys_policy = PPO(is_train=False, seed=0, dataset_name='multiwoz21', load_path="from_pretrained")     # sys_policy = MLEPolicy()
    # sys_policy.load(r"D:\AlignUS\convlab\policy\ppo\finished_experiments\ABUS")
    # sys_policy = RulePolicy()
    # template NLG
    sys_nlg = None
    # assemble
    sys_agent = PipelineAgent(
        sys_nlu, sys_dst, sys_policy, sys_nlg, name='sys')




    model_checkpoint = r'convlab\policy\genTUS\unify\experiments\multiwoz21_0_1\multiwoz21-exp'
    # usr_policy = UserPolicy(model_checkpoint, mode="semantic",dataset='multiwoz21')
    # user_agent = PipelineAgent(None, None, usr_policy, None, 'user')


    user_agent = AlignUS(large_name='chatgpt')

    # user_nlu = None
    # user_dst = None
    # user_policy = RulePolicy(character='usr')
    # user_nlg = None
    # user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name='user')

    analyzer = Analyzer(user_agent=user_agent, dataset='multiwoz')

    set_seed(20200202)
    # set_seed(3407)
    analyzer.comprehensive_analyze(
        sys_agent=sys_agent, model_name='AlignUS', total_dialog=100)


if __name__ == '__main__':
    test_end2end()
