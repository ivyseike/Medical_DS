import numpy as np
import torch
import torch.optim as optim

from agents.BaseAgent import BaseAgent
from qlearning.dqn_prior import KR_DQN, DQN, Master_KR_DQN, Worker_KR_DQN
from agents.disease_classifier import dl_classifier
from qlearning.network_bodies import AtariBody, SimpleBody
from utils.ReplayMemory import ExperienceReplayMemory, MutExperienceReplayMemory, PrioritizedReplayMemory, MutPrioritizedReplayMemory
import dialog_config
import copy
from utils.utils import *


class HRLAgentDQN(BaseAgent):
    def __init__(self, sym_dict, dise_dict, req_dise_sym_dict, dise_sym_num_dict,
                 tran_mat, dise_sym_pro, sym_dise_pro, sym_prio, act_set, slot_set, params, static_policy=False):
        super(HRLAgentDQN, self).__init__()

        self.device = dialog_config.device
        # parameters for DQN
        self.noisy = params.get('noisy', False)
        self.priority_replay = params.get('priority_replay', False)
        self.gamma = params.get('gamma', 0.9)
        self.lr = params.get('lr', 0.01)
        self.batch_size = params.get('batch_size', 30)
        self.target_net_update_freq = params.get('target_net_update_freq', 1)
        self.experience_replay_size = params.get('experience_replay_size', 10000)
        # self.learn_start = params.get('learn_start', 1)
        self.sigma_init = params.get('sigma_init', 0.5)
        self.priority_beta_start = params.get('priority_beta_start', 0.4)
        self.priority_beta_frames = params.get('priority_beta_frames', 10000)
        self.priority_alpha = params.get('priority_alpha', 0.6)
        self.dqn_hidden_size = params.get('dqn_hidden_size', 128)
        self.fix_buffer = params['fix_buffer']
        #print(self.fix_buffer)
        self.static_policy = static_policy
        self.origin_model = params.get('origin_model',1)
 
        self.predict_mode = False
        self.warm_start = params['warm_start']
        self.max_turn = params['max_turn'] + 4
        self.epsilon = params.get('epsilon', 0.1)

        self.sym_dict = sym_dict  # all symptoms
        self.dise_dict = dise_dict # all disease
        self.dise_num = len(self.dise_dict.keys())
        self.sym_num = len(self.sym_dict.keys())
        self.dise_group_num = len(dialog_config.all_label_request_slots)
        self.dise_start = 2
        self.sym_start = 2 + self.dise_num
        self.master_take_action = True
        self.sub_round = 0
        self.original_state = None
        self.accum_reward = 0
        self.repeat = 0

        self.req_dise_sym_dict = req_dise_sym_dict  # high freq dise sym relations
        self.dise_sym_num_dict = dise_sym_num_dict  # dise sym discrete
        self.tran_mat_flag = torch.from_numpy(np.where(tran_mat>0,1.0,0.0)).float().to(self.device)
        self.tran_mat = torch.from_numpy(tran_mat).float().to(self.device)
        self.sym_dise_pro = torch.from_numpy(sym_dise_pro).float().to(self.device)
        self.dise_sym_pro = torch.from_numpy(dise_sym_pro).float().to(self.device)
        self.sym_prio = torch.from_numpy(sym_prio).float().to(self.device)

        self.act_set = act_set  # all acts
        self.slot_set = slot_set  # slot for agent
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys()) # add five disease slot

        self.feasible_actions = dialog_config.feasible_actions
        #self.num_actions = len(self.feasible_actions) - self.symptom_num #66 + 4 + 2 = 72
        self.num_actions = self.dise_group_num + 1 #inform + call worker
        self.kg_enabled = params['kg_enabled']
        self.hrl_enabled = params['hrl_enabled']

        self.state_dimension = self.sym_num
        # self.state_dimension = 2 * self.act_cardinality + 1 * self.slot_cardinality + self.max_turn
        # self.dise_start = 2
        # self.sym_start = self.dise_start + self.dise_num

        # self.id2disease = {}
        self.id2lowerAgent = {}
        for i in range(self.dise_group_num):
            # self.id2disease[i] = dialog_config.sys_inform_slots_values[i]
            temp_parameter = copy.deepcopy(params)
            self.id2lowerAgent[i] = LowerAgent(i, self.act_set, self.slot_set, self.sym_dict, self.dise_num, self.sym_num, temp_parameter)
        
        self.disease_classifier = dl_classifier(self.state_dimension, self.dqn_hidden_size, self.dise_num, self.dise_dict, copy.deepcopy(params))

        self.declare_networks(params['trained_model_path'])
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.001)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=0.001)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        '''
        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()
        '''
        self.model.train()
        self.target_model.eval()
        self.update_count = 0

        self.declare_memory()        
        self.nsteps = params.get('nsteps', 1)
        self.nstep_buffer = []

        self.request_set = copy.deepcopy(dialog_config.sys_request_slots_highfreq)
        self.current_slots = {}

    def initialize_episode(self):
        self.current_slots = {}

    def declare_networks(self, path):
        #master
        self.model = Master_KR_DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions, self.tran_mat, self.sym_dise_pro, self.dise_sym_pro, self.sym_prio, self.dise_num, self.sym_num, self.kg_enabled)
        self.target_model = Master_KR_DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions, self.tran_mat, self.sym_dise_pro, self.dise_sym_pro, self.sym_prio, self.dise_num, self.sym_num, self.kg_enabled)

        if path is not None:
            if self.origin_model==1:
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.predict_mode = True
                self.warm_start = 2
            else:
                self.load_specific_state_dict(path)
                self.predict_mode = True
                self.warm_start = 2
            

    def load_specific_state_dict(self,path):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def declare_memory(self):
        if self.fix_buffer:
            self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)
        else:
            self.memory = MutExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else MutPrioritizedReplayMemory(self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def prepare_state_representation(self, state):
        # user_action = state['user_action']
        current_slots = state['current_slots']
        #print('current_slots', current_slots)
        # agent_last = state['agent_action']
        # user action
        # user_act_rep = torch.zeros(1, self.act_cardinality, device=self.device)
        # user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0
        # user inform slots
        # user_inform_slots_rep = torch.zeros(1, self.slot_cardinality)
        # for slot in user_action['inform_slots'].keys():
        #     if slot not in self.slot_set:
        #         continue
        #     user_inform_slots_rep[0, self.slot_set[slot]] = user_action['inform_slots'][slot]
        # current slots
        current_slots_rep = torch.zeros(1, self.state_dimension, device=self.device)
        for slot, val in current_slots['inform_slots'].items():
            # if current_slots['inform_slots'][slot] == -2:
            #     current_slots_rep[0, self.slot_set[slot]] = 0.5
            # else:
            #     current_slots_rep[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]
           
            # if slot is disease, slot position =1, if slot is sym, slot position = slot val
            if slot == 'disease': 
                continue
                # current_slots_rep[0, self.slot_set[slot]] = 1
            else:
                if slot not in self.sym_dict.keys():
                    continue
                current_slots_rep[0, self.sym_dict[slot]] = val

        # agent action
        # agent_act_rep = torch.zeros(1, self.act_cardinality, device=self.device)
        # if agent_last:
        #     agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0
        # agent request slots
        # agent_request_slots_rep = torch.zeros(1, self.slot_cardinality).cuda()
        # if agent_last:
        #     for slot in agent_last['request_slots'].keys():
        #         if slot not in self.slot_set:
        #             continue
        #         agent_request_slots_rep[0, self.slot_set[slot]] = 1.0
        # turn info
        # turn_onehot_rep = torch.zeros(1, self.max_turn, device=self.device)
        # turn_onehot_rep[0, state['turn']] = 1.0
        # final_representation = torch.cat((user_act_rep, agent_act_rep, current_slots_rep, turn_onehot_rep), 1)
        final_representation = current_slots_rep

        return final_representation

    def register_experience_replay_tuple(self, s_t, master_action, worker_action, reward, s_tplus1, episode_over, disease_tag):
        self.accum_reward += reward
        training_example = None
        if self.sub_round == 1 and self.master_take_action == False:
            self.original_state = s_t
        # print(self.sub_round)
        # print(self.master_take_action)
        # print(self.origin_state == None)
        if self.sub_round >= 5 or reward >= 0 or episode_over == True or self.repeat == 1:
            # print(self.sub_round)
            # print(self.accum_reward)
            # print(self.master_action)
            
            assert(self.original_state != None)
            state_t_rep = self.prepare_state_representation(self.original_state)
            reward_t = self.accum_reward
            state_tplus1_rep = self.prepare_state_representation(s_tplus1)
            training_example = (state_t_rep, master_action, reward_t, state_tplus1_rep, episode_over)
            self.sub_round = 0
            self.master_take_action = True
            self.original_state = None
            self.accum_reward = 0
            self.repeat = 0
        

        # only record experience of dqn train, and warm start
        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                # self.experience_replay_pool.append(training_example)
                if training_example is not None:
                    self.memory.push(training_example)
        else:  # Prediction Mode
            if training_example is not None:
                self.memory.push(training_example)
            if master_action < self.dise_group_num:
                worker_action_str = None
                for key, val in self.sym_dict.items():
                    if val == worker_action:
                        worker_action_str = key
                        break
                assert (worker_action_str != None)
                self.id2lowerAgent[master_action].register_experience_replay_tuple(s_t, worker_action_str, reward, s_tplus1, episode_over)
            else:
                self.disease_classifier.register_experience_tuple(state_t_rep, disease_tag)
    
        
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over = zip(*transitions)
        neg_batch_episode_over = []
        for i in range(self.batch_size):
            if batch_episode_over[i]:
                neg_batch_episode_over.append(0)
            else:
                neg_batch_episode_over.append(1)
        
        batch_state = torch.cat(batch_state, dim=0).view(-1, self.state_dimension)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.cat(batch_next_state, dim=0).view(-1, self.state_dimension)
        batch_episode_over = torch.tensor(neg_batch_episode_over, device=self.device, dtype=torch.float).squeeze().view(-1, 1)  # flag for existence of next state

        return batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights


    def get_sym_flag(self, batch_state):
        ones = torch.ones((1, self.sym_num)).to(self.device)
        
        for key in dialog_config.sys_request_slots:
            id = self.sym_dict[key]
            if batch_state[0][id] != 0:
                ones[0][self.sym_dict[key]] = 0
        
        return ones

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights = batch_vars

        # estimate
        # self.model.sample_noise()
        # batch_state_flag = self.get_sym_flag(batch_state[:,(2*self.act_cardinality+self.dise_num+1):(2*self.act_cardinality+self.slot_cardinality)])
        current_q_values = self.model(batch_state).gather(1, batch_action)   # get the output values of the groundtruth batch_action
        # target
        with torch.no_grad():
            # self.target_model.sample_noise()
            # batch_next_state_flag = self.get_sym_flag(batch_next_state[:,(2*self.act_cardinality+self.dise_num+1):(2*self.act_cardinality+self.slot_cardinality)])
            max_next_q_values = self.target_model(batch_next_state).max(dim=1)[0].view(-1, 1)  # max q value
            expected_q_values = batch_reward + batch_episode_over*(self.gamma ** self.nsteps)*max_next_q_values
        diff = (expected_q_values - current_q_values)
        # print(diff)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = 0.5*torch.pow(diff,2).squeeze()*weights
            #loss = self.huber(diff).squeeze() * weights
        else:
            #loss = self.huber(diff)
            loss = 0.5*torch.pow(diff,2) 
        loss = loss.mean()

        return loss


    def single_batch(self):
        batch_vars = self.prep_minibatch()
        loss = self.compute_loss(batch_vars)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.model.parameters():
        #    if param.requires_grad == True:    
        #        param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # self.update_target_model()
        return loss.item()

    def train(self):
        cur_bellman_err = 0.0
        #print(self.model.tran_mat)
        #print(self.model.tran_mat.requires_grad)# print(self.target_model.fc1.weight)
        if self.fix_buffer:
            for iter in range(int(len(self.memory)/self.batch_size)):
                cur_bellman_err += self.single_batch()
            # print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err)* self.batch_size / len(self.memory), len(self.memory)))
        else:
            for iter in range(int(len(self.memory)/self.batch_size)):
                cur_bellman_err += self.single_batch()
            # print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) * self.batch_size / len(self.memory), len(self.memory)))
            for i in range(len(self.id2lowerAgent)):
                self.id2lowerAgent[i].train()
            self.disease_classifier.train()
        #print(self.model.tran_mat)
        self.update_target_model()

    def run_policy(self, rep, state):
        act_ind = -1
        if np.random.random() < self.epsilon:
            act_ind = np.random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.memory.buffer) > self.experience_replay_size:
                    self.warm_start = 2
                act_ind = self.rule_policy(state)
            else:
                # sym_flag = self.get_sym_flag(rep[:, (2*self.act_cardinality+self.dise_num+1):(2*self.act_cardinality+self.slot_cardinality)])
                act_ind = self.model.predict(rep)
                # act_ind = self.model.predict(rep)
        #repeat = self.detect_repeat(state, act_ind)
        return act_ind

    def detect_repeat(self, state, act_ind):
        current_inform_slot = state['current_slots']['inform_slots']
        #print(current_inform_slot)
        repeat = 0
        action = self.feasible_actions[act_ind]
        #print(action)
        if action['diaact'] == 'request' and list(action['request_slots'].keys())[0] in current_inform_slot.keys():
            repeat = 1
        #print(repeat)
        return repeat
 
    def disease_from_dict(self, current_slots, sym_flag):

        if sym_flag == 0:
            dise = dialog_config.NO_MATCH
            for d in self.req_dise_sym_dict:
                dise = d
                for sym in self.req_dise_sym_dict[d]:
                    if sym not in current_slots['inform_slots'] or current_slots['inform_slots'][sym] != True:
                        dise = dialog_config.NO_MATCH
                if dise != dialog_config.NO_MATCH:
                    return dise
            return dise
        else:
            dise = dialog_config.NO_MATCH_BY_RATE
            max_sym_rate = 0.0
            for d in self.dise_sym_num_dict:
                tmp = [v for v in self.dise_sym_num_dict[d].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                cur_dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[d][sym]
                for sym in self.dise_sym_num_dict[d]:
                    cur_dise_sym_sum += self.dise_sym_num_dict[d][sym]
                # tmp_rate = float(len(tmp))/float(len(self.req_dise_sym_dict[dise]))
                tmp_rate = float(tmp_sum) / float(cur_dise_sym_sum)
                if tmp_rate > max_sym_rate:
                    max_sym_rate = tmp_rate
                    dise = d
            return dise

    def rule_policy(self, state):
        """ Rule Policy """
        current_slots = state['current_slots']
        act_slot_response = {}
        sym_flag = 1  # 1 for no left sym, 0 for still have
        for sym in self.request_set:
            if sym not in current_slots['inform_slots'].keys():
                sym_flag = 0
        dise = self.disease_from_dict(current_slots, sym_flag)
        if dise == dialog_config.NO_MATCH:  # no match but still have syms to ask
            cur_dise_sym_rate = {}
            for dise in self.dise_sym_num_dict:
                if dise not in cur_dise_sym_rate:
                    cur_dise_sym_rate[dise] = 0
                tmp = [v for v in self.dise_sym_num_dict[dise].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[dise][sym]
                for sym in self.dise_sym_num_dict[dise]:
                    dise_sym_sum += self.dise_sym_num_dict[dise][sym]
                # dise_sym_rate[dise] = float(len(tmp))/float(len(self.dise_sym_num_dict[dise]))
                    cur_dise_sym_rate[dise] = float(tmp_sum) / float(dise_sym_sum)

            sorted_dise = list(dict(sorted(cur_dise_sym_rate.items(), key=lambda d: d[1], reverse=True)).keys())
            left_set = []
            for i in range(len(sorted_dise)):
                max_dise = sorted_dise[i]
                left_set = [v for v in self.req_dise_sym_dict[max_dise] if v not in current_slots['inform_slots'].keys()]
                if len(left_set) > 0: break
            # if syms in request set of all disease have been asked, choose one sym in request set
            if len(left_set) == 0:
                print('this will not happen')
                left_set = [v for v in self.request_set if v not in current_slots['inform_slots'].keys()]
            slot = np.random.choice(left_set)
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}

        elif dise == dialog_config.NO_MATCH_BY_RATE: # no match and no sym to ask
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': 'UNK', 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {},

        else:  # match one dise by complete match or by rate
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': dise, 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}

        return self.action_index(act_slot_response)

    def state_to_action(self, state):
        self.representation = self.prepare_state_representation(state)
        if state['user_action']['turn'] == self.max_turn - 6: #26-4 = 22
            self.master_take_action = False
            self.master_action = self.dise_group_num
        elif self.master_take_action == True:
            self.master_action = self.run_policy(self.representation, state)
            self.master_take_action = False

        self.action = None
        if self.master_action < self.dise_group_num:
            self.sub_round += 1
            next_agent = self.id2lowerAgent[self.master_action]
            next_agent_rep = next_agent.prepare_state_representation(state)
            self.action_str = next_agent.run_policy(next_agent_rep, state)
            self.action = self.sym_dict[self.action_str]
            self.repeat = self.detect_repeat(state, self.action)
            act_slot_response = copy.deepcopy(self.feasible_actions[self.dise_start+self.dise_num+self.action])
        else:
            self.sub_round += 1
            next_agent = self.disease_classifier
            self.action = next_agent.predict(self.representation)
            act_slot_response = copy.deepcopy(self.feasible_actions[self.dise_start+self.action])
            repeat = None
        
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}, self.master_action, self.action

    def action_index(self, act_slot_response):
        """ Return the index of action """
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception("action index not found")
        return None

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("update target model!!!")
            self.target_model.load_state_dict(self.model.state_dict())

    # get the action index with the max action values
    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass

    def set_predict_mode(self, mode_):
        self.predict_mode = mode_
        for i in range(len(self.id2lowerAgent)):
            self.id2lowerAgent[i].predict_mode = mode_

class LowerAgent(BaseAgent):
    def __init__(self, id, act_set, slot_set, all_sym_dict, dise_num, symp_num, params):
        super(LowerAgent, self).__init__()

        self.device = dialog_config.device
        self.id = id

        self.dise_num = dise_num
        self.all_symp_num = symp_num
        self.all_sym_dict = all_sym_dict

        self.act_set = act_set
        self.slot_set = slot_set
        # self.act_cardinality = len(act_set.keys())
        # self.slot_cardinality = len(slot_set.keys())

        # parameters for DQN
        self.warm_start = params['warm_start']
        self.noisy = params.get('noisy', False)
        self.priority_replay = params.get('priority_replay', False)
        self.gamma = params.get('gamma', 0.9)
        self.lr = params.get('lr', 0.01)
        self.batch_size = params.get('batch_size', 30)
        self.target_net_update_freq = params.get('target_net_update_freq', 1)
        self.experience_replay_size = params.get('experience_replay_size', 10000)

        self.sigma_init = params.get('sigma_init', 0.5)
        self.priority_beta_start = params.get('priority_beta_start', 0.4)
        self.priority_beta_frames = params.get('priority_beta_frames', 10000)
        self.priority_alpha = params.get('priority_alpha', 0.6)
        self.dqn_hidden_size = params.get('dqn_hidden_size', 128)
        self.fix_buffer = params['fix_buffer']
        
        self.static_policy = False
        self.origin_model = params.get('origin_model',1)
 
        self.predict_mode = False
        self.max_turn = params['max_turn'] + 4
        self.epsilon = params.get('epsilon', 0.1)

        self.kg_enabled = params['kg_enabled']
        
        data_folder = params['data_folder'] + '/label' + str(self.id+1)
        self.sym_dict = text_to_dict('{}/symptoms.txt'.format(data_folder))  # all symptoms
        self.req_dise_sym_dict = load_pickle('{}/req_dise_sym_dict.p'.format(data_folder))
        # self.sym_num_dict = load_pickle('{}/sym_num_dict.p'.format(data_folder))

        # self.sym_dise_pro = None
        sym_dise_pro = np.loadtxt('{}/sym_dise_pro_label{}.txt'.format(data_folder, str(self.id+1)))
        self.sym_dise_pro = torch.from_numpy(sym_dise_pro).float().to(self.device)
        
        self.sym_num = len(self.sym_dict)
        self.num_actions = len(self.sym_dict)
        self.state_dimension = self.sym_num

        self.feasible_actions = [None] * self.sym_num
        for key, val in self.sym_dict.items():
            self.feasible_actions[val] = {'diaact':'request', 'inform_slots':{}, 'request_slots': {key: 'UNK'}}

        # self.state_dimension = 2 * self.act_cardinality + 1 * self.slot_cardinality + self.max_turn

        # self.sym_prio = torch.from_numpy(sym_prio).float().to(self.device)
        #self.tran_mat_flag = torch.from_numpy(np.where(tran_mat>0,1.0,0.0)).float().to(self.device)
        #self.tran_mat = torch.from_numpy(tran_mat).float().to(self.device)
        # self.sym_dise_pro = torch.from_numpy(sym_dise_pro).float().to(self.device)
        # self.dise_sym_pro = torch.from_numpy(dise_sym_pro).float().to(self.device)
        self.tran_mat = None
        
        self.declare_networks(params['trained_model_path'])
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.001)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=0.001)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        '''
        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()
        '''
        self.model.train()
        self.target_model.eval()
        self.update_count = 0

        self.declare_memory()        
        self.nsteps = params.get('nsteps', 1)
        self.nstep_buffer = []

        self.request_set = copy.deepcopy(dialog_config.sys_request_slots_highfreq)
        self.current_slots = {}

    def initialize_episode(self):
        self.current_slots = {}

    def declare_networks(self, path):
        
        self.model = Worker_KR_DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions, self.tran_mat, self.sym_dise_pro, 
                    self.dise_num, self.all_symp_num, self.kg_enabled)
        self.target_model = Worker_KR_DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions, self.tran_mat, self.sym_dise_pro, 
                    self.dise_num ,self.all_symp_num, self.kg_enabled)
        if path is not None:
            if self.origin_model==1:
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.predict_mode = True
                self.warm_start = 2
            else:
                self.load_specific_state_dict(path)
                self.predict_mode = True
                self.warm_start = 2

    def load_specific_state_dict(self,path):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def declare_memory(self):
        if self.fix_buffer:
            self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)
        else:
            self.memory = MutExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else MutPrioritizedReplayMemory(self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def prepare_state_representation(self, state):
        # user_action = state['user_action']
        current_slots = state['current_slots']
        #print('current_slots', current_slots)
        # agent_last = state['agent_action']
        # user action
        # user_act_rep = torch.zeros(1, self.act_cardinality, device=self.device)
        # user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0
        # user inform slots
        # user_inform_slots_rep = torch.zeros(1, self.slot_cardinality)
        # for slot in user_action['inform_slots'].keys():
        #     if slot not in self.slot_set:
        #         continue
        #     user_inform_slots_rep[0, self.slot_set[slot]] = user_action['inform_slots'][slot]
        # current slots
        current_slots_rep = torch.zeros(1, self.state_dimension, device=self.device)
        for slot, val in current_slots['inform_slots'].items():
            # if current_slots['inform_slots'][slot] == -2:
            #     current_slots_rep[0, self.slot_set[slot]] = 0.5
            # else:
            #     current_slots_rep[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]
           
            # if slot is disease, slot position =1, if slot is sym, slot position = slot val
            if slot == 'disease': 
                continue
            else:
                if slot in self.sym_dict.keys():
                    current_slots_rep[0, self.sym_dict[slot]] = val

        # agent action
        # agent_act_rep = torch.zeros(1, self.act_cardinality, device=self.device)
        # if agent_last:
        #     agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0
        # agent request slots
        # agent_request_slots_rep = torch.zeros(1, self.slot_cardinality).cuda()
        # if agent_last:
        #     for slot in agent_last['request_slots'].keys():
        #         if slot not in self.slot_set:
        #             continue
        #         agent_request_slots_rep[0, self.slot_set[slot]] = 1.0
        # turn info
        # turn_onehot_rep = torch.zeros(1, self.max_turn, device=self.device)
        # turn_onehot_rep[0, state['turn']] = 1.0
        # final_representation = torch.cat((user_act_rep, agent_act_rep, current_slots_rep, turn_onehot_rep), 1)
        final_representation = current_slots_rep

        return final_representation

    def register_experience_replay_tuple(self, s_t, worker_action_str, reward, s_tplus1, episode_over):
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.sym_dict[worker_action_str]
                
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)

        # only record experience of dqn train, and warm start
        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                # self.experience_replay_pool.append(training_example)
                self.memory.push(training_example)
        else:  # Prediction Mode
            # self.experience_replay_pool.append(training_example)
            self.memory.push(training_example)
    
        
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over = zip(*transitions)
        neg_batch_episode_over = []
        for i in range(self.batch_size):
            if batch_episode_over[i]:
                neg_batch_episode_over.append(0)
            else:
                neg_batch_episode_over.append(1)
        
        batch_state = torch.cat(batch_state, dim=0).view(-1, self.state_dimension)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.cat(batch_next_state, dim=0).view(-1, self.state_dimension)
        batch_episode_over = torch.tensor(neg_batch_episode_over, device=self.device, dtype=torch.float).squeeze().view(-1, 1)  # flag for existence of next state

        return batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights

    def get_sym_flag(self, batch_state):
        ones = torch.ones((1, self.sym_num)).to(self.device)
        for i in range(self.sym_num):
            ones[0][i] = 1 if batch_state[0][i] == 0 else 0
        
        return ones
        
        # ones = torch.ones(batch_state.size()).to(self.device)
        # zeros = torch.zeros(batch_state.size()).to(self.device)
        # return torch.cat((torch.ones(bs, self.sym_start).to(self.device),torch.where(batch_state == 0, ones, zeros)),1)

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights = batch_vars

        # estimate
        # self.model.sample_noise()
        batch_state_flag = self.get_sym_flag(batch_state)
        current_q_values = self.model(batch_state, batch_state_flag).gather(1, batch_action)   # get the output values of the groundtruth batch_action
        # target
        with torch.no_grad():
            # self.target_model.sample_noise()
            batch_next_state_flag = self.get_sym_flag(batch_next_state)
            max_next_q_values = self.target_model(batch_next_state,batch_next_state_flag).max(dim=1)[0].view(-1, 1)  # max q value
            expected_q_values = batch_reward + batch_episode_over*(self.gamma ** self.nsteps)*max_next_q_values
        diff = (expected_q_values - current_q_values)
        # print(diff)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = 0.5*torch.pow(diff,2).squeeze()*weights
            #loss = self.huber(diff).squeeze() * weights
        else:
            #loss = self.huber(diff)
            loss = 0.5*torch.pow(diff,2) 
        loss = loss.mean()

        return loss


    def single_batch(self):
        batch_vars = self.prep_minibatch()
        loss = self.compute_loss(batch_vars)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.model.parameters():
        #    if param.requires_grad == True:    
        #        param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # self.update_target_model()
        return loss.item()

    def train(self):
        cur_bellman_err = 0.0
        #print(self.model.tran_mat)
        #print(self.model.tran_mat.requires_grad)# print(self.target_model.fc1.weight)
        if self.fix_buffer:
            for iter in range(int(len(self.memory)/self.batch_size)):
                cur_bellman_err += self.single_batch()
            # print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err)* self.batch_size / len(self.memory), len(self.memory)))
        else:
            for iter in range(int(len(self.memory)/self.batch_size)):
                cur_bellman_err += self.single_batch()
            if len(self.memory):
                pass
                # print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) * self.batch_size / len(self.memory), len(self.memory)))
            else:
                print("worker" + str(self.id) + " no memory")
        #print(self.model.tran_mat)
        self.update_target_model()

    def run_policy(self, rep, state):
        act_ind = -1
        if np.random.random() < self.epsilon:
            act_ind = np.random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.memory.buffer) > self.experience_replay_size:
                    self.warm_start = 2
                act_ind = self.rule_policy(state)
            else:
                sym_flag = self.get_sym_flag(rep)
                # sym_flag = self.get_sym_flag(rep[:, (2*self.act_cardinality+self.dise_num+1):(2*self.act_cardinality+self.slot_cardinality)])
                act_ind = self.model.predict(rep, sym_flag)
        #repeat = self.detect_repeat(state, act_ind)
        for key, val in self.sym_dict.items():
            if val == act_ind:
                return key
        return None

    def detect_repeat(self, state, act_ind):
        current_inform_slot = state['current_slots']['inform_slots']
        #print(current_inform_slot)
        repeat = 0
        action = self.feasible_actions[act_ind]
        #print(action)
        if action['diaact'] == 'request' and list(action['request_slots'].keys())[0] in current_inform_slot.keys():
            repeat = 1
        #print(repeat)
        return repeat
 
    def disease_from_dict(self, current_slots, sym_flag):

        if sym_flag == 0: #所有用户已告知的症状都与某一个疾病关联的话，返回这个疾病
            dise = dialog_config.NO_MATCH
            for d in self.req_dise_sym_dict:
                dise = d
                for sym in self.req_dise_sym_dict[d]:
                    if sym not in current_slots['inform_slots'] or current_slots['inform_slots'][sym] != True:
                        dise = dialog_config.NO_MATCH
                if dise != dialog_config.NO_MATCH:
                    return dise
            return dise
        else:
            dise = dialog_config.NO_MATCH_BY_RATE
            max_sym_rate = 0.0
            for d in self.dise_sym_num_dict:
                tmp = [v for v in self.dise_sym_num_dict[d].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                cur_dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[d][sym]
                for sym in self.dise_sym_num_dict[d]:
                    cur_dise_sym_sum += self.dise_sym_num_dict[d][sym]
                # tmp_rate = float(len(tmp))/float(len(self.req_dise_sym_dict[dise]))
                tmp_rate = float(tmp_sum) / float(cur_dise_sym_sum)
                if tmp_rate > max_sym_rate:
                    max_sym_rate = tmp_rate
                    dise = d
            return dise

    def rule_policy(self, state):
        """ Rule Policy """
        current_slots = state['current_slots']
        act_slot_response = {}
        sym_flag = 1  # 1 for no left sym, 0 for still have
        for sym in self.request_set:
            if sym not in current_slots['inform_slots'].keys():
                sym_flag = 0
        dise = self.disease_from_dict(current_slots, sym_flag)
        if dise == dialog_config.NO_MATCH:  # no match but still have syms to ask
            cur_dise_sym_rate = {}
            for dise in self.dise_sym_num_dict:
                if dise not in cur_dise_sym_rate:
                    cur_dise_sym_rate[dise] = 0
                tmp = [v for v in self.dise_sym_num_dict[dise].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[dise][sym]
                for sym in self.dise_sym_num_dict[dise]:
                    dise_sym_sum += self.dise_sym_num_dict[dise][sym]
                # dise_sym_rate[dise] = float(len(tmp))/float(len(self.dise_sym_num_dict[dise]))
                    cur_dise_sym_rate[dise] = float(tmp_sum) / float(dise_sym_sum)

            sorted_dise = list(dict(sorted(cur_dise_sym_rate.items(), key=lambda d: d[1], reverse=True)).keys())
            left_set = []
            for i in range(len(sorted_dise)):
                max_dise = sorted_dise[i]
                left_set = [v for v in self.req_dise_sym_dict[max_dise] if v not in current_slots['inform_slots'].keys()]
                if len(left_set) > 0: break
            # if syms in request set of all disease have been asked, choose one sym in request set
            if len(left_set) == 0:
                print('this will not happen')
                left_set = [v for v in self.request_set if v not in current_slots['inform_slots'].keys()]
            slot = np.random.choice(left_set)
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}

        elif dise == dialog_config.NO_MATCH_BY_RATE: # no match and no sym to ask
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': 'UNK', 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {},

        else:  # match one dise by complete match or by rate
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': dise, 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}

        return self.action_index(act_slot_response)

    def state_to_action(self, state):
        self.representation = self.prepare_state_representation(state)
        self.action, repeat = self.run_policy(self.representation, state)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}, repeat

    def action_index(self, act_slot_response):
        """ Return the index of action """
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception("action index not found")
        return None

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("update target model!!!")
            self.target_model.load_state_dict(self.model.state_dict())

    # get the action index with the max action values
    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass



class AgentDQN(BaseAgent):
    def __init__(self, sym_dict=None, dise_dict=None, req_dise_sym_dict=None, dise_sym_num_dict=None,
                 tran_mat=None, dise_sym_pro=None, sym_dise_pro=None, sym_prio = None, act_set=None, slot_set=None, params=None, static_policy=False):
        super(AgentDQN, self).__init__()

        self.device = dialog_config.device
        # parameters for DQN
        self.noisy = params.get('noisy', False)
        self.priority_replay = params.get('priority_replay', False)
        self.gamma = params.get('gamma', 0.9)
        self.lr = params.get('lr', 0.01)
        self.batch_size = params.get('batch_size', 30)
        self.target_net_update_freq = params.get('target_net_update_freq', 1)
        self.experience_replay_size = params.get('experience_replay_size', 10000)
        # self.learn_start = params.get('learn_start', 1)
        self.sigma_init = params.get('sigma_init', 0.5)
        self.priority_beta_start = params.get('priority_beta_start', 0.4)
        self.priority_beta_frames = params.get('priority_beta_frames', 10000)
        self.priority_alpha = params.get('priority_alpha', 0.6)
        self.dqn_hidden_size = params.get('dqn_hidden_size', 128)
        self.fix_buffer = params['fix_buffer']
        #print(self.fix_buffer)
        self.static_policy = static_policy
        self.origin_model = params.get('origin_model',1)
 
        self.predict_mode = False
        self.warm_start = params['warm_start']
        self.max_turn = params['max_turn'] + 4
        self.epsilon = params.get('epsilon', 0.1)
        self.sym_dict = sym_dict  # all symptoms
        self.dise_dict = dise_dict
        self.req_dise_sym_dict = req_dise_sym_dict  # high freq dise sym relations
        self.dise_sym_num_dict = dise_sym_num_dict  # dise sym discrete
        # self.tran_mat_flag = torch.from_numpy(np.where(tran_mat>0,1.0,0.0)).float().to(self.device)
        # self.tran_mat = torch.from_numpy(tran_mat).float().to(self.device)
        # self.dise_num = len(dise_dict.keys())
        # self.symp_num = len(sym_dict.keys())
        # self.sym_dise_pro = torch.from_numpy(sym_dise_pro).float().to(self.device)
        # self.dise_sym_pro = torch.from_numpy(dise_sym_pro).float().to(self.device)
        # self.sym_prio = torch.from_numpy(sym_prio).float().to(self.device)
        # self.act_set = act_set  # all acts
        # self.slot_set = slot_set  # slot for agent
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys()) # add five disease slot
        self.feasible_actions = dialog_config.feasible_actions  # actions
        self.num_actions = len(self.feasible_actions)
        self.state_dimension = 2 * self.act_cardinality + 1 * self.slot_cardinality + self.max_turn
        self.dise_start = 2 
        
        self.sym_start = self.dise_start + self.dise_num

        self.kg_enabled = params['kg_enabled']

        self.declare_networks(params['trained_model_path'])
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.001)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=0.001)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        '''
        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()
        '''
        self.model.train()
        self.target_model.eval()
        self.update_count = 0

        self.declare_memory()        
        self.nsteps = params.get('nsteps', 1)
        self.nstep_buffer = []

        self.request_set = copy.deepcopy(dialog_config.sys_request_slots_highfreq)
        self.current_slots = {}

    def initialize_episode(self):
        self.current_slots = {}

    def declare_networks(self, path):
        
        self.model = KR_DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions, self.tran_mat, self.dise_start, self.act_cardinality, self.slot_cardinality, self.sym_dise_pro, self.dise_sym_pro, self.sym_prio, self.dise_num, self.symp_num, self.kg_enabled)
        self.target_model = KR_DQN(self.state_dimension, self.dqn_hidden_size, self.num_actions, self.tran_mat, self.dise_start, self.act_cardinality, self.slot_cardinality, self.sym_dise_pro, self.dise_sym_pro, self.sym_prio, self.dise_num, self.symp_num, self.kg_enabled)
        if path is not None:
            if self.origin_model==1:
                checkpoint = torch.load(path)
                self.model.load_state_dict(checkpoint['state_dict'])
                self.predict_mode = True
                self.warm_start = 2
            else:
                self.load_specific_state_dict(path)
                self.predict_mode = True
                self.warm_start = 2

    def load_specific_state_dict(self,path):
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(path)['state_dict']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def declare_memory(self):
        if self.fix_buffer:
            self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)
        else:
            self.memory = MutExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else MutPrioritizedReplayMemory(self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def prepare_state_representation(self, state):
        user_action = state['user_action']
        current_slots = state['current_slots']
        #print('current_slots', current_slots)
        agent_last = state['agent_action']
        # user action
        user_act_rep = torch.zeros(1, self.act_cardinality, device=self.device)
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0
        # user inform slots
        # user_inform_slots_rep = torch.zeros(1, self.slot_cardinality)
        # for slot in user_action['inform_slots'].keys():
        #     if slot not in self.slot_set:
        #         continue
        #     user_inform_slots_rep[0, self.slot_set[slot]] = user_action['inform_slots'][slot]
        # current slots
        current_slots_rep = torch.zeros(1, self.slot_cardinality, device=self.device)
        for slot in current_slots['inform_slots']:
            if slot not in self.slot_set:
                continue
            # if current_slots['inform_slots'][slot] == -2:
            #     current_slots_rep[0, self.slot_set[slot]] = 0.5
            # else:
            #     current_slots_rep[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]
           
            # if slot is disease, slot position =1, if slot is sym, slot position = slot val
            if slot == 'disease': 
                current_slots_rep[0, self.slot_set[slot]] = 1
            else:
                current_slots_rep[0, self.slot_set[slot]] = current_slots['inform_slots'][slot]

        # agent action
        agent_act_rep = torch.zeros(1, self.act_cardinality, device=self.device)
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0
        # agent request slots
        # agent_request_slots_rep = torch.zeros(1, self.slot_cardinality).cuda()
        # if agent_last:
        #     for slot in agent_last['request_slots'].keys():
        #         if slot not in self.slot_set:
        #             continue
        #         agent_request_slots_rep[0, self.slot_set[slot]] = 1.0
        # turn info
        turn_onehot_rep = torch.zeros(1, self.max_turn, device=self.device)
        turn_onehot_rep[0, state['turn']] = 1.0
        final_representation = torch.cat((user_act_rep, agent_act_rep, current_slots_rep, turn_onehot_rep), 1)

        return final_representation

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over):
        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over)

        # only record experience of dqn train, and warm start
        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                # self.experience_replay_pool.append(training_example)
                self.memory.push(training_example)
        else:  # Prediction Mode
            # self.experience_replay_pool.append(training_example)
            self.memory.push(training_example)
    
        
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over = zip(*transitions)
        neg_batch_episode_over = []
        for i in range(self.batch_size):
            if batch_episode_over[i]:
                neg_batch_episode_over.append(0)
            else:
                neg_batch_episode_over.append(1)
        
        batch_state = torch.cat(batch_state, dim=0).view(-1, self.state_dimension)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.cat(batch_next_state, dim=0).view(-1, self.state_dimension)
        batch_episode_over = torch.tensor(neg_batch_episode_over, device=self.device, dtype=torch.float).squeeze().view(-1, 1)  # flag for existence of next state

        return batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights

    def get_sym_flag(self, batch_state):
        bs = batch_state.size(0)
        
        ones = torch.ones(batch_state.size()).to(self.device)
        zeros = torch.zeros(batch_state.size()).to(self.device)
        return torch.cat((torch.ones(bs, self.sym_start).to(self.device),torch.where(batch_state == 0, ones, zeros)),1)

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, batch_episode_over, indices, weights = batch_vars

        # estimate
        # self.model.sample_noise()
        batch_state_flag = self.get_sym_flag(batch_state[:,(2*self.act_cardinality+self.dise_num+1):(2*self.act_cardinality+self.slot_cardinality)])
        current_q_values = self.model(batch_state, batch_state_flag).gather(1, batch_action)   # get the output values of the groundtruth batch_action
        # target
        with torch.no_grad():
            # self.target_model.sample_noise()
            batch_next_state_flag = self.get_sym_flag(batch_next_state[:,(2*self.act_cardinality+self.dise_num+1):(2*self.act_cardinality+self.slot_cardinality)])
            max_next_q_values = self.target_model(batch_next_state,batch_next_state_flag).max(dim=1)[0].view(-1, 1)  # max q value
            expected_q_values = batch_reward + batch_episode_over*(self.gamma ** self.nsteps)*max_next_q_values
        diff = (expected_q_values - current_q_values)
        # print(diff)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = 0.5*torch.pow(diff,2).squeeze()*weights
            #loss = self.huber(diff).squeeze() * weights
        else:
            #loss = self.huber(diff)
            loss = 0.5*torch.pow(diff,2) 
        loss = loss.mean()

        return loss


    def single_batch(self):
        batch_vars = self.prep_minibatch()
        loss = self.compute_loss(batch_vars)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.model.parameters():
        #    if param.requires_grad == True:    
        #        param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # self.update_target_model()
        return loss.item()

    def train(self):
        cur_bellman_err = 0.0
        #print(self.model.tran_mat)
        #print(self.model.tran_mat.requires_grad)# print(self.target_model.fc1.weight)
        if self.fix_buffer:
            for iter in range(int(len(self.memory)/self.batch_size)):
                cur_bellman_err += self.single_batch()
            print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err)* self.batch_size / len(self.memory), len(self.memory)))
        else:
            for iter in range(int(len(self.memory)/self.batch_size)):
                cur_bellman_err += self.single_batch()
            print("cur bellman err %.4f, experience replay pool %s" % (float(cur_bellman_err) * self.batch_size / len(self.memory), len(self.memory)))
        #print(self.model.tran_mat)
        self.update_target_model()

    def run_policy(self, rep, state):
        act_ind = -1
        if np.random.random() < self.epsilon:
            act_ind = np.random.randint(0, self.num_actions - 1)
        else:
            if self.warm_start == 1:
                if len(self.memory.buffer) > self.experience_replay_size:
                    self.warm_start = 2
                act_ind = self.rule_policy(state)
            else:
                sym_flag = self.get_sym_flag(rep[:, (2*self.act_cardinality+self.dise_num+1):(2*self.act_cardinality+self.slot_cardinality)])
                act_ind = self.model.predict(rep, sym_flag)
        repeat = self.detect_repeat(state, act_ind)
        return act_ind, repeat

    def detect_repeat(self, state, act_ind):
        current_inform_slot = state['current_slots']['inform_slots']
        #print(current_inform_slot)
        repeat = 0
        action = self.feasible_actions[act_ind]
        #print(action)
        if action['diaact'] == 'request' and list(action['request_slots'].keys())[0] in current_inform_slot.keys():
            repeat = 1
        #print(repeat)
        return repeat
 
    def disease_from_dict(self, current_slots, sym_flag):

        if sym_flag == 0: #所有用户已告知的症状都与某一个疾病关联的话，返回这个疾病
            dise = dialog_config.NO_MATCH
            for d in self.req_dise_sym_dict:
                dise = d
                for sym in self.req_dise_sym_dict[d]:
                    if sym not in current_slots['inform_slots'] or current_slots['inform_slots'][sym] != True:
                        dise = dialog_config.NO_MATCH
                if dise != dialog_config.NO_MATCH:
                    return dise
            return dise
        else:
            dise = dialog_config.NO_MATCH_BY_RATE
            max_sym_rate = 0.0
            for d in self.dise_sym_num_dict:
                tmp = [v for v in self.dise_sym_num_dict[d].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                cur_dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[d][sym]
                for sym in self.dise_sym_num_dict[d]:
                    cur_dise_sym_sum += self.dise_sym_num_dict[d][sym]
                # tmp_rate = float(len(tmp))/float(len(self.req_dise_sym_dict[dise]))
                tmp_rate = float(tmp_sum) / float(cur_dise_sym_sum)
                if tmp_rate > max_sym_rate:
                    max_sym_rate = tmp_rate
                    dise = d
            return dise

    def rule_policy(self, state):
        """ Rule Policy """
        current_slots = state['current_slots']
        act_slot_response = {}
        sym_flag = 1  # 1 for no left sym, 0 for still have
        for sym in self.request_set:
            if sym not in current_slots['inform_slots'].keys():
                sym_flag = 0
        dise = self.disease_from_dict(current_slots, sym_flag)
        if dise == dialog_config.NO_MATCH:  # no match but still have syms to ask
            cur_dise_sym_rate = {}
            for dise in self.dise_sym_num_dict:
                if dise not in cur_dise_sym_rate:
                    cur_dise_sym_rate[dise] = 0
                tmp = [v for v in self.dise_sym_num_dict[dise].keys() if v in current_slots['inform_slots'].keys()]
                tmp_sum = 0
                dise_sym_sum = 0
                for sym in tmp:
                    tmp_sum += self.dise_sym_num_dict[dise][sym]
                for sym in self.dise_sym_num_dict[dise]:
                    dise_sym_sum += self.dise_sym_num_dict[dise][sym]
                # dise_sym_rate[dise] = float(len(tmp))/float(len(self.dise_sym_num_dict[dise]))
                    cur_dise_sym_rate[dise] = float(tmp_sum) / float(dise_sym_sum)

            sorted_dise = list(dict(sorted(cur_dise_sym_rate.items(), key=lambda d: d[1], reverse=True)).keys())
            left_set = []
            for i in range(len(sorted_dise)):
                max_dise = sorted_dise[i]
                left_set = [v for v in self.req_dise_sym_dict[max_dise] if v not in current_slots['inform_slots'].keys()]
                if len(left_set) > 0: break
            # if syms in request set of all disease have been asked, choose one sym in request set
            if len(left_set) == 0:
                print('this will not happen')
                left_set = [v for v in self.request_set if v not in current_slots['inform_slots'].keys()]
            slot = np.random.choice(left_set)
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}

        elif dise == dialog_config.NO_MATCH_BY_RATE: # no match and no sym to ask
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': 'UNK', 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {},

        else:  # match one dise by complete match or by rate
            act_slot_response['diaact'] = "inform"
            act_slot_response['inform_slots'] = {'disease': dise, 'taskcomplete': "PLACEHOLDER"}
            act_slot_response['request_slots'] = {}

        return self.action_index(act_slot_response)

    def state_to_action(self, state):
        self.representation = self.prepare_state_representation(state) #111 for muzhi, 397 for simulated
        self.action, repeat = self.run_policy(self.representation, state)
        act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}, repeat

    def action_index(self, act_slot_response):
        """ Return the index of action """
        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        raise Exception("action index not found")
        return None

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            print("update target model!!!")
            self.target_model.load_state_dict(self.model.state_dict())

    # get the action index with the max action values
    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2] * (self.gamma ** i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass

    def set_predict_mode(self, mode_):
        self.predict_mode = mode_
        # for i in range(len(self.id2lowerAgent)):
        #     self.id2lowerAgent[i].predict_mode = mode_

