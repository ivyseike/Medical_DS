import torch
import torch.nn.functional
import torch.optim as optim
import os
import numpy as np
from collections import namedtuple
import pickle
import copy
import random
import dialog_config


class Model(torch.nn.Module):
    """
    DQN model with one fully connected layer, written in pytorch.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        # different layers. Two layers.
        self.policy_layer = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size, bias=True),
            torch.nn.Dropout(0.3),
            torch.nn.LeakyReLU(),
            # torch.nn.Linear(hidden_size,hidden_size),
            # torch.nn.Dropout(0.5),
            # torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_size, output_size, bias=True)
        )

        # one layer.
        #self.policy_layer = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        q_values = self.policy_layer(x.float())
        return torch.nn.functional.softmax(q_values, dim=1)

class dl_classifier(object):
    def __init__(self, input_size, hidden_size, disease_group_num, disease_num, dise_dict, params):
        hidden_size = 256
        self.device = dialog_config.device
        self.model = Model(input_size=input_size, hidden_size=hidden_size, output_size=disease_group_num).to(self.device)
        self.lowerClassifiers = []

        for i in range(disease_group_num):
            self.lowerClassifiers.append(lower_classifier(input_size=input_size, hidden_size=hidden_size,
                disease_num=disease_num, params=params))
           

        self.dise_dict = dise_dict
        self.dise_num = len(self.dise_dict)

        self.lr = params['lr']
        self.batch_size = 256
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.001)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.batch = []

    def train(self):
        if len(self.batch) < self.batch_size:
            return 
        batch = random.sample(self.batch, self.batch_size)
        state, disease = zip(*batch) 
        state = torch.cat(state, 0)
        disease_tensor = torch.zeros((self.batch_size, self.dise_num))
        for i in range(len(disease)):
            disease_tensor[i][disease[i]] = 1
        
        # slot = torch.LongTensor(batch.slot).to(self.device)
        # disease = torch.LongTensor(batch.disease).to(self.device)
        out = self.model.forward(state)
        #print(disease.shape)
        #print(out.shape)
        #print(out.shape, disease)
        loss = self.criterion(out, disease_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        # self.model.eval()
        with torch.no_grad():
            # print(batch.slot.shape)
            Ys = self.model.forward(x)
            max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        # self.model.train()
        return max_index.item()
    
    def register_experience_tuple(self, state, disease_tag):
        self.batch.append((state, self.dise_dict[disease_tag]))

    # def train_dl_classifier(self, epochs):
    #     for iter in range(epochs):
    #         #print(batch[0][0].shape)
    #         loss = self.train(batch)
    #         if iter%100==0:
    #             print('epoch:{},loss:{:.4f}'.format(iter, loss["loss"]))

    def test_dl_classifier(self):
        self.model.eval()
        self.test_batch = self.create_data(train_mode=False)
        batch = self.Transition(*zip(*self.test_batch))
        slot = torch.LongTensor(batch.slot).to(self.device)
        #disease = torch.LongTensor(batch.disease).to(self.device)
        disease = batch.disease
        Ys, pred = self.predict(slot)
        #print(pred)
        num_correct = len([1 for i in range(len(disease)) if disease[i]==pred[i]])
        print("the test accuracy is %f", num_correct / len(self.test_batch))
        self.model.train()

    def test(self, test_batch):
        #self.model.eval()

        batch = self.Transition(*zip(*test_batch))
        slot = torch.LongTensor(batch.slot).to(self.device)
        #disease = torch.LongTensor(batch.disease).to(self.device)
        disease = batch.disease
        Ys, pred = self.predict(slot.cpu())
        #print(pred)
        num_correct = len([1 for i in range(len(disease)) if disease[i]==pred[i]])
        #print("the test accuracy is %f", num_correct / len(self.test_batch))
        test_acc = num_correct / len(test_batch)
        #self.model.train()
        return test_acc

    def save_model(self,  model_performance, episodes_index, checkpoint_path):
        if os.path.isdir(checkpoint_path) == False:
            os.makedirs(checkpoint_path)
        agent_id = self.parameter.get("agent_id").lower()
        disease_number = self.parameter.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_match_rate = model_performance["average_match_rate"]
        average_match_rate2 = model_performance["average_match_rate2"]
        model_file_name = os.path.join(checkpoint_path, "model_d" + str(disease_number) + str(agent_id) + "_s" + str(
            success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn) \
                                       + "_mr" + str(average_match_rate) + "_mr2-" + str(
            average_match_rate2) + "_e-" + str(episodes_index) + ".pkl")

        torch.save(self.model.state_dict(), model_file_name)

    def restore_model(self, saved_model):
        """
        Restoring the trained parameters for the model. Both current and target net are restored from the same parameter.

        Args:
            saved_model (str): the file name which is the trained model.
        """
        print("loading trained model", saved_model)
        if torch.cuda.is_available() is False:
            map_location = 'cpu'
        else:
            map_location = None
        self.model.load_state_dict(torch.load(saved_model,map_location=map_location))

    def eval_mode(self):
        self.model.eval()



class lower_classifier(object):
    def __init__(self, input_size, hidden_size, disease_num, params):
        hidden_size = 256
        self.device = dialog_config.device
        self.model = Model(input_size=input_size, hidden_size=hidden_size, output_size=disease_num).to(self.device)
           
        self.dise_dict = dise_dict
        self.dise_num = len(self.dise_dict)

        self.lr = params['lr']
        self.batch_size = 256
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=0.001)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.batch = []

    def train(self):
        if len(self.batch) < self.batch_size:
            return 
        batch = random.sample(self.batch, self.batch_size)
        state, disease = zip(*batch) 
        state = torch.cat(state, 0)
        disease_tensor = torch.zeros((self.batch_size, self.dise_num))
        for i in range(len(disease)):
            disease_tensor[i][disease[i]] = 1
        
        # slot = torch.LongTensor(batch.slot).to(self.device)
        # disease = torch.LongTensor(batch.disease).to(self.device)
        out = self.model.forward(state)
        #print(disease.shape)
        #print(out.shape)
        #print(out.shape, disease)
        loss = self.criterion(out, disease_tensor)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, x):
        # self.model.eval()
        with torch.no_grad():
            # print(batch.slot.shape)
            Ys = self.model.forward(x)
            max_index = np.argmax(Ys.detach().cpu().numpy(), axis=1)
        # self.model.train()
        return max_index.item()
    
    def register_experience_tuple(self, state, disease_tag):
        self.batch.append((state, self.dise_dict[disease_tag]))

    # def train_dl_classifier(self, epochs):
    #     for iter in range(epochs):
    #         #print(batch[0][0].shape)
    #         loss = self.train(batch)
    #         if iter%100==0:
    #             print('epoch:{},loss:{:.4f}'.format(iter, loss["loss"]))

    def test_dl_classifier(self):
        self.model.eval()
        self.test_batch = self.create_data(train_mode=False)
        batch = self.Transition(*zip(*self.test_batch))
        slot = torch.LongTensor(batch.slot).to(self.device)
        #disease = torch.LongTensor(batch.disease).to(self.device)
        disease = batch.disease
        Ys, pred = self.predict(slot)
        #print(pred)
        num_correct = len([1 for i in range(len(disease)) if disease[i]==pred[i]])
        print("the test accuracy is %f", num_correct / len(self.test_batch))
        self.model.train()

    def test(self, test_batch):
        #self.model.eval()

        batch = self.Transition(*zip(*test_batch))
        slot = torch.LongTensor(batch.slot).to(self.device)
        #disease = torch.LongTensor(batch.disease).to(self.device)
        disease = batch.disease
        Ys, pred = self.predict(slot.cpu())
        #print(pred)
        num_correct = len([1 for i in range(len(disease)) if disease[i]==pred[i]])
        #print("the test accuracy is %f", num_correct / len(self.test_batch))
        test_acc = num_correct / len(test_batch)
        #self.model.train()
        return test_acc

    def save_model(self,  model_performance, episodes_index, checkpoint_path):
        if os.path.isdir(checkpoint_path) == False:
            os.makedirs(checkpoint_path)
        agent_id = self.parameter.get("agent_id").lower()
        disease_number = self.parameter.get("disease_number")
        success_rate = model_performance["success_rate"]
        average_reward = model_performance["average_reward"]
        average_turn = model_performance["average_turn"]
        average_match_rate = model_performance["average_match_rate"]
        average_match_rate2 = model_performance["average_match_rate2"]
        model_file_name = os.path.join(checkpoint_path, "model_d" + str(disease_number) + str(agent_id) + "_s" + str(
            success_rate) + "_r" + str(average_reward) + "_t" + str(average_turn) \
                                       + "_mr" + str(average_match_rate) + "_mr2-" + str(
            average_match_rate2) + "_e-" + str(episodes_index) + ".pkl")

        torch.save(self.model.state_dict(), model_file_name)

    def restore_model(self, saved_model):
        """
        Restoring the trained parameters for the model. Both current and target net are restored from the same parameter.

        Args:
            saved_model (str): the file name which is the trained model.
        """
        print("loading trained model", saved_model)
        if torch.cuda.is_available() is False:
            map_location = 'cpu'
        else:
            map_location = None
        self.model.load_state_dict(torch.load(saved_model,map_location=map_location))

    def eval_mode(self):
        self.model.eval()