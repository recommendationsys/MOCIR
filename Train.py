#!/usr/bin/python
# encoding: utf-8

from copyreg import pickle
import numpy as np
import copy as cp
import pickle as pk

import logger
from util import *

import copy as cp

import torch

from model import model
MEMORYSIZE = 50000
BATCHSIZE = 128
THRESHOLD = 300

start = 0
end = 3000

def decay_function1(x):
    x = 50+x
    return max(2.0/(1+np.power(x,0.2)),0.001)

START = decay_function1(start)
END = decay_function1(end)

def decay_function(x):
    x = max(min(end,x),start)
    return (decay_function1(x)-END)/(START-END+0.0000001)


class Train(object):
    def __init__(self,dataset,fa,args):
        self.dataset = dataset
        self.fa = fa
        self.args = args
        self.tau = 0
        self.memory = []
        self.optimizer_nce = torch.optim.Adam(params=self.fa.Model.parameters(), lr=args.learning_rate)
        self.optimizer = torch.optim.Adam(self.fa.parameters(), lr=args.learning_rate) 
        self.loss_func = torch.nn.MSELoss() 
        self.aug_nce_fct = torch.nn.CrossEntropyLoss()

        with open("data/u_rates.pkl", "rb") as f:
            self.rates = pk.load(f)
        self.T = args.T


        

    def train(self):
        for epoch in range(self.args.training_epoch):
            logger.log(epoch)
            self.collecting_data_update_model("training", epoch)
            if epoch % 100 == 0 and epoch>=300:
                self.collecting_data_update_model("validation", epoch)
                self.collecting_data_update_model("evaluation", epoch)

    def collecting_data_update_model(self, type="training", epoch=0):
        if type=="training":
            self.fa.train()
            selected_users = np.random.choice(self.dataset.training,(self.args.inner_epoch,))
        elif type=="validation":
            self.fa.train()
            selected_users = self.dataset.validation
        elif type=="evaluation":
            self.fa.eval()
            selected_users = self.dataset.evaluation
        else:
            selected_users = range(1,3)
        infos = {item:[] for item in self.args.ST}
        used_actions = []
        if 0 in self.dataset.training:
            print(1)
        if 0 in self.dataset.validation:
            print(2)
        if 0 in self.dataset.evaluation:
            print(3)

        for uuid in selected_users: 
            actions = {}
            rwds = 0
            done = False
            state = self.reset_with_users(uuid) 
            

            while not done:
                data = {"uid": [state[0][0]]}

                for path in self.dataset.meta_path:
                    data['p_'+path+'_nei'] = self.dataset.__dict__[path]
                p_r,p_d,pnt = self.convert_item_seq2matrix([[0] + [item[0] for item in state[1]]])
                data['p_rec'] = p_r
                data['p_t'] = pnt
                data['p_d'] = p_d
                policy = self.fa(data,0)

                if type == "training":
                    if np.random.random()<5*THRESHOLD/(THRESHOLD+self.tau):
                        policy = torch.rand(self.args.item_num+1)
                    for item in actions: policy[item] = -torch.inf 
                    action = torch.argmax(policy).item() 
                    
                else:
                    for item in actions: policy[item] = -torch.inf
                    action = torch.argmax(policy).item()
                s_pre = cp.deepcopy(state)  
                state_next, rwd, done, info = self.step(action) 
                if type == "training":
                    self.memory.append([s_pre,action,rwd,done,cp.deepcopy(state_next)]) 
                actions[action] = 1         
                rwds += rwd               
                state = state_next         
                if len(state[1]) in self.args.ST:
                    infos[len(state[1])].append(info)      
            used_actions.extend(list(actions.keys()))
        if type == "training":
            if len(self.memory)>=self.args.batch:
                self.memory = self.memory[-MEMORYSIZE:]
                batch = [self.memory[item] for item in np.random.choice(range(len(self.memory)),(self.args.batch,))] 
                data = self.convert_batch2dict(batch,epoch)
                value = self.fa(data,0)
                max_pi = []
                for i in range(len(data['iid'])):
                    a_indices = (i, data['iid'][i])
                    pi = value[a_indices[0],a_indices[1]]
                    max_pi.append(pi)
                npi = torch.stack(max_pi).to(torch.float64)
                
                loss = self.loss_func(npi, data['goal'])

                self.optimizer.zero_grad()
                loss.backward() 
                self.optimizer.step()

                logger.record_tabular("loss ", "|".join([str(round(loss.item(),4))]))
                self.tau += 5
        for item in self.args.ST:
            logger.record_tabular(str(item)+"precision",round(np.mean([i["precision"] for i in infos[item]]),4))
            logger.record_tabular(str(item)+"recall",round(np.mean([i["recall"] for i in infos[item]]),4))
            logger.log(str(item)+" precision: ",round(np.mean([i["precision"] for i in infos[item]]),4))
        logger.record_tabular("epoch", epoch)
        logger.record_tabular("type", type)
        logger.dump_tabular()

    def convert_batch2dict(self,batch,epoch):
        uids = []
        pos_recs = []
        next_pos = []
        iids = []
        goals = []
        dones = []
        for item in batch:      
            uids.append(item[0][0][1])  
            ep = item[0][1] 
            pos_recs.append([0] + [j[0] for j in ep])    
            iids.append(item[1])    
            goals.append(item[2])   
            if item[3]:dones.append(0.0)
            else:dones.append(1.0)  
            ep = item[4][1]
            next_pos.append([0] + [j[0] for j in ep])

        data = {"uid": uids, "iid": iids}
        for path in self.dataset.meta_path:
            data['p_' + path + '_nei'] = self.dataset.__dict__[path]
        p_r, p_d, pnt = self.convert_item_seq2matrix(pos_recs)
        data["p_rec"] = p_r
        data["p_t"] = pnt
        data['p_d'] = p_d
        path_emb,item_emb = self.fa(data, 1)

        nce_logits1, nce_labels1 = info_nce(path_emb, item_emb, temp=0.5,
                                          batch_size=self.args.batch, sim='dot')

        loss_nce = 0.01 * self.aug_nce_fct(nce_logits1, nce_labels1)


        self.optimizer_nce.zero_grad()
        loss_nce.backward()
        self.optimizer_nce.step()

        logger.record_tabular("loss_nce ", "|".join([str(round(loss_nce.item(), 4))]))


        data = {"uid":uids}
        for path in self.dataset.meta_path:
            data['p_'+path+'_nei'] = self.dataset.__dict__[path]
        p_r, p_d, pnt = self.convert_item_seq2matrix(next_pos)
        data["p_rec"] = p_r
        data["p_t"] = pnt
        data['p_d'] = p_d
        value = self.fa(data,0)
        value[:,0] = -500   
        
        value = value.detach().numpy()
        goals = np.max(value,axis=-1)*np.asarray(dones)*min(self.args.gamma,decay_function(max(end-epoch,0)+1)) + goals
        goals = torch.from_numpy(goals)

        data = {"uid":uids,"iid":iids,"goal":goals}
        for path in self.dataset.meta_path:
            data['p_'+path+'_nei'] = self.dataset.__dict__[path]
        p_r, p_d, pnt = self.convert_item_seq2matrix(pos_recs)
        data["p_rec"] = p_r
        data["p_t"] = pnt
        data['p_d'] = p_d
        return data
    
    def convert_item_seq2matrix(self, item_seq):
        max_length = max([len(item) for item in item_seq])  
        matrix1 = np.zeros((max_length, len(item_seq)),dtype=np.int32)   
        matrix2 = np.zeros((max_length, len(item_seq)),dtype=np.int32)
        for x, xx in enumerate(item_seq):
            for y, yy in enumerate(xx):
                matrix1[y, x] = yy      
                if yy == 0:
                    matrix2[y, x] = yy
                else:
                    matrix2[y, x] = self.dataset.m_directors[yy]
        target_index = list(zip([len(i) - 1 for i in item_seq], range(len(item_seq))))
        return matrix1, matrix2, target_index

    def step(self, action):
       
        if action in self.rates[self.state[0][0]] and (not action in self.short):  
            rate = self.rates[self.state[0][0]][action]
            if rate>=4:
                reward = 1
            else:
                reward = 0
        else:
            rate = 0
            reward = 0

        if len(self.state[1]) < self.T - 1:    
            done = False
        else:
            done = True
        self.short[action] = 1  
        t = self.state[1] + [[action, reward, done]]   
        info = {"precision": self.precision(t),
                "recall": self.recall(t, self.state[0][0]),
                "rate":rate}
        self.state[1].append([action, reward, done, info])
        return self.state, reward, done, info  



    

    def reset_with_users(self, uid):
        self.state = [(uid,1), []]
        self.short = {}
        return self.state

    def precision(self, episode):
        return sum([i[1] for i in episode])

    def recall(self, episode, uid):
        ii = sum(1 for key, value in self.rates[uid].items() if value >= 4)
        if ii == 0:
            ii = 0.0000001
        return sum([i[1] for i in episode]) / ii

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask

def info_nce(z_i, z_j, temp, batch_size, sim='dot'):

    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)
    # print(z.shape)

    if sim == 'cos':
        sim = torch.nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp
    # print(sim.shape)
    # print(batch_size)
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(batch_size)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return logits, labels