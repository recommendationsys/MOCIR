import pandas as pd
import numpy as np
import torch
import re
import random
import collections
random.seed(13)
import copy as cp
import pickle
import copy

class DataSet(object):
    def __init__(self, args):
        self.T = args.T
        self.i_fea = {}
        self.i_rates = {}
        self.i_genres = {}
        self.i_actors = {}
        self.i_directors = {}
        self.mgm = {}
        
        self.mam = {}
        self.mdm = {}
        self.args = args
        
        self.input_dir = args.data_path

        self.ui_data = pd.read_csv(self.input_dir + 'ratings.dat', names=['user', 'item', 'rating', 'timestamp'], sep="::", engine='python')
        self.user_data = pd.read_csv(self.input_dir + 'users.dat', names=['user', 'gender', 'age', 'occupation_code', 'zip'], sep="::", engine='python')
        self.item_data = pd.read_csv(self.input_dir + 'movies_extrainfos.dat', names=['item', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'], sep="::", engine='python', encoding="utf-8")

        self.rate_list = self.load_list("{}/m_rate.txt".format(self.input_dir))   
        self.genre_list = self.load_list("{}/m_genre.txt".format(self.input_dir))
        self.actor_list = self.load_list("{}/m_actor.txt".format(self.input_dir))
        self.director_list = self.load_list("{}/m_director.txt".format(self.input_dir))

        self.user_list = list(set(self.ui_data.user.tolist()))
        self.item_list = list(set(self.ui_data.item.tolist()))
        
        print("user_num:", len(self.user_list), "item_num", len(self.item_list))
        print("user_data.user.shape", len(self.user_data.user))

    
    

    def load_list(self, fname):
        list_ = []
        with open(fname, encoding="utf-8") as f:
            for line in f.readlines():  
                list_.append(line.strip()) 
        return list_
    
    
    def item_converting(self, row, rate_list, genre_list, director_list, actor_list):
        rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()


        genre_idx = torch.zeros(1, len(genre_list)).long()   

        genre_id = []
        for genre in str(row['genre']).split(", "):
            idx = genre_list.index(genre)  
            genre_idx[0, idx] = 1  
            genre_id.append(idx + 1)  


        director_idx = torch.zeros(1, len(director_list)).long()  

        director_id = []
        for director in str(row['director']).split(", "):
            idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
            director_idx[0, idx] = 1
            director_id.append(idx + 1)  


        actor_idx = torch.zeros(1, len(actor_list)).long() 

        actor_id = []
        for actor in str(row['actors']).split(", "):
            idx = actor_list.index(actor)
            actor_idx[0, idx] = 1
            actor_id.append(idx + 1)

        return torch.cat((rate_idx, genre_idx),1), rate_idx, genre_idx, director_idx, actor_idx, genre_id, director_id,actor_id
    

    def reverse_dict(self, d):
        re_d = collections.defaultdict(list)
        for k, v_list in d.items():
            for v in v_list:
                re_d[v].append(k)
        return dict(re_d)

    def find_nei(self,m_g,g_m):
        m_g_m = {}
        for i in m_g:
            mm = []
            for j in m_g[i]:
                mm += g_m[j]
            m_g_m[i] = list(set(mm))
        return m_g_m

    
    def find_fea(self, mmm, i_fea):
        mxm = cp.deepcopy(mmm)
        for i in mxm:
            mxm_fea = []
            for j,jj in enumerate(mxm[i]):
                mxm_fea.append(i_fea[jj])
            mxm[i] = torch.cat(mxm_fea, dim=0)
        return mxm

    def dataprocess(self):

        m_genres = {}
        m_directors = {}        
        m_actors = {}          
        m_idx = {}
        self.items = {}

        with open('data/m_idx.pkl', 'rb') as f:
            m_idx = pickle.load(f)

        for idx, row in self.item_data.iterrows():  
            m_info = self.item_converting(row, self.rate_list, self.genre_list, self.director_list, self.actor_list)
            if row['item'] not in self.item_list:
                continue
            self.i_fea[m_idx[row['item']]] = m_info[0]
            self.i_rates[m_idx[row['item']]] = m_info[1]
            self.i_genres[m_idx[row['item']]] = m_info[2]
            self.i_directors[m_idx[row['item']]] = m_info[3]
            self.i_actors[m_idx[row['item']]] = m_info[4]
            
            m_genres[m_idx[row['item']]] = m_info[5]
            m_directors[m_idx[row['item']]] = m_info[6]
            m_actors[m_idx[row['item']]] = m_info[7]

        g_movies = self.reverse_dict(m_genres)
        a_movies = self.reverse_dict(m_actors)
        d_movies = self.reverse_dict(m_directors)
        

        self.mgm = self.find_nei(m_genres,g_movies)
        self.mam = self.find_nei(m_actors,a_movies)
        self.mdm = self.find_nei(m_directors,d_movies)

        self.m_g_m = self.find_fea(self.mgm,self.i_fea)
        self.m_a_m = self.find_fea(self.mam,self.i_fea)
        self.m_d_m = self.find_fea(self.mdm,self.i_fea)

        self.m_directors = m_directors
        for director in self.m_directors:
            self.m_directors[director] = self.m_directors[director][0]
        
        self.meta_path = ['mgm','mam','mdm']
        
        self.setup_train_test()


    def setup_train_test(self):
        users = self.user_list
        np.random.shuffle(users)
        self.training, self.validation, self.evaluation = np.split(np.asarray(users), [int(.85 * self.user_num),int(.9 * self.user_num)])
        
    @property
    def rate_num(self):
        return len(self.rate_list)
    @property
    def genre_num(self):
        return len(self.genre_list)
    @property
    def actor_num(self):
        return len(self.actor_list)
    @property
    def director_num(self):
        return len(self.director_list)
    
    @property
    def user_num(self):
        return len(self.user_list)
    
    @property
    def item_num(self):
        return len(self.item_list)
