import numpy as np
import pandas as pd
import random
SEED=0
random.seed(SEED)
np.random.seed(SEED)
class Agent:
    ### START CODE HERE ###

    def __init__(self, actions):
        self.actions = actions
        self.epsilon = 0.9
        self.epsilon_min = 0
        self.q_table={}
        self.model=env_model()
        # self.STEPS=500
        # self.step = 0
        self.gamma = 0.7
        self.alpha = 0.1
        self.N=20

    def choose_action(self, s): 
        state=(s[0],s[1],s[2],s[3],s[4])
        if state not in self.q_table.keys():
            self.q_table[state]=np.array([0.0,0.0,0.0,0.0])
        
        if random.random()<self.epsilon:
            action = np.random.choice(np.where(self.q_table[state]>=0.0)[0])
        else:
            action=np.random.choice(np.where(self.q_table[state]==np.max(self.q_table[state]))[0])
        # self.step+=1
        if self.epsilon>self.epsilon_min:
            self.epsilon-=(self.epsilon-self.epsilon_min)/70
        return action

    def learn(self,s,a,r,s_,done):
        state=(s[0],s[1],s[2],s[3],s[4])
        next_state=(s_[0],s_[1],s_[2],s_[3],s_[4])
        if next_state not in self.q_table.keys():
            self.q_table[next_state]=np.array([0.0,0.0,0.0,0.0])
        if not done:
            q_target=r+self.gamma*np.max(self.q_table[next_state])
        else:
            q_target=r
        #print(q_target,r)
        self.q_table[state][a]+=self.alpha*(q_target-self.q_table[state][a])
        



class env_model:
    def __init__(self):
        self.model={}

    def update(self,s,a,r,s_,done):
        state=(s[0],s[1],s[2],s[3],s[4])
        next_state=(s_[0],s_[1],s_[2],s_[3],s_[4])
        if state not in self.model.keys():
            self.model[state]={}
        # if a not in self.model[s].keys():
        #     self.model[s][a]=[0,(0,0,0,0)]
        self.model[state][a]=[r,next_state,done]
    
    def sample_s_a(self):
        # print(list(self.model.keys()))
        idx = np.random.choice(range(self.get_len()))
        s = list(self.model.keys())[idx]
        a = np.random.choice(list(self.model[s].keys()))
        return s,a


    def get_r_s_done(self,s,a):
        tmp=self.model[s][a]
        return tmp[0],tmp[1],tmp[2]

    def get_len(self):
        return len(self.model.keys())
    ### END CODE HERE ###