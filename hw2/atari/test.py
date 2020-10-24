# -*- coding:utf-8 -*-
# DQN homework.
import os
import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers
from utils import *
import keras.backend as K
from tensorboardX import SummaryWriter
from keras.models import load_model

# hyper-parameter.  
EPISODES = 5000

class SumTree(object):
    data_pointer = 0
    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        b_memory=[]
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i]= idx
            b_memory.append(data)
            #print(data.shape)  # s,a,r,s,done
            #print(b_memory.shape)
            #print(data)
            #b_memory[i] = data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see MsPacman learning, then change to True
        self.render = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.9
        self.learning_rate = 0.0001 # 0.00025
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = (self.epsilon-self.epsilon_min) / 1000000  # modified 10000
        self.batch_size = 32
        self.train_start = 32
        self.time_step=0
        # create replay memory using deque
        self.maxlen = 10000 
        #self.memory = deque(maxlen=self.maxlen)
        self.memory = Memory(capacity=self.maxlen)
        # create main model
        self.model_target = self.build_model()
        self.model_eval = self.build_model()

    # approximate Q function using Neural Network
    # you can modify the network to get higher reward.
    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size,activation='relu'))
        model.add(Dense(128*4, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(self.action_size, activation='linear'))
        # model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model_eval.predict(state)
            return np.argmax(q_value[0])

    def get_action_test(self, state):
        q_value = self.model_eval.predict(state)
        return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory  s,a => s',r
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.store([state, action, reward, next_state, done])
        # epsilon decay.
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        #if len(self.memory) < self.train_start:
            #return
        if self.time_step<10000:
            return
        batch_size = self.batch_size
        tree_idx,mini_batch, ISWeights = self.memory.sample(batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]  # state 
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model_eval.predict(update_input)  # Q value
        target_val = self.model_target.predict(update_target)  # target Q value
        abs_errors = np.zeros(target.shape[0])
        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from model
            if done[i]:  # finished
                abs_errors[i] = abs(target[i][action[i]]-reward[i])
                target[i][action[i]] = reward[i]
            else:
                tmp = reward[i] + self.discount_factor * (np.amax(target_val[i]))
                abs_errors[i] = abs(target[i][action[i]]-tmp)
                target[i][action[i]] = tmp
                

        # and do the model fit!
        self.model_eval.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        self.memory.batch_update(tree_idx,abs_errors)
    
    def eval2target(self):  # copy weights
        self.model_target.set_weights(self.model_eval.get_weights())
        
import sys, time

def copy_file_backup(save):
    import shutil, sys, getpass, socket
    backup_dir = os.path.join(save, 'backup_code')
    os.makedirs(backup_dir)
    with open(os.path.join(backup_dir, 'CLI argument.txt'), 'w') as f:
        res = ''.join(['hostName: ', socket.gethostname(), '\n',
                    'account: ', getpass.getuser(), '\n',
                    'save_path: ', os.path.realpath(save), '\n', 
                    'CUDA_VISIBLE_DEVICES: ', str(os.environ.get('CUDA_VISIBLE_DEVICES')), '\n'])
        f.write(res)

        for i, _ in enumerate(sys.argv):
            f.write(sys.argv[i] + '\n')
        
    script_file = sys.argv[0]
    shutil.copy(script_file, backup_dir)
    os.makedirs(os.path.join(backup_dir, 'current_experiment'))
    for file_path in os.listdir(sys.path[0]):
        if file_path not in ['log','logs', 'data', '__pycache__']:
            shutil.copy(os.path.join(sys.path[0], file_path), os.path.join(backup_dir, 'current_experiment'))
            
class Logger(object):
    def __init__(self, filename='terminal log.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
        self.log.write(''.join([time.strftime("%y-%m-%d %H:%M:%S",  time.localtime(time.time())), '\n\n']))

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def __del__(self):
        self.log.write(''.join(['\n', time.strftime("%y-%m-%d %H:%M:%S",  time.localtime(time.time()))]))
        self.log.close()
        
def redirect_stdout(save_path):
    sys.stdout = Logger(os.path.join(save_path, 'stdout.txt'), sys.stdout)
    sys.stderr = Logger(os.path.join(save_path, 'stderr.txt'), sys.stderr)
    
if __name__ == "__main__":
    # load the gym env
    env = gym.make('MsPacman-ram-v0')
    env.seed(0)
    # set  random seeds to get reproduceable result(recommended)
    set_random_seed(0)
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    # create the agent
    agent = DQNAgent(state_size, action_size)
    agent.model_eval=load_model('model_eval.h5')
    # log the training result
    scores, episodes = [], []
    graph_episodes = []
    graph_score = []
    best_score=0
    best_index=0
    avg_length = 10
    sum_score = 0
    rep_freq = 10
    TEST = 20
    it=0
    tune_lr=[2300,4000]
    save_path = os.path.join('/root/AU332/homework2/atariDQN/logs',time.strftime("%y%m%d_%H%M%S"))
    writer = SummaryWriter(log_dir=os.path.join(save_path, 'Tensorboard_Results'))
   
    # os.makedirs(save_path)
    redirect_stdout(save_path)
    copy_file_backup(save_path)
    mp1 = os.path.join(save_path,"model_eval.h5")
    mp2 = os.path.join(save_path,"model_target.h5")
    # print(tf.test.is_built_with_cuda())
    # train DQN
    for e in range(1):
        # K.set_value(agent.model_eval.optimizer.lr, 0.5 * K.get_value(agent.model_eval.optimizer.lr))


        
        # print info and draw the figure.

        if e%avg_length == 0:
            graph_episodes.append(e)
            graph_score.append(sum_score / avg_length)
            sum_score = 0
            writer.add_scalar('score_logs', graph_score[-1], e)
            '''
            # plot the reward each avg_length episodes
            pylab.plot(graph_episodes, graph_score, 'r')
            pylab.savefig(os.path.join(save_path,"pacman_avg.png"))
        '''
        if True:  # test every 100 epochs
            mean_score=0
            for k in range(TEST):
                done = False
                test_score = 0
                state = env.reset()
                state = np.reshape(state, [1, state_size]) # 0~255
                state = state / 255.0
                lives = 3
                while not done: 
                    dead = False         
                    while not dead:
                        # render the gym env
                        if agent.render:
                            env.render()
                        # get action for the current state
                        action = agent.get_action_test(state)
                        # take the action in the gym env, obtain the next state
                        next_state, reward, done, info = env.step(action)
                        next_state = np.reshape(next_state, [1, state_size])
                        next_state = next_state / 255.0
                        # judge if the agent dead
                        dead = info['ale.lives']<lives
                        lives = info['ale.lives']
                        # update score value
                        test_score += reward
                        # go to the next state
                        state = next_state
                print ('episode: ',e,' test: ',k,'Evaluation Reward:',test_score)
                mean_score+=test_score
            mean_score/=TEST  
            if mean_score>best_score:
                best_score=mean_score
                best_index=e
                # save the network if you want to test it.
                print(mean_score)
                print("best score:",best_score)
                print("best_episode:",best_index)
    writer.close()