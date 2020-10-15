from maze_env import Maze
from RL_brain import Agent
import time


if __name__ == "__main__":
    ### START CODE HERE ###
    # This is an agent with random policy. You can learn how to interact with the environment through the code below.
    # Then you can delete it and write your own code.
    u=1
    env = Maze()
    agent = Agent(actions=list(range(env.n_actions)))
    # print(list(range(env.n_actions)))
    for episode in range(200):
        s = env.reset()  # x1,y1,x2,y2
        episode_reward = 0
        s.append(False)
        while True:
            env.render()                 # You can comment all render() to turn off the graphical interface in training process to accelerate your code.
            a = agent.choose_action(s)
            s_, r, done = env.step(a)
            agent.learn(s,a,r,s_,done)
            agent.model.update(s,a,r,s_,done)
            episode_reward += r
            # print('a:',a)
            # print('s:',s)
            # print('r:',r)
            # print('s_:',s_)
            # print('done:',done)
            # if u==4:
            #     break
            # u+=1
            for i in range(agent.N):
                # if agent.model.get_len()==1:
                #     break
                ms,ma=agent.model.sample_s_a()
                mr,ms_,mdone=agent.model.get_r_s_done(ms,ma)
                agent.learn(ms,ma,mr,ms_,mdone)
                
            s = s_
            #print(s)
            if done:
                env.render()
                time.sleep(0.5)
                break
        # print(agent.q_table)
        print('episode:', episode, 'episode_reward:', episode_reward)

    ### END CODE HERE ###

    print('\ntraining over\n')
