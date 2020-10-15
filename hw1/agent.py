import random, re, datetime, copy
from queue import PriorityQueue

class Agent(object):
    def __init__(self, game):
        self.game = game

    def getAction(self, state):
        raise Exception("Not implemented yet")


class RandomAgent(Agent):
    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)


class SimpleGreedyAgent(Agent):  
    # a one-step-lookahead greedy agent that returns action with max vertical advance
    def getAction(self, state):
        legal_actions = self.game.actions(state)  # actions:[pos,new_pos]

        self.action = random.choice(legal_actions)

        player = self.game.player(state)  # state:[player,board]
        if player == 1:   # choose action with the biggest step
            max_vertical_advance_one_step = max([action[0][0] - action[1][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[0][0] - action[1][0] == max_vertical_advance_one_step]
        else:
            max_vertical_advance_one_step = max([action[1][0] - action[0][0] for action in legal_actions])
            max_actions = [action for action in legal_actions if
                           action[1][0] - action[0][0] == max_vertical_advance_one_step]
        self.action = random.choice(max_actions)


class FlappyAgent(Agent):
    def __init__(self, game):
        super().__init__(game)
        self.lastaction = None  # my last action
        self.exp_depth = 2   # current exploration depth, start from 2 
        self.end = 0   # not used
        self.breadth = 20     # the maximal breadth one layer extends
    def getAction(self, state):
        legal_actions = self.game.actions(state)
        self.action = random.choice(legal_actions)

        player = self.game.player(state)

        depth = 2   # initial exploration depth 
        self.exp_depth = depth  
 
        value = -1e5   # namely negative infinity

        pq = PriorityQueue()  # sort legal actions
        for action in legal_actions:
            if player==1:
                if action[0][0]>3:      # constrain the most rows one piece can step backward in order to prune 
                    if action[0][0] - action[1][0] < -1:
                        continue
                else:
                    if action[0][0] - action[1][0] < -2:
                        continue                    
            else:
                if action[0][0] <17:
                    if action[0][0] - action[1][0] > 1:
                        continue
                else:
                    if action[0][0] - action[1][0] > 2:
                        continue
            if self.lastaction != None and action[0]==self.lastaction[1] and action[1] == self.lastaction[0]: # prevent being stuck 
                continue
            pq.put((-(2*player-3)*(action[1][0]-action[0][0]),action)) # sort by step length
        next_pq=PriorityQueue()
        while True:
            cnt = 0
            while self.breadth > cnt and not pq.empty():
                action = pq.get()[1]
                if self.getValue(self.game.succ(state, action),player) == 1e4:   # I win 
                    self.action = action 
                    break
                cnt += 1
                next_value=self.minimax(value,1e5,self.game.succ(state,action),depth,player)  # player is always me
                next_pq.put((-next_value,action))
                if next_value > value:
                    value = next_value
                    self.action=action
                    
            self.lastaction = self.action
            depth+=1  # if there is time left, increase the exploration depth
            self.exp_depth=depth
            del pq
            pq = PriorityQueue()  # renew the priority queue
            while not next_pq.empty():
                pq.put(next_pq.get())
            

        

    def getValue(self, state, player):  
        board = state[1]
        my_pos = board.getPlayerPiecePositions(player)
        enemy_pos = board.getPlayerPiecePositions(3-player)
        my_dist_from_home = 0  # the average of vertical distance of all my pieces from home
        enemy_dist_from_home = 0 
        my_col_dist = 0  # the sum of horizontal distance of some pieces from the middle column
        enemy_col_dist = 0
        var = 0 # the variance of my rows
        var2=0
        value=0  # reward for arrived pieces
        value2=0
        
        my_arr_num=0  # the number of arrived pieces 
        #enemy_arr_num=0
        if player==1:  
            special_loc1=[(2,1),(2,2),(3,2)]   # terminal for three special pieces 
            # special_loc2=[(18,1),(18,2),(17,2)]
            for pos in my_pos:
                my_dist_from_home += (20 - pos[0])/10.0  # average distance 

                if pos[0]<=4:
                    if pos in special_loc1:   # get reward if arriving at right terminal, lose reward if arriving at wrong terminal 
                        if board.board_status[pos] == 3:
                            value+=5
                            my_arr_num+=1
                        else:
                            value-=5
                    else:
                        if board.board_status[pos] == 1:
                            value+=5
                            my_arr_num+=1
                        else:
                            value-=5
                            if pos[0]==1:
                                value-=50
                else:
                    if pos[0]&1:
                        if pos[1]==(board.getColNum(pos[0])+1)/2:  # the exact middle column may lead to "stuck" situation 
                            my_col_dist+=1
                        else:
                            my_col_dist += abs(pos[1]-(board.getColNum(pos[0])+1)/2)-1
                    else:
                        my_col_dist+=abs(pos[1]-(board.getColNum(pos[0])+1)/2)
            
            for pos in enemy_pos:   # the same for enemy except for the reward
                enemy_dist_from_home+=pos[0]/10.0
                if pos[0]<16:
                    if pos[0]&1:
                        if pos[1]==(board.getColNum(pos[0])+1)/2:
                            enemy_col_dist+=1
                        else:
                            enemy_col_dist += abs(pos[1]-(board.getColNum(pos[0])+1)/2)-1
                    else:
                        enemy_col_dist+=abs(pos[1]-(board.getColNum(pos[0])+1)/2)
            for pos in my_pos:
                var += abs(20-pos[0] - my_dist_from_home)  # wrong
            for pos in enemy_pos:
                var2 += abs(pos[0] - enemy_dist_from_home)
        else:
            #special_loc2=[(2,1),(2,2),(3,2)]
            special_loc1=[(18,1),(18,2),(17,2)]
            for pos in my_pos:
                my_dist_from_home += pos[0] /10.0
                
                if pos[0]>=16:
                    # value+=10
                    if pos in special_loc1:
                        if board.board_status[pos] == 4:
                            value+=5
                            my_arr_num+=1
                        else:
                            value-=5
                    else:
                        if board.board_status[pos] == 2:
                            value+=5
                            my_arr_num+=1
                        else:
                            value-=5
                            if pos[0]==19:
                                value-=50
                else:
                    if pos[0]&1:
                        if pos[1]==(board.getColNum(pos[0])+1)/2:
                            my_col_dist+=1
                        else:
                            my_col_dist += abs(pos[1]-(board.getColNum(pos[0])+1)/2)-1
                    else:
                        my_col_dist+=abs(pos[1]-(board.getColNum(pos[0])+1)/2)
            for pos in enemy_pos:
                enemy_dist_from_home+=20-pos[0]
                if pos[0]>4:
                    if pos[0]&1:
                        if pos[1]==(board.getColNum(pos[0])+1)/2:
                            enemy_col_dist+=1
                        else:
                            enemy_col_dist += abs(pos[1]-(board.getColNum(pos[0])+1)/2)-1
                    else:
                        enemy_col_dist+=abs(pos[1]-(board.getColNum(pos[0])+1)/2)
            for pos in my_pos:
                var += abs(pos[0] - my_dist_from_home)
            for pos in enemy_pos:
                var2 += abs(20-pos[0] - enemy_dist_from_home)
        if my_arr_num==10:  # I win
            return 1e4
        #if enemy_arr_num==10:  # I lose
            #return -1e4
        score = 24*(my_dist_from_home-enemy_dist_from_home) - 0.2 * (var-var2)-0.6*(my_col_dist-enemy_col_dist)+value # -value2
        return score


                        
    def minimax(self, alpha, beta, state, depth, player):
        if depth==0:
            return self.getValue(state, player)
        depth -= 1
        if state[0]!=player:  # it's enemy's turn
            if depth != self.exp_depth-1:
                if self.getValue(state,player)==1e4:  # I win
                    return 1e4
            
            actions = self.game.actions(state) 
            pq = PriorityQueue()
            for action in actions:  # enemy's actions
                pq.put(((2 * player-3) * (action[1][0] - action[0][0]), action))
            cnt =0
            value=1e5
            while self.breadth>cnt and not pq.empty():
                action=pq.get()[1]
                cnt+=1
                value=min(value,self.minimax(alpha,beta,self.game.succ(state,action),depth,player))  # min node
                if value<=alpha:   # prune
                    return value
                beta=min(beta,value)
            return value
        else: # it's my turn
            #if self.getValue(state,player) == -1e4:
                #return -1e4 
            actions=self.game.actions(state)
            value=-1e5
            pq=PriorityQueue()
            for action in actions:  # my actions
                pq.put((-(2 * player-3) * (action[1][0] - action[0][0]), action))
            cnt=0
            while self.breadth>cnt and not pq.empty():
                action=pq.get()[1]
                cnt+=1
                value=max(value,self.minimax(alpha,beta,self.game.succ(state,action),depth,player))  # max node
                if value>=beta:  # prune
                    return value
                alpha=max(alpha,value)
            return value
    
    ''' For lack of time, the functions below are not used but they may have certain potential to improve our agent. '''
    def GetFirstLastPiece(self, all_pos, player):
        first = all_pos[0][0]
        last = all_pos[0][0]        
        if player==1:
            for pos in all_pos:
                first = min(first,pos[0])
                last = max(last, pos[0])
        else:
            for pos in all_pos:
                first = max(first,pos[0])
                last = min(last,pos[0])
        return first,last

    def near_end(self, my_last, enemy_last, player):
        return enemy_last > my_last if player == 1 else my_last > enemy_last

    def midtime(self, my_first, enemy_first, player):
        return my_first - enemy_first <= 2 if player == 1 else enemy_first - my_first <= 2
    
    def maximax(self, state, depth, player):
        if depth==0:
            return self.getValue2(state, player)
        if self.getValue2(state,player)==1e4:
            return 1e4 # *depth
        depth -= 1

        actions=self.game.actions(state)
        value=-1e5
        pq=PriorityQueue()
        for action in actions:  # my actions
            pq.put((-(2 * player-3) * (action[1][0] - action[0][0]), action))
        cnt=0
        while self.breadth>cnt and not pq.empty():
            action=pq.get()[1]
            cnt+=1
            value=max(value,self.maximax(self.pseudo_succ(state,action),depth,player))
        return value


    def getValue2(self, state, player):  # suppose depth is even and this is always my turn
        board = state[1]
        my_pos = board.getPlayerPiecePositions(player)
        my_dist_from_home = 0  # the average of vertical distance of all my pieces from home
        my_col_dist = 0  # the sum of horizontal distance of some pieces from the middle column
        var = 0 # the variance of my rows
        value=0  # reward for arrived pieces
        my_arr_num=0  # the number of arrived pieces 

        if player==1:  
            special_loc1=[(2,1),(2,2),(3,2)]   # terminal for three special pieces 
            for pos in my_pos:
                my_dist_from_home += (20 - pos[0])/10.0  # average distance 

                if pos[0]<=4:
                    if pos in special_loc1:   # get reward if arriving at right terminal, lose reward if arriving at wrong terminal 
                        if board.board_status[pos] == 3:
                            value+=5
                            my_arr_num+=1
                        else:
                            value-=10
                    else:
                        if board.board_status[pos] == 1:
                            value+=5
                            my_arr_num+=1
                        else:
                            value-=5
                            if pos[0]==1:
                                value-=5
                else:
                    if pos[0]&1:
                        if pos[1]==(board.getColNum(pos[0])+1)/2:  # the exact middle column may lead to "stuck" situation 
                            my_col_dist+=1
                        else:
                            my_col_dist += abs(pos[1]-(board.getColNum(pos[0])+1)/2)-1
                    else:
                        my_col_dist+=abs(pos[1]-(board.getColNum(pos[0])+1)/2)

            for pos in my_pos:
                var += abs(20-pos[0] - my_dist_from_home) 

        else:
            special_loc1=[(18,1),(18,2),(17,2)]
            for pos in my_pos:
                my_dist_from_home += pos[0] /10.0
                
                if pos[0]>=16:
                    if pos in special_loc1:
                        if board.board_status[pos] == 4:
                            value+=5
                            my_arr_num+=1
                        else:
                            value-=10
                    else:
                        if board.board_status[pos] == 2:
                            value+=5
                            my_arr_num+=1
                        else:
                            value-=5
                            if pos[0]==19:
                                value-=5
                else:
                    if pos[0]&1:
                        if pos[1]==(board.getColNum(pos[0])+1)/2:
                            my_col_dist+=1
                        else:
                            my_col_dist += abs(pos[1]-(board.getColNum(pos[0])+1)/2)-1
                    else:
                        my_col_dist+=abs(pos[1]-(board.getColNum(pos[0])+1)/2)

            for pos in my_pos:
                var += abs(pos[0] - my_dist_from_home)
        if my_arr_num==10:  # I win
            return 1e4

        score = 24*my_dist_from_home - 0.2 * var-0.6*my_col_dist+value
        return score

    def pseudo_succ(self, state, action): # always my turn, regardless of enemy's action
        board = copy.deepcopy(state[1])
        board.board_status[action[1]] = board.board_status[action[0]]
        board.board_status[action[0]] = 0
        return (state[0], board)
    
