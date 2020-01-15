#

## Non-markov problem
##
##          A          Win 1.0
##        /   \      /
##  start      middle
##        \   /      \ 
##          B          Loss -1.0
##

import random

# Reward functio

R = {"start" :  0,
     "A" : 0,
     "B" : 0,
     "middle" : 0,
     "win" : 1.0,
     "loss" : -1.0,
     None : 0.0
}

# State transition probavility function
#
def PSS(history):
    s_current = history[-1]
    s_next = None
    
    if s_current is "start":
        coin = random.uniform(0, 1)
        if coin <= 0.5:
            s_next = "A"
        else:
            s_next = "B"
        
    elif s_current is "A" or s_current is "B":
        s_next = "middle"

    elif s_current is "middle":
        s_previous = history[-2]
        if s_previous is "A":
            s_next = "win"
        elif s_previous is "B":
            s_next = "loss"
        
    elif s_current is "win" or s_current is "loss":
        s_next = None

    return (s_next, R[s_current])
    

## The agent just experiences a set of 
class Agent:
    def __init__(self, states = R.keys(), alpha = 0.1, gamma = 0.9):
        self.V = {}

        # Inits all state values to zero
        for s in states:
            self.V[s] = 0.0

        # sets the two parameters
        self.alpha = alpha
        self.gamma = gamma
        
    def learn_td(self, s_now, s_next, r):
        print("state %s, reward %.2f" % (s_now, r))
        a = self.alpha
        g = self.gamma
        rpe = r + g * self.V[s_next] - self.V[s_now]
        self.V[s_now] += a * rpe


def run(agent, n = 1000):
    for i in range(n):
        history = ["start"]

        while PSS(history)[0] is not None:
            s_now = history[-1]
            s_next, r = PSS(history)
            history.append(s_next)
            agent.learn_td(s_now, s_next, r)

        s_now = history[-1]
        s_next, r = PSS(history)
        history.append(s_next)
        agent.learn_td(s_now, s_next, r)
