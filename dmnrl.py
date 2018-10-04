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

# Reward matrix

R = {"start" :  0,
     "A" : 0,
     "B" : 0,
     "middle" : 0,
     "win" : 1.0,
     "loss" : -1.0,
     None : 0.0
}

# State transition probability function
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
    def __init__(self, states = R.keys(), alpha = 0.1,
                 gamma = 0.9, lmbda = 0.9):
        self.V = {}
        self.E = {}
        
        # Inits all state values to zero
        for s in states:
            self.E[s] = 0.0
            self.V[s] = 0.0

        # sets the three parameters
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda
    
        
    def learn_tdlambda(self, s_now, s_next, r):
        print("state %s, reward %.2f" % (s_now, r))
        a = self.alpha
        g = self.gamma
        l = self.lmbda
        rpe = r + g * self.V[s_next] - self.V[s_now]
        #self.V[s_now] += a * rpe
        for s in self.E.keys():
            self.E[s] *= (l * g)
            if s == s_now:
                 self.E[s] += 1

        for s in self.E.keys():
            self.V[s] += a * rpe * self.E[s]


def run(agent, n = 1000):
    for i in range(n):
        history = ["start"]

        while PSS(history)[0] is not None:
            s_now = history[-1]
            s_next, r = PSS(history)
            history.append(s_next)
            agent.learn_tdlambda(s_now, s_next, r)

        s_now = history[-1]
        s_next, r = PSS(history)
        history.append(s_next)
        agent.learn_tdlambda(s_now, s_next, r)
