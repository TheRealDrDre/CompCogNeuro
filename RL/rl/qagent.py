import random
import numpy as np

class QAgent():
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        self.q = {}

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)
        
    def learnQ(self, state, action, reward, value):
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    def run(self, task, n):
        i = 0
        while i < n:
            s = task.state
            a = self.chooseAction(s)
            new_s, r = task.executeAction(a)
            self.learn(s, a, r, new_s)
            i += 1

    def calculateV(self, task):
        v = np.zeros(task.grid.shape)
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                maxq = max([self.getQ((i, j), a) for a in self.actions])
                v[i,j] =maxq
        return v

    
def run_trials_for_altair(environment, agent, n, collect=True):
    """Runs N trials"""
    state_action = {} 
    # init all state_actions
    for i in range(4):
        for j in range(4):
            for direction in ['down', 'right', 'up', 'left']:
                state_action[str(((i, j), direction))] = [0] 

    for j in range(n):
        run_trial(environment, agent)
        all_keys = set(state_action.keys())
        # keys with new values
        for key, val in agent.Q.items():
            state_action[str(key)].append(val)
            all_keys.remove(str(key))
        # keys without new values
        for key in all_keys:
            state_action[str(key)].append(state_action[str(key)][-1])
        environment.state = Maze.INITIAL_STATE
        
    import pandas as pd
    location = []
    run = []
    q_value = []
    for loc in state_action.keys():
        for i in range(len(state_action[loc])):
            location.append(loc)
            run.append(i)
            q_value.append(state_action[loc][i])
    df = pd.DataFrame({"location":location, "run":run, "q_value":q_value})
    return df    

m = Maze()
a = Agent()
df = run_trials_for_altair(m, a, 100)

import altair as alt

slider = alt.binding_range(min=1, max=100, step=1)
select_run = alt.selection_single(name="iteration", fields=['run'], bind=slider)
alt.data_transformers.enable('default', max_rows=None)
alt.Chart(df).mark_bar().encode(
    x='location:N',
    y=alt.Y('q_value:Q', scale=alt.Scale(domain=(0, 11))),
).add_selection(
    select_run
).transform_filter(
    select_run
)