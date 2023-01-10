import random
import numpy as np


class Maze():
    """A maze environment"""

    ACTIONS = ("up", "down", "left", "right")
    STATE = (0, 0)
    
    def __init__(self, fname = "grid.txt"):
        self.grid = np.loadtxt(fname)
        self.state = self.STATE

        
    def executeAction(self, action):
        """Executes one of four possible actions: up, down, right, and left"""
        s = self.state
        new_s = s
        new_r = -1
        
        if action in self.ACTIONS:
            if action == "up":
                if s[0] > 0:
                    new_s = (s[0] - 1, s[1])
                    new_r = self.grid[new_s[0], new_s[1]]
            
            elif action == "left":
                if s[1] > 0:
                    new_s = (s[0], s[1] - 1)
                    new_r = self.grid[new_s[0], new_s[1]]
            
            elif action == "down":
                if s[0] < (self.grid.shape[0] - 1):
                    new_s = (s[0] + 1, s[1])
                    new_r = self.grid[new_s[0], new_s[1]]

            else:
                if s[1] < (self.grid.shape[1] -1):
                    new_s = (s[0], s[1] + 1)
                    new_r = self.grid[new_s[0], new_s[1]]

        # If you hit the jackpot, you go back to square #1
        if new_r == 10:
            new_s =  self.STATE

        # Update and return tuple
        self.state = new_s
        return (new_s, new_r)

    def print_state(self):
        "Prints a grid representation of the state"
        bar = "-" * ( 4 * self.grid.shape[1] + 1)
        for i in range(self.grid.shape[0]):
            row = "|"
            for j in range(self.grid.shape[1]):
                cell = " "
                if i == self.state[0] and j == self.state[1]:
                    cell = "*"
                row += (" %s |" % cell)
            print(bar)
            print(row)
        print(bar)

