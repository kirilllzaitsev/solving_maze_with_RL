import numpy as np
from numpy.random import random_integers as rnd
import matplotlib.pyplot as plt

class Environment:
    
    def __init__(self, size):
        self.name = 'I am ENVIRONMENT'
        self._size = size
        self._action_space = self.init_action_space(size)
        self.state = (1, 1)

    def update_state(self, action):
        '''
        Update current env state.

        Inputs:
        - action (int): number in range(0, num_actions)

        Returns:
        - next_state (tuple): agent's position after action
        '''
        pass

    def get_state(self):
        return self.state

    def init_action_space(self, shape, complexity=.5, density =.5):
        # Only ODD shapes
        # Adjust complexity and density relative to action_space size
        complexity = int(complexity*(5*(shape[0]+shape[1])))
        density    = int(density*(shape[0]//2*shape[1]//2))
        # Build actual action_space
        Z = np.zeros(shape, dtype=bool)
        # Fill borders
        Z[0,:] = Z[-1,:] = 1
        Z[:,0] = Z[:,-1] = 1
        # Make isles
        for i in range(density):
            x, y = rnd(0,shape[1]//2)*2, rnd(0,shape[0]//2)*2
            Z[y,x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:           
                    neighbours.append( (y,x-2) )
                if x < shape[1]-2:  
                    neighbours.append( (y,x+2) )
                if y > 1:           
                    neighbours.append( (y-2,x) )
                if y < shape[0]-2:  
                    neighbours.append( (y+2,x) )
                if len(neighbours):
                    y_,x_ = neighbours[rnd(0,len(neighbours)-1)]
                    if Z[y_,x_] == 0:
                        Z[y_,x_] = 1
                        Z[y_+(y-y_)//2, x_+(x-x_)//2] = 1
                        x, y = x_, y_
        return Z.astype('int')

    def display_env(self):
        plt.figure(figsize=(10,5))
        plt.imshow(self._action_space,cmap=plt.cm.binary,interpolation='nearest')
        plt.xticks([]),plt.yticks([])
        plt.show()

    def step(self, action):
        return tuple([1,2,3])

    def reset(self):
        pass

