import numpy as np
import gymnasium as gym

import gym_env

from utils import create_transition_matrix_mapping, get_transition_matrix

class SR:
    def __init__(self, env_name, alpha, gamma, epsilon=0.4, num_steps=25000):
        self.env = gym.make(env_name)
        self.start_loc = self.env.unwrapped.start_loc
        self.target_loc = self.env.unwrapped.target_loc
        self.maze = self.env.unwrapped.maze
        self.size = self.maze.size
        self.mapping = create_transition_matrix_mapping(self.maze)
        self.T = get_transition_matrix(self.env, self.size, self.mapping)

        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Rewards
        self.r = np.full(len(self.T), -1)
        self.r[self.terminals] = 10

        # Params
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_steps = num_steps

        # Model
        self.M= np.eye(self.size)	
        self.W= np.zeros(self.size)
        self.V= np.zeros(self.size)
        self.one_hot=np.eye(self.size)

    def onehot_row(self, successor_s):	
        row = np.zeros( len(self.W)) 
        row[successor_s] = 1
        return row

    def get_value(self):
        # self.V = self.M@self.W		
        self.V = self.M@self.r
        return self.V

    def select_action(self, state, policy="e-greedy", epsilon=0.0, target_loc=None):
        if policy == "softmax":
            return
        elif policy == "e-greedy":
            if np.random.uniform(low=0,high=1) < epsilon:
                return self.env.unwrapped.random_action()
            else:
                action_values = np.full(self.env.action_space.n, -np.inf)
                for action in self.env.unwrapped.get_available_actions(state):
                    direction = self.env.unwrapped._action_to_direction[action]
                    new_state = state + direction
                    if self.maze[new_state[0], new_state[1]] == "1":
                        continue
                    action_values[action] = self.V[self.mapping[(new_state[0],new_state[1])]]

                return np.argmax(action_values)
            
        elif policy == "test":
            action_values = np.full(self.env.action_space.n, -np.inf)
            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction

                # Need this to make it work for now
                if np.array_equal(new_state, target_loc):
                    return action

                if self.maze[new_state[0], new_state[1]] == "1":
                    continue
                action_values[action] = self.V[self.mapping[(new_state[0],new_state[1])]]

            return np.argmax(action_values)  

    def learn(self):
        """
        Agent explores maze 
        """
        self.env.reset()

        # Iterate through number of steps
        for i in range(self.num_steps):
            # Current state
            state = self.env.unwrapped.agent_loc
            state_idx = self.mapping[(state[0], state[1])]

            # Choose action (random for now)
            action = self.env.unwrapped.random_action()
            # action = self.select_action(state, epsilon=self.epsilon)

            # Take action
            obs, _, done, _, _ = self.env.step(action)

            if done:
                self.env.reset()
                continue

            # Unpack observation to get new state
            next_state = obs["agent"]
            next_state_idx = self.mapping[(next_state[0], next_state[1])]

            # Update Successor Representation
            self.M[state_idx] = (1-self.alpha) * self.M[state_idx] + self.alpha * ( self.one_hot[state_idx] + self.gamma * self.M[next_state_idx]  )

            # Update state
            state = next_state
        
        self.V = self.get_value()




## OLD LINEAR-RL THAT WORKED ON LATENT-NEW BEFORE
class LinearRL:
    def __init__(self, env_name, alpha=0.01, _lambda=1.0, gamma=0.9, epsilon=0.4, num_steps=25000):
        self.env = gym.make(env_name)
        self.start_loc = self.env.unwrapped.start_loc
        self.target_loc = self.env.unwrapped.target_loc
        self.maze = self.env.unwrapped.maze
        self.size = self.maze.size
        self.height, self.width = self.maze.shape
        self.mapping = self.create_transition_matrix_mapping()
        self.T = self.get_transition_matrix(size=self.size, mapping=self.mapping)

        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]
        # self.P = self.T[:, self.terminals]
        # Calculate reward
        self.r = np.full(len(self.T), -1)     # our reward at each non-terminal state to be -1
        self.r[self.terminals] = 10           # reward at terminal state is 10
        self.expr = np.exp(self.r[self.terminals] / _lambda)

        # Params
        self.alpha = alpha
        self.gamma = gamma
        self._lambda = _lambda
        self.epsilon = epsilon
        self.num_steps = num_steps

        # Model
        self.DR = np.eye(self.size)
        # self.DR = np.zeros((self.size, self.size))
        self.Z = np.zeros(self.size)
        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)
    
    def create_transition_matrix_mapping(self):
        """
        Creates a mapping from maze state indices to transition matrix indices
        """
        n = len(self.maze)  # Size of the maze (N)

        mapping = {}
        matrix_idx = 0

        for i in range(n):
            for j in range(n):
                mapping[(i,j)] = matrix_idx
                matrix_idx += 1

        return mapping
    
    def get_transition_matrix(self, size, mapping):
        T = np.zeros(shape=(size, size))
        # loop through the maze
        for row in range(self.maze.shape[0]):
            for col in range(self.maze.shape[1]):            
                # if we hit a barrier
                if self.maze[row,col] == '1':
                    continue

                idx_cur = mapping[row, col]

                # check if current state is terminal
                if self.maze[row,col] == 'G':
                    T[idx_cur, idx_cur] = 1
                    continue

                state = (row,col)
                successor_states = self.env.unwrapped.get_successor_states(state)
                for successor_state in successor_states:
                    idx_new = mapping[successor_state[0][0], successor_state[0][1]]
                    T[idx_cur, idx_new] = 1/len(successor_states)
        
        return T
    
    def update_V(self):
        self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr
        self.Z[self.terminals] = np.exp(self.r[self.terminals] / self._lambda)
        self.V = np.round(np.log(self.Z), 2)
    
    def importance_sampling(self, state, s_new_idx):
        successor_states = self.env.unwrapped.get_successor_states(state)
        p = 1/len(successor_states)
        w = (p * self.Z[s_new_idx]) / sum(p * self.Z[self.mapping[(s[0][0],s[0][1])]] for s in successor_states)
        
        return w
    
    def select_action(self, state, policy="e-greedy", epsilon=0.0, target_loc=None):
        if policy == "softmax":
            return
        elif policy == "e-greedy":
            if np.random.uniform(low=0,high=1) < epsilon:
                return self.env.unwrapped.random_action()
            else:
                action_values = np.full(self.env.action_space.n, -np.inf)
                for action in self.env.unwrapped.get_available_actions(state):
                    direction = self.env.unwrapped._action_to_direction[action]
                    new_state = state + direction
                    if self.maze[new_state[0], new_state[1]] == "1":
                        continue
                    action_values[action] = round(np.log(self.Z[self.mapping[(new_state[0],new_state[1])]]), 2)

                return np.argmax(action_values)
            
        elif policy == "test":
            action_values = np.full(self.env.action_space.n, -np.inf)
            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction

                # Need this to make it work for now
                if np.array_equal(new_state, target_loc):
                    return action

                if self.maze[new_state[0], new_state[1]] == "1":
                    continue
                action_values[action] = round(np.log(self.Z[self.mapping[(new_state[0],new_state[1])]]), 2)
                print(action_values)

            return np.argmax(action_values)
    
    def learn(self):
        """
        Agent randomly explores the maze and and updates its DR as it goes
        """
        self.env.reset()

        # Iterate through number of steps
        for i in range(self.num_steps):
            # Current state
            state = self.env.unwrapped.agent_loc
            state_idx = self.mapping[(state[0], state[1])]

            # Choose action (random for now)
            action = self.env.unwrapped.random_action()
            # action = self.select_action(state, epsilon=self.epsilon)

            # Take action
            obs, _, done, _, _ = self.env.step(action)

            if done:
                self.env.reset()
                continue

            # Unpack observation to get new state
            next_state = obs["agent"]
            next_state_idx = self.mapping[(next_state[0], next_state[1])]

            # Update Default Representation
            # w = self.importance_sampling(state, next_state_idx)
            # w = 1 if np.isnan(w) or w == 0 else w
            w = 1
            TDE =  self.one_hot[state_idx][~self.terminals] + self.gamma * self.DR[next_state_idx][~self.terminals]
            # TDE =  self.one_hot[state_idx] + self.gamma * self.DR[next_state_idx]

            self.DR[state_idx][~self.terminals] = (1 - self.alpha) * self.DR[state_idx][~self.terminals] + self.alpha * TDE * w
            # self.DR[state_idx] = (1 - self.alpha) * self.DR[state_idx] + self.alpha * TDE * w


            # Update Z-Values
            self.Z[state_idx] = self.DR[state_idx][~self.terminals] @ self.P @ self.expr
            # self.Z[state_idx] = self.DR[state_idx] @ self.P @ self.expr

            # Update state
            state = next_state
        
        self.Z[self.terminals] = np.exp(self.r[self.terminals] / self._lambda)
        self.V = np.round(np.log(self.Z), 2)