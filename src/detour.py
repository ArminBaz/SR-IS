import os
import random

import numpy as np
import gymnasium as gym

import gym_env
from utils import decision_policy, woodbury
from utils import get_transition_matrix, create_mapping_nb
# from models import LinearRL

# Set the random seed for NumPy
seed = 182
np.random.seed(seed)

# Hyperparams
reward = -0.1
alpha = 0.55
beta = 2.0
_lambda = 1.0
num_steps = 80000
decay = True
decay_params = [0.99, 150]

# # Set the random seed for NumPy
# seed = 182
# np.random.seed(seed)

# # Hyperparams
# reward = -0.1
# alpha = 0.55
# beta = 1.5
# _lambda = 1.0
# num_steps = 150000
# decay = True
# decay_params = [0.99, 300]

# For plotting
prob_locs = [12, 4, 6]
colors = [3, 2, 4]

# Save dir
save_dir = os.path.join('..', 'figures/')

## Helper functions
def exponential_decay(initial_learning_rate, decay_rate, global_step, decay_steps):
    """
    Applies exponential decay to the learning rate.
    """
    learning_rate = initial_learning_rate * decay_rate ** (global_step / decay_steps)
    return learning_rate

## LinearRL Agent
class LinearRL:
    def __init__(self, env_name, reward=-0.2, alpha=0.1, beta=1, _lambda=1.0, epsilon=0.4, num_steps=25000, policy="random", imp_samp=False, decay=False, decay_params=None, random_restarts=True):
        self.env = gym.make(env_name)
        self.start_loc = self.env.unwrapped.start_loc
        self.target_locs = self.env.unwrapped.target_locs
        self.maze = self.env.unwrapped.maze
        self.walls = self.env.unwrapped.get_walls()
        self.size = self.maze.size - len(self.walls)   # Size of the state space is the = size of maze - number of blocked states
        self.height, self.width = self.maze.shape

        # Create mapping and Transition matrix
        self.mapping = create_mapping_nb(self.maze, self.walls)
        self.reverse_mapping = {index: (i, j) for (i, j), index in self.mapping.items()}
        self.T = get_transition_matrix(self.env, self.mapping)

        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]
        # Set reward
        self.reward_nt = reward   # Non-terminal state reward
        self.reward_t = 10    # Terminal state reward
        self.r = np.full(len(self.T), self.reward_nt)
        self.r[self.terminals] = self.reward_t
        self.expr_t = np.exp(self.r[self.terminals] / _lambda)
        # Precalculate exp(r) for use with LinearRL equations
        self.expr_nt = np.exp(self.reward_nt / _lambda)

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = self.expr_nt
        self._lambda = _lambda
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.policy = policy
        self.imp_samp = imp_samp
        self.decay = decay

        if decay:
            self.inital_lr = alpha
            self.decay_rate = decay_params[0]
            self.decay_steps = decay_params[1]


        # Model
        self.DR = self.get_DR()
        self.Z = np.full(self.size, 0.01)

        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def get_DR(self):
        """
        Returns the DR initialization based on what decision policy we are using, values are filled with 0.01 if using softmax to avoid div by zero
        """
        if self.policy == "random":
            DR = np.eye(self.size)
            DR[np.where(self.terminals)[0], np.where(self.terminals)[0]] = (1/(1-self.gamma))
        
        elif self.policy == "softmax":
            DR = np.full((self.size, self.size), 0.01)
            np.fill_diagonal(DR, 1)
            DR[np.where(self.terminals)[0], np.where(self.terminals)[0]] = (1/(1-self.gamma))

        return DR
    
    def get_D_inv(self):
        """
        Calculates the DR directly using matrix inversion, used for testing
        """
        I = np.eye(self.size)
        D_inv = np.linalg.inv(I-self.gamma*self.T)

        return D_inv

    def update_V(self):
        self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
        self.Z[self.terminals] = self.expr_t
        self.V = np.round(np.log(self.Z), 5)
    
    def importance_sampling(self, state, s_prob):
        """
        Performs importance sampling P(x'|x)/u(x'|x). P(.) is the default policy, u(.) us the decision policy
        """
        successor_states = self.env.unwrapped.get_successor_states(state)
        p = 1/len(successor_states)
        w = p/s_prob

        return w

    def select_action(self, state):
        """
        Action selection based on our policy
        Options are: [random, softmax, egreedy, test]
        """
        if self.policy == "random":
            return self.env.unwrapped.random_action()
        
        elif self.policy == "softmax":
            successor_states = self.env.unwrapped.get_successor_states(state)
            action_probs = np.full(self.env.action_space.n, 0.0)

            v_sum = sum(
                        np.exp((np.log(self.Z[self.mapping[(s[0][0],s[0][1])]] + 1e-20)) / self.beta) for s in successor_states
                        )

            # if we don't have enough info, random action
            if v_sum == 0:
                return self.env.unwrapped.random_action() 

            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction
                # print(self.Z[self.mapping[(new_state[0], new_state[1])]])
                action_probs[action] = np.exp( (np.log( self.Z[self.mapping[(new_state[0], new_state[1])]] + 1e-20 )) / self.beta ) / v_sum

            action = np.random.choice(self.env.action_space.n, p=action_probs)
            s_prob = action_probs[action]

            return action, s_prob
            
        elif self.policy == "test":
            action_values = np.full(self.env.action_space.n, -np.inf)
            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction

                # Need this to make it work for now
                if np.any(np.all(self.target_locs == new_state, axis=1)):
                    return action

                if self.maze[new_state[0], new_state[1]] == "1":
                    continue
                action_values[action] = self.V[self.mapping[(new_state[0], new_state[1])]]

            return np.nanargmax(action_values)

    def take_action(self, state):
        """
        Select an action according to policy and take it. 
        """
        # Choose action
        if self.policy == "softmax":
            action, s_prob = self.select_action(state)
        else:
            action = self.select_action(state)

        # Take action
        obs, _, done, _, _ = self.env.step(action)

        # Unpack observation to get new state
        next_state = obs["agent"]
        next_state_idx = self.mapping[(next_state[0], next_state[1])]

        # Importance sampling
        if self.imp_samp:
            w = self.importance_sampling(state, s_prob)
            w = 1 if np.isnan(w) or w == 0 else w
        else:
            w = 1
        
        return next_state, next_state_idx, w, done

    def update(self, state_idx, next_state_idx, w, step):
        """
        Update the DR
        """
        ## If we are using lr decay
        if self.decay:
            self.alpha = exponential_decay(self.inital_lr, self.decay_rate, step, self.decay_steps)

        ## Update default representation
        target = self.one_hot[state_idx] + self.gamma * self.DR[next_state_idx]
        self.DR[state_idx] = (1 - self.alpha) * self.DR[state_idx] + self.alpha * target * w

        ## Update Z-Values
        self.Z = self.DR[:,~self.terminals] @ self.P @ self.expr_t

    def learn(self, seed=None):
        """
        Agent explores the maze according to its decision policy and and updates its DR as it goes
        """
        # print(f"Decision Policy: {self.policy}, Number of Iterations: {self.num_steps}, lr={self.alpha}, temperature={self.beta}, importance sampling={self.imp_samp}")
        self.env.reset(seed=seed, options={})

        # Iterate through number of steps
        for i in range(self.num_steps):
            # Update terminal state information after 2 steps
            if i == 2:
                self.Z[self.terminals] = self.expr_t

            # Current state
            state = self.env.unwrapped.agent_loc
            state_idx = self.mapping[(state[0], state[1])]

            # Take action
            next_state, next_state_idx, w, done = self.take_action(state)

            # Update  DR
            self.update(state_idx, next_state_idx, w, i)

            if done:
                self.env.reset(seed=seed, options={})
                continue
            
            # Update state
            state = next_state

        # Update DR at terminal state
        self.Z[self.terminals] = np.exp(self.r[self.terminals] / self._lambda)
        self.V = np.round(np.log(self.Z), 2)

agent_with_imp = LinearRL(env_name="tolman-9x9-nb", reward=reward, _lambda=_lambda, beta=beta, alpha=alpha, num_steps=num_steps, policy="softmax", imp_samp=True, decay=decay, decay_params=decay_params)
agent_with_imp.learn(seed=seed)
if agent_with_imp.decay:
    print(agent_with_imp.decay_rate, agent_with_imp.decay_steps)
print(agent_with_imp.alpha)

pii_with_imp = decision_policy(agent_with_imp, agent_with_imp.Z)
probs_train_with_imp = pii_with_imp[5][prob_locs]
probs_train_with_imp_n = probs_train_with_imp / np.sum(probs_train_with_imp)

# Get new maze and initialize a new agent
env_blocked = gym.make("tolman-9x9-b")
maze_blocked = env_blocked.unwrapped.maze
new_agent = LinearRL(env_name="tolman-9x9-nb", _lambda=_lambda, beta=beta, alpha=alpha)

# Block transition to from state 12 -> state 15
# transition_state = 12
# blocked_state = 15
transition_state = 15
blocked_state = 18

# Make a new transition matrix that doesn't allow transition to blocked state
T_new = np.copy(agent_with_imp.T)

# Set transitions to the blocked state to 0
T_new[transition_state, blocked_state] = 0
T_new[blocked_state, transition_state] = 0

# Calculate the sum of transitions for each row, excluding transitions to the blocked state
row_sums = np.sum(T_new, axis=1)

# Normalize non-zero transitions
non_zero_indices = np.where(row_sums != 0)
T_new[non_zero_indices] /= row_sums[non_zero_indices][:, np.newaxis]

D_new_with = woodbury(agent_with_imp, T_new, inv=False)

Z_values_with, V_values_with = np.zeros(new_agent.size), np.zeros(new_agent.size)
Z_values_with[~agent_with_imp.terminals] = D_new_with[~agent_with_imp.terminals][:,~agent_with_imp.terminals] @ agent_with_imp.P @ np.array([np.exp(10)])
Z_values_with[agent_with_imp.terminals] = np.exp(10)

np.where(Z_values_with == np.min(Z_values_with))
Z_values_with += np.abs(np.min(Z_values_with)) + 0.1

V_values_with = np.log(Z_values_with)
print(V_values_with[4], V_values_with[6], V_values_with[12])