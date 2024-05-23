import numpy as np
import gymnasium as gym


import gym_env
from utils import get_transition_matrix, create_mapping_nb, gen_nhb_exp

class LinearRL:
    def __init__(self, env_name, alpha=0.1, beta=1, _lambda=1.0, epsilon=0.4, num_steps=25000, policy="random", imp_samp=False):
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
        self.reward_nt = -1   # Non-terminal state reward
        self.reward_t = -1    # Terminal state reward
        self.r = np.full(len(self.T), self.reward_nt)
        self.r[self.terminals] = self.reward_t
        self.expr_t = np.exp(self.r[self.terminals] / _lambda)
        # Precalculate exp(r) for use with LinearRL equations
        self.expr_nt = np.exp(self.reward_nt / _lambda)

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = self.expr_nt
        # self.gamma = 0.9
        self._lambda = _lambda
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.policy = policy
        self.imp_samp = imp_samp

        # Model
        self.DR = self.get_DR()
        self.Z = np.full(self.size, 0.01)

        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def get_states(self):
        """
        Returns all non-blocked states as well as a mapping of each state (i,j) -> to an index (k)
        """
        states = []
        index_mapping = {}
        index = 0
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] in ['0', 'S', 'G']:
                    states.append((i, j))
                    index_mapping[(i, j)] = index
                    index += 1

        return states, index_mapping

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

    def update_V(self):
        self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
        self.Z[self.terminals] = self.expr_t
        self.V = np.round(np.log(self.Z), 2)
    
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
            successor_states = self.env.unwrapped.get_successor_states(state)      # succesor_states = [(state, terminated), ...]
            action_probs = np.full(self.env.action_space.n, 0.0)

            v_sum = sum(
                        np.exp(np.round((np.log(self.Z[self.mapping[(s[0][0],s[0][1])]] + 1e-20)),3) / self.beta) for s in successor_states
                        )

            # if we don't have enough info, random action
            if v_sum == 0:
                return self.env.unwrapped.random_action() 

            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction
                
                action_probs[action] = np.exp(np.round((np.log(self.Z[self.mapping[(new_state[0], new_state[1])]] + 1e-20)),3) / self.beta ) / v_sum

            action = np.random.choice(self.env.action_space.n, p=action_probs)
            s_prob = action_probs[action]

            return action, s_prob
            
        elif self.policy == "test":
            action_values = np.full(self.env.action_space.n, -np.inf)
            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction

                # Need this to make it work for now
                # np.any(np.all(self.target_locs == new_state, axis=1))
                if np.any(np.all(self.target_locs == new_state, axis=1)):
                    return action

                if self.maze[new_state[0], new_state[1]] == "1":
                    continue
                action_values[action] = round(np.log(self.Z[self.mapping[(new_state[0],new_state[1])]]), 2)

            return np.nanargmax(action_values)

    def get_D_inv(self):
        """
        Calculates the DR directly using matrix inversion, used for testing
        """
        I = np.eye(self.size)
        D_inv = np.linalg.inv(I-self.gamma*self.T)

        return D_inv

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
            
            ## Update default representation
            target = self.one_hot[state_idx] + self.gamma * self.DR[next_state_idx]
            self.DR[state_idx] = (1 - self.alpha) * self.DR[state_idx] + self.alpha * target * w

            ## Update Z-Values
            self.Z = self.DR[:,~self.terminals] @ self.P @ self.expr_t

            if done:
                self.env.reset(seed=seed, options={})
                continue
            
            # Update state
            state = next_state

        # Update DR at terminal state
        self.Z[self.terminals] = np.exp(self.r[self.terminals] / self._lambda)
        self.V = np.round(np.log(self.Z), 2)

class LinearRL_NHB:
    def __init__(self, alpha=0.25, beta=10, _lambda=10, epsilon=0.4, num_steps=25000, policy="softmax", imp_samp=True, exp_type="policy_reval"):
        # Hard code start and end locations as well as size
        self.start_loc = 0
        self.target_locs = [3,4,5]
        self.size = 9
        self.agent_loc = self.start_loc
        self.exp_type = exp_type

        # Construct the transition probability matrix and envstep functions
        self.T = self.construct_T()
        self.envstep = gen_nhb_exp()
        
        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]

        # Set reward
        self.reward_nt = -1   # Non-terminal state reward (set to 0 for SR)
        self.r = np.full(len(self.T), self.reward_nt)
        # Reward of terminal states depends on if we are replicating reward revaluation or policy revaluation
        if self.exp_type == "policy_reval":
            self.r_term_1 = [0, 15, 30]
            self.r_term_2 = [45, 15, 30]
        elif self.exp_type == "reward_reval":
            self.r_term_1 = [15, 0, 30]
            self.r_term_2 = [45, 0, 30]
        elif self.exp_type == "trans_reval":
            self.r_term_1 = [15, 0, 30]
        else:
            print("Incorrect experiment type (exp_type)")
            return(0)
        self.r[self.terminals] = self.r_term_1

        # Precalculate exp(r) for use with LinearRL equations
        self.expr_t = np.exp(self.r[self.terminals] / _lambda)
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

        # Model
        self.DR = self.get_DR()
        self.Z = np.full(self.size, 0.01)

        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def construct_T(self):
        """
        Manually construt the transition matrix
        """
        # For NHB two-step task
        T = np.zeros((9, 9))
        T[0, 1:3] = 0.5
        T[1, 3:5] = 0.5
        T[2, 4:6] = 0.5
        T[3, 6] = 1
        T[4, 7] = 1
        T[5, 8] = 1
        T[6:9, 6:9] = np.eye(3)

        return T
    
    def construct_T_new(self):
        """
        Manually construct a new transition matrix for the transition revaluation problem
        """
        T = np.zeros((9, 9))
        T[0, 1:3] = 0.5
        T[1, 4:6] = 0.5
        T[2, 3:5] = 0.5
        T[3, 6] = 1
        T[4, 7] = 1
        T[5, 8] = 1
        T[6:9, 6:9] = np.eye(3)

        return T

    
    def update_exp(self):
        """
        Update the terminal state values or transition matrix (experiment dependent)
        """
        if self.exp_type in ["reward_reval", "policy_reval"]:
            self.r[self.terminals] = self.r_term_2

    def get_DR(self):
        """
        Returns the DR initialization based on what decision policy we are using, values are filled with 0.01 if using softmax to avoid div by zero
        """
        if self.policy == "softmax":
            DR = np.full((self.size, self.size), 0.01)
            np.fill_diagonal(DR, 1)
            DR[np.where(self.terminals)[0], np.where(self.terminals)[0]] = (1/(self.gamma))
        else:
            DR = np.eye(self.size)

        return DR

    def update_Z(self):
        self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
        self.Z[self.terminals] = self.expr_t

    def update_V(self):
        self.V = np.round(np.log(self.Z), 2)
    
    def get_successor_states(self, state):
        """
        Manually define the successor states based on which state we are in
        """
        return np.where(self.T[state, :] != 0)[0]

    def importance_sampling(self, state, s_prob):
        """
        Performs importance sampling P(x'|x)/u(x'|x). P(.) is the default policy, u(.) is the decision policy
        """
        successor_states = self.get_successor_states(state)
        p = 1/len(successor_states)
        w = p/s_prob
                
        return w

    def select_action(self, state):
        """
        Action selection based on our policy
        Options are: [random, softmax, egreedy, test]
        """
        if self.policy == "random":
            action = np.random.choice([0,1])

            return action
        
        elif self.policy == "softmax":
            successor_states = self.get_successor_states(state)
            action_probs = np.full(2, 0.0)   # We can hardcode this because every state has 2 actions

            v_sum = sum(np.exp((np.log(self.Z[s] + 1e-20)) / self.beta) for s in successor_states)

            # if we don't have enough info, random action
            if v_sum == 0:
                return  np.random.choice([0,1])

            for action in [0,1]:
                new_state, done = self.envstep[state, action]

                # If we hit a done state our action doesn't matter
                if done:
                    action = np.random.choice([0,1])
                    return action, 1
                action_probs[action] = np.exp((np.log(self.Z[new_state] + 1e-20)) / self.beta ) / v_sum
                
            action = np.random.choice([0,1], p=action_probs)
            s_prob = action_probs[action]

            return action, s_prob

    def get_D_inv(self):
        """
        Calculates the DR directly using matrix inversion, used for testing
        """
        I = np.eye(self.size)
        D_inv = np.linalg.inv(I-self.gamma*self.T)

        return D_inv

    def learn(self):
        """
        Agent explores the maze according to its decision policy and and updates its DR as it goes
        """
        # Iterate through number of steps
        for i in range(self.num_steps):
            # Agent gets some knowledge of terminal state values
            if i == 2:
                self.Z[self.terminals] = self.expr_t
            # Current state
            state = self.agent_loc

            # Choose action
            if self.policy == "softmax":
                action, s_prob = self.select_action(state)
            else:
                action = self.select_action(state)
        
            # Take action
            next_state, done = self.envstep[state, action]
            
            # Importance sampling
            if self.imp_samp:
                w = self.importance_sampling(state, s_prob)
                w = 1 if np.isnan(w) or w == 0 else w
            else:
                w = 1
            
            # Update default representation
            target = self.one_hot[state] + self.gamma * self.DR[next_state]
            self.DR[state] = (1 - self.alpha) * self.DR[state] + self.alpha * target * w

            # Update Z-Values
            self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
            
            if done:
                self.agent_loc = self.start_loc
                continue
            
            # Update state
            state = next_state
            self.agent_loc = state

        # Update DR at terminal state
        self.update_Z()
        self.update_V()

class SR_NHB:
    def __init__(self, alpha=0.1, beta=1, gamma=0.904, num_steps=25000, policy="random", exp_type="policy_reval"):
        # Hard code start and end locations as well as size
        self.start_loc = 0
        self.target_locs = [3,4,5]
        self.size = 9
        self.agent_loc = self.start_loc
        self.exp_type = exp_type

        # Construct the transition probability matrix and envstep functions
        self.T = self.construct_T()
        self.envstep = gen_nhb_exp()
        
        # Get terminal states
        self.terminals = np.diag(self.T) == 1

        # Set reward
        self.reward_nt = -1   # Non-terminal state reward (set to 0 for SR)
        self.r = np.full(len(self.T), self.reward_nt)
        # Reward of terminal states depends on if we are replicating reward revaluation or policy revaluation
        if self.exp_type == "policy_reval":
            self.r_term_1 = [0, 15, 30]
            self.r_term_2 = [45, 15, 30]
        elif self.exp_type == "reward_reval":
            self.r_term_1 = [15, 0, 30]
            self.r_term_2 = [45, 0, 30]
        else:
            print("Incorrect experiment type (exp_type)")
            return(0)
        self.r[self.terminals] = self.r_term_1

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_steps = num_steps
        self.policy = policy

        # Model
        self.SR = np.eye(self.size)
        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def construct_T(self):
        """
        Manually construct a T that is biased by a decision policy
        """
        T = np.zeros((9, 9))
        T[0,1] = 0.3
        T[0,2] = 0.7
        T[1,3] = 0.1
        T[1,4] = 0.9
        T[2,4] = 0.2
        T[2,5] = 0.8
        T[3, 6] = 1
        T[4, 7] = 1
        T[5, 8] = 1
        T[6:9, 6:9] = np.eye(3)

        return T
    
    def update_term(self):
        """
        Update the terminal state values (experiment dependent)
        """
        self.r[self.terminals] = self.r_term_2

    def update_V(self):
        self.V = self.SR @ self.r
    
    def get_successor_states(self, state):
        """
        Manually define the successor states based on which state we are in
        """
        return np.where(self.T[state, :] != 0)[0]

    def select_action(self, state):
        """
        Action selection based on our policy
        Options are: [random, softmax, egreedy, test]
        """
        if self.policy == "random":
            successor_states = self.get_successor_states(state)
            return np.random.choice(successor_states)
        
        elif self.policy == "softmax":
            successor_states = self.get_successor_states(state)
            action_probs = np.full(2, 0.0)   # We can hardcode this because every state has 2 actions

            V = self.V[successor_states]
            exp_V = np.exp(V / self.beta)
            action_probs = exp_V / exp_V.sum()
    
            action = np.random.choice([0,1], p=action_probs)

            return action

    def learn(self):
        """
        Agent explores the maze according to its decision policy and and updates its DR as it goes
        """
        # Iterate through number of steps
        for i in range(self.num_steps):
            # Current state
            state = self.agent_loc

            # Choose action
            if self.policy == "softmax":
                action = self.select_action(state)
        
            # Take action
            next_state, done = self.envstep[state, action]
            
            # Update default representation
            target = self.one_hot[state] + self.gamma * self.SR[next_state]
            self.SR[state] = (1 - self.alpha) * self.SR[state] + self.alpha * target

            # Update Values
            self.update_V()

            if done:
                self.agent_loc = self.start_loc
                continue
            
            # Update state
            state = next_state
            self.agent_loc = state

        # Update DR at terminal state
        self.update_V()