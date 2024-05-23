import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.animation as manimation
import random


def update_terminal_reward(agent, loc, r):
    """
    Update the reward for the terminal state of the agent according to loc

    Args:
        agent (LinearRL class) : The LinearRL agent
        loc (int) : The terminal location to change the reward of [0->n] n= number of terminal locations - 1
        r (float) : The new reward to change r[loc] to
    """
    # Get location of reward and change
    r_loc = np.argwhere(agent.terminals)[loc]
    agent.r[r_loc] = r
    # Update expr_t inside of the agent
    agent.expr_t = np.exp(agent.r[agent.terminals] / agent._lambda)

def replan_barrier(agent, T_new, delta_locs, inv=False):
    """
    Performs replanning when the transition structure of the environment changes

    Args:
        agent (LinearRL Agent Class) : Original DR to use
        T_new (array) : New transition matrix when transition to a state is blocked
        delta_locs (list) : States whose transitions have changed
    Returns:
        D (array) : Updated DR after performing Woodbury
    """
    T = agent.T
    r = agent.r

    L0 = np.diag(np.exp(-r/agent._lambda)) - T
    L = np.diag(np.exp(-r/agent._lambda)) - T_new

    delta = L[delta_locs] - L0[delta_locs]

    if inv:
        D0 = np.linalg.inv(L0)
    else:
        D0 = agent.gamma * agent.DR
    D0_j = D0[:,delta_locs]

    I = np.eye(len(delta_locs))
    inv = np.linalg.inv(I + np.dot(delta, D0_j))

    B = np.dot( np.dot(D0_j, inv), np.dot(delta, D0) )

    D = D0 - B

    return D

def policy_reval(agent):
    """
    Performs replanning when the reward structure of the environment has changed
    
    Args:
        agent (LinearRL class) : The LinearRL agent

    Returns:
        V_new (array) : New value of each state
    """
    r_new = agent.r
    expr_new = np.exp(r_new[agent.terminals] / agent._lambda)
    Z_new = np.zeros(len(r_new))

    Z_new[~agent.terminals] = agent.DR[~agent.terminals][:,~agent.terminals] @ agent.P @ expr_new
    Z_new[agent.terminals] = expr_new
    V_new = np.round(np.log(Z_new), 2)

    return V_new, Z_new

def new_goal(agent, T, loc):
    """
    New Environment is the same as the old one, with the inclusion of a new goal state that we want to use the old DR to
    plan towards
    
    Args:
    agent (LinearRL class): The LinearRL agent 
    T (array): The transition matrix of the new environment
    loc (tuple): Location of the new goal state
    """
    D0 = agent.DR
    L0 = np.diag(np.exp(-agent.r)) - agent.T
    L = np.diag(np.exp(-agent.r)) - T

    idx = agent.mapping[loc]

    d = L[idx, :] - L0[idx, :]
    m0 = D0[:,idx]

    # Convert d to a row vector of size (1, m)
    d = d.reshape(1, -1)

    # Convert m0 to a column vector of size (m, 1)
    m0 = m0.reshape(-1, 1)

    # Get the amount of change to the DR
    alpha = (np.dot(m0,d)) / (1 + (np.dot(d,m0)))
    change = np.dot(alpha,D0)

    # Apply change to DR
    D = np.copy(D0)
    D -= change

    # Set agent's DR to new DR
    agent.DR = D

    # Update terminals
    agent.terminals = np.diag(T) == 1
    # Update P
    agent.P = T[~agent.terminals][:,agent.terminals]
    # Update reward
    agent.r = np.full(len(T), -1)     # our reward at each non-terminal state to be -1
    agent.r[agent.terminals] = 20         # reward at terminal state is 20
    agent.expr = np.exp(agent.r[agent.terminals] / agent._lambda)

def create_mapping(maze):
    """
    Creates a mapping from maze state indices to transition matrix indices
    """
    n = len(maze)  # Size of the maze (N)

    mapping = {}
    matrix_idx = 0

    for i in range(n):
        for j in range(n):
            mapping[(i,j)] = matrix_idx
            matrix_idx += 1

    return mapping

def create_mapping_nb(maze, walls):
    """
    Creates a mapping from maze state indices to transition matrix indices
    This mapping *excludes* blocks that are inacessible, hence the nb stands for "not blocked"
    """
    n = len(maze)  # Size of the maze (N)

    # Create a mapping from maze state indices to transition matrix indices
    mapping = {}
    matrix_idx = 0

    for i in range(n):
        for j in range(n):
            if (i, j) not in walls:
                mapping[(i, j)] = matrix_idx
                matrix_idx += 1

    return mapping

def get_transition_matrix_nb(env, size, mapping):
    """
    Creates a state -> state transition matrix. This means the transition matrix *excludes* blocks that are inacessible, hence the nb stands for "not blocked"
    """
    barriers = []
    maze = env.unwrapped.maze

    T = np.zeros(shape=(size, size))
    # loop through the maze
    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):  
            # if we hit a barrier
            if maze[row,col] == '1':
                barriers.append(mapping[row, col])
                continue

            idx_cur = mapping[row, col]

            # check if current state is terminal
            if maze[row,col] == 'G':
                T[idx_cur, idx_cur] = 1
                continue

            state = (row,col)
            successor_states = env.unwrapped.get_successor_states(state)
            for successor_state in successor_states:
                idx_new = mapping[successor_state[0][0], successor_state[0][1]]
                T[idx_cur, idx_new] = 1/len(successor_states)
    
    return T, barriers

def get_transition_matrix(env, mapping):
    """
    Creates a block -> block transition matrix. This means the transition matrix *includes* blocks that are inacessible
    """
    maze = env.unwrapped.maze

    T = np.zeros(shape=(len(mapping), len(mapping)))
    # loop through the maze
    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):            
            # if we hit a barrier
            if maze[row,col] == '1':
                continue

            idx_cur = mapping[row, col]

            # check if current state is terminal
            if maze[row,col] == 'G':
                T[idx_cur, idx_cur] = 1
                continue

            state = (row,col)
            successor_states = env.unwrapped.get_successor_states(state)
            for successor_state in successor_states:
                idx_new = mapping[successor_state[0][0], successor_state[0][1]]
                T[idx_cur, idx_new] = 1/len(successor_states)
    
    return T

def get_map(agent):
    # Replace 'S' and 'G' with 0
    m = np.where(np.isin(agent.maze, ['S', 'G']), '0', agent.maze)

    # Convert the array to int
    m = m.astype(int)
    
    return m

def get_full_maze_values(agent):
    """
    Function that prints out the values of each state and labels blocked states
    """
    v_maze = np.zeros_like(agent.maze, dtype=np.float64)
    for row in range(v_maze.shape[0]):
        for col in range(v_maze.shape[1]):
            if agent.maze[row, col] == "1":
                v_maze[row,col] = -np.inf
                continue
            v_maze[row,col] = agent.V[agent.mapping[(row,col)]]
    
    return v_maze

def decision_policy(agent, Z):
    """
    Performs matrix version of equation 6 from the LinearRL paper

    Args:
        agent (LinearRL class) : The LinearRL agent
        Z (array) : The Z-Values to operate on (usually just agent.Z)

    Returns:
        pii (array) : The decision policy
    """
    G = np.dot(agent.T, Z)

    expv_tiled = np.tile(Z, (len(Z), 1))
    G = G.reshape(-1, 1)
    
    zg = expv_tiled / G
    pii = agent.T * zg

    return pii

def gen_nhb_exp():
    """
    Defines the environment for the Nature Human Behavior experiment introduced by Momennejad et al.
    """
    envstep=[]
    for s in range(6):
        # actions 0=left, 1=right
        envstep.append([[0,0], [0,0]])  # [s', done]
    envstep = np.array(envstep)

    # State 0 -> 1, 2
    envstep[0,0] = [1,0]
    envstep[0,1] = [2,0]

    # State 1 -> 3, 4
    envstep[1,0] = [3,0]
    envstep[1,1] = [4,0]

    # State 2 -> 4, 5
    envstep[2,0] = [4,0]
    envstep[2,1] = [5,0]

    # State 3 -> 3'
    envstep[3,0] = [6,1]
    envstep[3,1] = [6,1]

    # State 4 -> 4'
    envstep[4,0] = [7,1]
    envstep[4,1] = [7,1]

    # State 5 -> 5'
    envstep[5,0] = [8,1]
    envstep[5,1] = [8,1]
    
    return envstep

def test_agent(agent, state=None):
    """
    Function to test the agent
    """
    # Set the policy to testing
    agent.policy = "test"

    traj = []

    agent.env.reset()
    if state is None:
        state = agent.start_loc

    # set the start and agent location
    agent.env.unwrapped.start_loc, agent.env.unwrapped.agent_loc = state, state
    # print(f"Starting in state: {state}")
    steps = 0
    done = False
    while not done:
        action = agent.select_action(state)

        obs, _, done, _, _ = agent.env.step(action)
        next_state = obs["agent"]
        traj.append(next_state)
        # print(f"Took action: {action} and arrived in state: {next_state}")

        steps += 1
        state = next_state
    # print(f"Took {steps} steps")

    return traj