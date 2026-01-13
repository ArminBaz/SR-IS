import pickle
import os

import numpy as np
from scipy.io import loadmat

from cbm.individual_fit import individual_fit, Config
from cbm.model_selection import bms
from prepare_data_decothi import create_transition_matrix_open, extract_all_subjects_decothi


# Constants
TERM_IDX = 33  # Terminal state index (fixed across all mazes)
N_STATES = 100  # 10x10 grid


def maze_to_transition_matrix(maze):
    """
    Convert a maze to a transition matrix.
    """
    working_maze = maze.copy()
    working_maze[working_maze > 0] = 0

    rows, cols = working_maze.shape

    # Create mapping from maze coordinates to state indices
    coord_to_state = {}
    state_to_coord = {}
    state_idx = 0

    for i in range(rows):
        for j in range(cols):
            if working_maze[i, j] == -1:
                state_idx += 1
            else:
                coord_to_state[(i, j)] = state_idx
                state_to_coord[state_idx] = (i, j)
                state_idx += 1

    # Initialize transition matrix
    T = np.zeros((N_STATES, N_STATES))

    # Possible moves: up, down, left, right
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # For each accessible state
    for state, (i, j) in state_to_coord.items():
        # Find accessible neighbors
        accessible_neighbors = []
        for di, dj in moves:
            ni, nj = i + di, j + dj
            # Check if neighbor is within bounds and accessible
            if 0 <= ni < rows and 0 <= nj < cols and working_maze[ni, nj] != -1:
                neighbor_state = coord_to_state[(ni, nj)]
                accessible_neighbors.append(neighbor_state)

        # Set uniform transition probabilities to accessible neighbors
        if accessible_neighbors:
            prob = 1.0 / len(accessible_neighbors)
            for neighbor in accessible_neighbors:
                T[state, neighbor] = prob

    # Terminal state transitions to itself
    T[TERM_IDX, TERM_IDX] = 1.0

    return T


def policy(model, D, state, V, T_maze, T_model, parameters):
    """
    Compute action probabilities for a given state.
    """
    const = 1e-10

    # Get successor states for this state in the current maze
    succ_states = np.where(T_maze[state] > 0)[0]
    # For non-terminal states, compute via successor representation
    terminals = np.zeros(N_STATES, dtype=bool)
    terminals[TERM_IDX] = True
    nonterminals = ~terminals
    P = T_model[nonterminals][:,terminals]

    if model == 'sr_is':
        gamma = parameters[0]
        lambda_ = parameters[1]
        beta = parameters[2]

        # Compute partition function
        t = np.zeros(N_STATES)
        t = np.full(N_STATES, lambda_ * np.log(gamma))
        t[terminals] = 1.0  # Terminal reward
        rmax = 1.0
        t_scaled = np.exp((t[terminals] - rmax) / lambda_)

        Z = np.zeros(N_STATES)
        # Z[nonterminals] = (gamma*D[nonterminals][:, TERM_IDX] * t_scaled)
        Z[nonterminals] = (gamma*D[nonterminals][:,nonterminals]) @ P @ t_scaled
        Z[terminals] = t_scaled
        Z[Z < const] = const

        log_Z = np.log(Z)
        log_Z += rmax / lambda_
        V_succ = log_Z[succ_states] * lambda_
        V_succ /= beta

    elif model == 'sr':
        beta = parameters[0]

        # V = D @ r, where r has reward only at terminal
        r = np.zeros(N_STATES)
        r[terminals] = 1.0
        V_full = D @ r
        V_succ = V_full[succ_states]
        V_succ /= beta

    elif model == 'mb':
        beta = parameters[0]
        V_succ = V[succ_states]
        V_succ /= beta

    # Softmax
    probs = np.zeros(N_STATES)
    exp_V = np.exp(V_succ - np.max(V_succ))
    probs[succ_states] = exp_V / (np.sum(exp_V) + const)

    return probs


def define_model_parameters(model, parameters=None):
    """
    Transform raw parameters to model-specific parameters.
    """
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    exp_func = lambda x: np.exp(x)

    if parameters is None:
        parameters = np.zeros(5)

    alpha = 0.0
    gamma = 0.0
    lambda_ = 0.0

    if model == 'sr_is':
        alpha = sigmoid(parameters[0])
        gamma = sigmoid(parameters[1])
        lambda_ = exp_func(parameters[2])
        beta = lambda_
        policy_params = [gamma, lambda_, beta]
        pnames = ['alpha', 'gamma', 'lambda']

    elif model == 'sr':
        alpha = sigmoid(parameters[0])
        gamma = sigmoid(parameters[1])
        beta = exp_func(parameters[2])
        policy_params = [beta]
        pnames = ['alpha', 'gamma', 'beta']

    elif model == 'mb':
        gamma = sigmoid(parameters[0])
        beta = exp_func(parameters[1])
        policy_params = [beta]
        pnames = ['gamma', 'beta']

    return pnames, alpha, gamma, lambda_, policy_params


def model_likelihood(parameters, data, model, mazes):
    """
    Compute log-likelihood for one subject across all maze configurations.
    """
    _, alpha, gamma, lambda_, policy_params = define_model_parameters(model, parameters)

    terminals = np.zeros(N_STATES, dtype=bool)
    terminals[TERM_IDX] = True

    one_hot = np.eye(N_STATES)
    T_open = create_transition_matrix_open()

    loglik = 0.0
    r = np.zeros(N_STATES)
    r[terminals] = 1

    # Loop through each maze configuration
    for configuration in data:
        maze_idx = configuration['maze']
        states_diff_starts = configuration['states']

        # Get transition matrix for this maze
        T_maze = maze_to_transition_matrix(mazes[maze_idx])
        # inialize M
        # M = np.eye(N_STATES)
        # M[~terminals, :] = 1    
        # np.fill_diagonal(M, (1e-10+1 - gamma)**-1)
        
        I = np.eye(N_STATES)
        M = np.linalg.inv(I - gamma * T_open + 1e-10 * I)
        L = np.diag(np.exp(-r / lambda_)) - T_open
        D = np.copy(M)
        T_model = np.copy(T_open)
        
        V = np.zeros(N_STATES)

        # Loop through different starting points in this maze
        for states in states_diff_starts:
            # Process consecutive state pairs
            for i in range(len(states) - 1):
                state = states[i] - 1  # Convert to 0-indexed
                next_state = states[i + 1] - 1
                
                T_model[state] = T_maze[state]
                Rep = D if model == 'sr_is' else M
                
                probs = policy(model, Rep, state, V, T_maze, T_model, policy_params)

                # Add log-likelihood
                loglik += np.log(probs[next_state] + 1e-10)

                # Update model
                if model == 'sr_is':
                    # Importance sampling weight
                    succ_states = np.where(T_maze[state] > 0)[0]
                    p_uniform = 1.0 / len(succ_states)
                    w = p_uniform / max(probs[next_state], 1e-2)

                    target = one_hot[state] + gamma * D[next_state]
                    D[state] = (1 - alpha) * D[state] + alpha * target * w

                elif model == 'sr':
                    target = one_hot[state] + gamma * M[next_state]
                    M[state] = (1 - alpha) * M[state] + alpha * target

                elif model == 'mb':
                    # Update transition knowledge (if state not seen in this maze yet)
                    # For MB, we do backward propagation after reaching terminal
                    if terminals[next_state]:
                        # Backward value iteration
                        V[TERM_IDX] = 1.0
                        max_iters = 100
                        tolerance = 1e-6

                        for iteration in range(max_iters):
                            V_old = V.copy()
                            for s in range(N_STATES):
                                if not terminals[s]:
                                    succ_s = np.where(T_model[s] > 0)[0]
                                    if len(succ_s) > 0:
                                        V[s] = gamma * np.max(V[succ_s])

                            # Check convergence
                            if np.max(np.abs(V - V_old)) < tolerance:
                                break

                # Break if reached terminal
                # if next_state == TERM_IDX:
                #     break

    return loglik


def define_prior(model):
    """
    Define prior mean and variance for model parameters.

    Parameters:
    -----------
    model : str
        Model name

    Returns:
    --------
    prior_mean : numpy array
        Prior mean
    prior_variance : numpy array
        Prior variance (diagonal)
    """
    pnames, _, _, _, _ = define_model_parameters(model, parameters=None)
    num_dim = len(pnames)
    prior_mean = np.zeros(num_dim)
    prior_variance = np.zeros(num_dim)

    for i in range(num_dim):
        if pnames[i].startswith('beta') or pnames[i].startswith('lambda'):
            prior_variance[i] = 100.0
        else:
            prior_variance[i] = 6.25

    return prior_mean, prior_variance


if __name__ == "__main__":
    # Configuration
    species = 'humans'  # or 'rats'
    model_names = ['sr_is', 'sr', 'mb']

    print(f"\n{'='*70}")
    print(f"Fitting De-Cothi {species} data")
    print(f"{'='*70}\n")

    # Load data
    data = extract_all_subjects_decothi(species)

    # Load mazes
    mazes = loadmat('data/de-cothi/mazes.mat')['mazes'][0]
    print(f"Loaded {len(data)} subjects")
    print(f"Loaded {len(mazes)} maze configurations\n")

    # Fit models
    lme = []
    cbms = []

    for model_name in model_names:
        np.random.seed(42)
        cbm_file = f'fit_decothi/{species}/{model_name}.pkl'

        print(f"\n{'='*70}")
        print(f"Fitting model: {model_name}")
        print(f"{'='*70}")

        prior_mean, prior_variance = define_prior(model_name)
        cfg = Config(d=len(prior_mean), num_init=1, num_init_med=3, num_init_up=5)

        # Create likelihood function with mazes bound
        mdl = lambda p, d: model_likelihood(p, d, model=model_name, mazes=mazes)

        if not os.path.exists(cbm_file):
            os.makedirs(os.path.dirname(cbm_file), exist_ok=True)
            cbm = individual_fit(data, mdl, prior_mean, prior_variance, cbm_file, cfg)

        # Load fitted results
        with open(cbm_file, 'rb') as f:
            cbm = pickle.load(f)

        cbms.append(cbm)
        lme.append(cbm.output.log_evidence)

        print(f"\nModel {model_name} fitted successfully")
        print(f"Mean log-evidence: {np.mean(cbm.output.log_evidence):.2f}")

    # Bayesian Model Selection
    print(f"\n{'='*70}")
    print("Bayesian Model Selection")
    print(f"{'='*70}\n")

    lme = np.column_stack(lme)  # subjects × models
    bms_result = bms(lme)

    print(f"Models: {model_names}")
    print(f"Expected model frequencies: {np.array2string(bms_result.model_frequency, suppress_small=True, precision=3)}")
    print(f"Protected exceedance probabilities: {np.array2string(bms_result.protected_exceedance_prob, suppress_small=True, precision=3)}")
    print(f"Bayes Omnibus Risk: {bms_result.bor:.3f}\n")
