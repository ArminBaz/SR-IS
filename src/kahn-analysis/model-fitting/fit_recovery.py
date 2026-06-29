import pickle
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from cbm.individual_fit import individual_fit, Config
from cbm.model_selection import bms

def policy(model, D, r, state, V, parameters):
    n_states = 11
    T = np.zeros((n_states, n_states))
    T[0, 1:3] = 0.5
    T[1, 3:5] = 0.5
    T[2, 5:7] = 0.5
    T[3, 7] = 1.0
    T[4, 8] = 1.0
    T[5, 9] = 1.0
    T[6, 10] = 1.0
    T[7:11, 7:11] = np.eye(4)
    terminals =  np.diag(T) == 1
    succ_states = np.where(T[state] > 0)[0]
    if model == 'linrl':
        gamma = parameters[0]            
        lambda_ =parameters[1]    
        beta = parameters[2]
        const = 1e-10
        
        D = gamma * np.linalg.inv(np.eye(n_states-4) - gamma * T[~terminals][:,~terminals])            

        t = np.zeros(n_states-4) # P @ np.exp(r[terminals] / lambda_)
        rmax = max(r[terminals])
        # tmax = np.exp( rmax / lambda_)
        t[3:] = np.exp( (r[terminals] - rmax) / lambda_)
        Z = np.zeros(n_states)
        Z[~terminals] = D @ t
        Z[terminals] = t[3:]
        Z[Z < const] = const  # prevent log(0)
        
        log_Z = np.log(Z)            
        log_Z += rmax / lambda_                   
        V = log_Z[succ_states] * lambda_

        V /= beta            
    
    if model == 'sr_is':
        gamma = parameters[0]
        lambda_ = parameters[1]
        beta = parameters[2]
        const = 1e-10        

        t = np.zeros(n_states-4) # P @ np.exp(r[terminals] / lambda_)
        rmax = max(r[terminals])
        # tmax = np.exp( rmax / lambda_)
        t[3:] = np.exp( (r[terminals] - rmax) / lambda_)
        Z = np.zeros(n_states)
        Z[~terminals] = (gamma * D[~terminals][:,~terminals]) @ t
        Z[terminals] = t[3:]
        Z[Z < const] = const  # prevent log(0)
        
        log_Z = np.log(Z)            
        log_Z += rmax / lambda_                   
        V = log_Z[succ_states] * lambda_

        V /= beta
    elif model == 'sr':
        beta = parameters[0]

        V = D @ r
        V = V[succ_states]
        V /= beta
    elif model == 'mb':
        beta = parameters[0]
        V = np.zeros(n_states)
        V[terminals] = r[terminals]
        V[3:7] = r[terminals]
        V[2] = np.max(V[5:7])
        V[1] = np.max(V[3:5])
        V[0] = np.max(V[1:3])
        V[1:3] = V[1:3]/beta
        V[3:7] = V[3:7]/beta
        V = V[succ_states]
    elif model == 'hybrid':
        beta1 = parameters[0]
        beta2 = parameters[1]
        V_sr = D @ r
        V_mb = np.zeros(n_states)
        V_mb[terminals] = r[terminals]
        V_mb[3:7] = r[terminals]
        V_mb[2] = np.max(V_mb[5:7])
        V_mb[1] = np.max(V_mb[3:5])
        V_mb[0] = np.max(V_mb[1:3])
        V_net = V_mb/beta1 + V_sr/beta2
        V = V_net[succ_states]      
    
    probs = np.zeros(n_states)
    exp_Q = np.exp(V - np.max(V))
    probs[succ_states] = exp_Q / (np.sum(exp_Q))
    return probs


def define_model_parameters(model, parameters=None):
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    exp_func = lambda x: np.exp(x)

    if parameters is None:
        parameters = np.zeros(5)

    alpha_r = 0.0
    alpha_D = 0.0
    gamma = 0.0
    if model == 'linrl':
        alpha_r = sigmoid(parameters[0])  
        gamma = sigmoid(parameters[1])        
        lambda_ = -1/np.log(gamma + 1e-10) # assume c=1, so lambda = -1/log(gamma)
        beta = lambda_
        policy_params = [gamma, lambda_, beta]
        pnames = ['alpha_r', 'gamma']      
    if model == 'sr_is':
        alpha_r = sigmoid(parameters[0])
        alpha_D = sigmoid(parameters[1])
        gamma = sigmoid(parameters[2])
        lambda_ = exp_func(parameters[3])
        beta = lambda_
        policy_params = [gamma, lambda_, beta]
        pnames = ['alpha_r', 'alpha_D', 'gamma', 'lambda']
    if model == 'sr':
        alpha_r = sigmoid(parameters[0])
        alpha_D = sigmoid(parameters[1])
        gamma = sigmoid(parameters[2])
        beta = exp_func(parameters[3])
        policy_params = [beta]
        pnames = ['alpha_r', 'alpha_D', 'gamma', 'beta']
    if model == 'hybrid':
        alpha_r = sigmoid(parameters[0])
        alpha_D = sigmoid(parameters[1])
        gamma = sigmoid(parameters[2])
        beta1 = exp_func(parameters[3])
        beta2 = exp_func(parameters[4])
        policy_params = [beta1, beta2]
        pnames = ['alpha_r', 'alpha_D', 'gamma', 'beta1', 'beta2']
    if model == 'mb':
        alpha_r = sigmoid(parameters[0])
        beta = exp_func(parameters[1])
        policy_params = [beta]
        pnames = ['alpha_r', 'beta']

    return pnames, alpha_r, alpha_D, gamma, policy_params


def model(parameters, data, model, simulate=False):
    _, alpha_r, alpha_D, gamma, policy_params = define_model_parameters(model, parameters)

    n_states = 11
    terminals = np.zeros(n_states, dtype=bool)
    terminals[7:] = True

    M = np.eye(n_states)
    M[~terminals, :] = 1
    np.fill_diagonal(M, (1e-10+1 - gamma)**-1)

    r = np.zeros(n_states)

    loglik = 0.0
    V = np.zeros(n_states)
    simulated_data = []

    for t in range(len(data)):
        if simulate:
            trial_type = data[t]['type']
            trial_rewards = data[t]['rewards']

            if trial_type == 'traversal':
                sim_states = [1]
                current = 0
            else:
                obs_boat = data[t]['obs_boat']
                current = obs_boat + 3
                sim_states = [current + 1]

            while not terminals[current]:
                probs = policy(model, M, r, current, V, policy_params)
                next_state = np.random.choice(n_states, p=probs)
                sim_states.append(next_state + 1)

                if model == 'sr_is':
                    w = 0.5 / max(probs[next_state], 1e-2)
                else:
                    w = 1
                target = np.eye(n_states)[current] + gamma * M[next_state]
                M[current] = (1 - alpha_D) * M[current] + alpha_D * target * w
                V[current] = (1 - alpha_r) * V[current] + alpha_r * gamma * V[next_state]
                current = next_state

            boat_idx = current - 7
            reward = int(trial_rewards[boat_idx])

            state = sim_states[-1] - 1
            M[state] = (1 - alpha_D) * M[state] + alpha_D * np.eye(n_states)[state]
            r[state] = (1 - alpha_r) * r[state] + alpha_r * reward

            simulated_data.append({'states': sim_states, 'reward': reward})
        else:
            states = data[t]['states']
            reward = data[t]['reward']
            for i in range(len(states)-1):
                state = states[i]-1
                next_state = states[i+1]-1

                probs = policy(model, M, r, state, V, policy_params)
                loglik += np.log(probs[next_state] + 1e-10)

                if model == 'sr_is':
                    w = 0.5/(max(probs[next_state], 1e-2))
                else:
                    w = 1
                target = np.eye(n_states)[state] + gamma * M[next_state]
                M[state] = (1 - alpha_D) * M[state] + alpha_D * target * w
                V[state] = (1 - alpha_r) * V[state] + alpha_r * gamma * V[next_state]

            state = states[-1]-1
            M[state] = (1 - alpha_D) * M[state] + alpha_D * np.eye(n_states)[state]
            r[state] = (1 - alpha_r) * r[state] + alpha_r * reward

    if simulate:
        return simulated_data
    return loglik


def define_prior(model):
    pnames, _, _, _, _ = define_model_parameters(model, parameters=None)
    num_dim = len(pnames)
    prior_mean = np.zeros(num_dim)
    prior_variance = np.zeros(num_dim)
    for i in range(num_dim):
        if pnames[i].startswith('beta'):
            prior_variance[i] = 100.0
        else:
            prior_variance[i] = 6.25
    return prior_mean, prior_variance


def generate_experiment_design():
    n_blocks = 22
    boat_probs = np.array([0.15, 0.85, 0.325, 0.675])

    transitions = ['congruent'] * 15 + ['within_island'] * 3 + ['policy_inconsistent'] * 3
    np.random.shuffle(transitions)

    design = []
    for b in range(n_blocks):
        n_pairs = np.random.randint(8, 13)  # draws from integers 8, 9, 10, 11, 12 (to match "22 reward blocks of between 8 to 12 trial pairs")
        for _ in range(n_pairs):
            design.append({'type': 'traversal',
                           'rewards': np.random.binomial(1, boat_probs)})
            design.append({'type': 'observation',
                           'obs_boat': np.random.randint(4),
                           'rewards': np.random.binomial(1, boat_probs)})

        if b < n_blocks - 1:
            transition = transitions[b]
            i1_best = 0 if boat_probs[0] > boat_probs[1] else 1
            i1_worst = 1 - i1_best
            i2_best = 2 if boat_probs[2] > boat_probs[3] else 3
            i2_worst = 5 - i2_best

            if transition == 'congruent':
                boat_probs[i1_best], boat_probs[i2_best] = boat_probs[i2_best], boat_probs[i1_best]
                boat_probs[i1_worst], boat_probs[i2_worst] = boat_probs[i2_worst], boat_probs[i1_worst]
            elif transition == 'within_island':
                boat_probs[0], boat_probs[1] = boat_probs[1], boat_probs[0]
                boat_probs[2], boat_probs[3] = boat_probs[3], boat_probs[2]
            else:
                boat_probs[i1_best], boat_probs[i2_worst] = boat_probs[i2_worst], boat_probs[i1_best]
                boat_probs[i1_worst], boat_probs[i2_best] = boat_probs[i2_best], boat_probs[i1_worst]

    return design


def model_recovery(data_str, n_subjects=100):

    if data_str == 'set1':
        model_names = ['sr_is', 'sr', 'mb', 'hybrid', 'linrl']
        ratios = [0.377, 0.048, 0.124, 0.211, 0.241]
    elif data_str == 'set2':
        model_names = ['sr_is', 'sr', 'mb', 'hybrid', 'linrl']
        ratios = [0.2, .2, .2, .2, .2]
    else:
        raise ValueError(f"Unknown data_str: {data_str}")

    save_dir = f'fits_new/recovery/{data_str}'
    os.makedirs(save_dir, exist_ok=True)
    data_file = os.path.join(save_dir, 'data.pkl')
    truth_file = os.path.join(save_dir, 'truth.pkl')

    if os.path.exists(data_file) and os.path.exists(truth_file):
        with open(data_file, 'rb') as f:
            simulated_subjects = pickle.load(f)
        with open(truth_file, 'rb') as f:
            truth = pickle.load(f)
        true_models = truth.get('models', [])
        true_params = truth.get('parameters', [])
        print(f"Loaded {len(simulated_subjects)} simulated subjects from {data_file}")
        return simulated_subjects, true_models, true_params

    np.random.seed(42)

    simulated_subjects = []
    true_models = []
    true_params = []

    for i, model_name in enumerate(model_names):
        n_sim = int(round(n_subjects * ratios[i]))
        if n_sim == 0:
            continue
        fit_file = f'fit_kahn/{model_name}.pkl'
        with open(fit_file, 'rb') as f:
            cbm = pickle.load(f)

        fitted_params = np.array(cbm.output.parameters)
        param_mean = fitted_params.mean(axis=0)
        param_std = fitted_params.std(axis=0)

        for j in range(n_sim):
            sampled = np.random.normal(param_mean, param_std)
            design = generate_experiment_design()
            sim = model(sampled, design, model_name, simulate=True)
            simulated_subjects.append(sim)
            true_models.append(model_name)
            true_params.append(sampled)

    with open(data_file, 'wb') as f:
        pickle.dump(simulated_subjects, f)
    with open(truth_file, 'wb') as f:
        pickle.dump({'models': true_models, 'parameters': true_params}, f)
    print(f"Saved {len(simulated_subjects)} simulated subjects to {data_file}")
    return simulated_subjects, true_models, true_params


def run_fitting(data, data_str):
    model_names = ['sr_is', 'sr', 'mb', 'hybrid', 'linrl']
    im = [0, 1, 2, 3, 4]
    fitted_model_names = [model_names[i] for i in im]
    lme = []
    for i in range(len(im)):
        i_model = im[i]
        np.random.seed(42)
        model_name = model_names[i_model]
        cbm_file = f'{data_str}/{model_name}.pkl'
        print(f"Model is: {model_name}")
        prior_mean, prior_variance = define_prior(model_name)
        cfg = Config(num_init=1, num_init_med=3, num_init_up=5)
        mdl = lambda p, d: model(p, d, model=model_name)
        if not os.path.exists(cbm_file):
            os.makedirs(os.path.dirname(cbm_file), exist_ok=True)
            cbm = individual_fit(data, mdl, prior_mean, prior_variance, cbm_file, cfg)
        with open(cbm_file, 'rb') as f:
            cbm = pickle.load(f)
        lme.append(cbm.output.log_evidence)

    lme = np.column_stack(lme)
    bms_result = bms(lme)
    print(f"\nExpected model frequencies: {np.array2string(bms_result.model_frequency, suppress_small=True, precision=3)}")
    print(f"Protected exceedance probabilities: {np.array2string(bms_result.protected_exceedance_prob, suppress_small=True, precision=3)}")
    return lme, fitted_model_names, bms_result

import matplotlib.pyplot as plt


def plot_confusion_matrix(data_str, lme, fitted_model_names, model_idx=None):
    truth_file = f'fits_new/recovery/{data_str}/truth.pkl'
    with open(truth_file, 'rb') as f:
        truth = pickle.load(f)
    true_models = truth['models']
    true_params = truth.get('parameters', None)

    if model_idx is None:
        selected_model_names = list(fitted_model_names)
        selected_lme = lme
    else:
        selected_model_names = [fitted_model_names[i] for i in model_idx]
        selected_lme = lme[:, model_idx]

    true_model_names = [m for m in list(dict.fromkeys(true_models)) if m in selected_model_names]

    true_mask = [m in selected_model_names for m in true_models]
    filtered_true_models = [m for m in true_models if m in selected_model_names]
    filtered_true_params = [p for p, m in zip(true_params, true_models) if true_params is not None and m in selected_model_names] if true_params is not None else None

    best_fit_idx = np.argmax(selected_lme[np.array(true_mask)], axis=1)
    best_fit_models = [selected_model_names[i] for i in best_fit_idx]

    n_true_models = len(true_model_names)
    n_fit_models = len(selected_model_names)
    confusion = np.zeros((n_true_models, n_fit_models))
    for true_m, fit_m in zip(filtered_true_models, best_fit_models):
        i = true_model_names.index(true_m)
        j = selected_model_names.index(fit_m)
        confusion[i, j] += 1

    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums,
                               out=np.zeros_like(confusion),
                               where=row_sums > 0)

    # Print mean generating parameters for each true model.
    if filtered_true_params is not None and len(filtered_true_params) == len(filtered_true_models):
        print("\nMean generating parameters by model:")
        for model_name in true_model_names:
            idx = [k for k, m in enumerate(filtered_true_models) if m == model_name]
            if len(idx) == 0:
                continue
            params_mat = np.vstack([np.asarray(filtered_true_params[k], dtype=float) for k in idx])
            mean_params = params_mat.mean(axis=0)
            pnames, alpha_r, alpha_D, gamma, policy_params = define_model_parameters(
                model_name, parameters=mean_params
            )

            transformed = {}
            if model_name == 'lrl':
                transformed = {
                    'c': policy_params[0],
                    'lambda': policy_params[1],
                    'beta': policy_params[2],
                }
            elif model_name == 'sr_is':
                transformed = {
                    'alpha_r': alpha_r,
                    'alpha_D': alpha_D,
                    'gamma': gamma,
                    'lambda': policy_params[1],
                    'beta': policy_params[2],
                }
            elif model_name == 'sr':
                transformed = {
                    'alpha_r': alpha_r,
                    'alpha_D': alpha_D,
                    'gamma': gamma,
                    'beta': policy_params[0],
                }
            elif model_name == 'mb':
                transformed = {
                    'alpha_r': alpha_r,
                    'beta': policy_params[0],
                }
            elif model_name == 'hybrid':
                transformed = {
                    'alpha_r': alpha_r,
                    'alpha_D': alpha_D,
                    'gamma': gamma,
                    'beta1': policy_params[0],
                    'beta2': policy_params[1],
                }
            else:
                transformed = {pnames[j]: mean_params[j] for j in range(min(len(pnames), len(mean_params)))}

            transformed_str = ", ".join(f"{k}={v:.3f}" for k, v in transformed.items())
            print(f"  {model_name}: {transformed_str}")
    else:
        print("\nMean generating parameters by model: unavailable (missing truth parameters).")

    # Print a text version of the normalized confusion matrix.
    row_label_width = max(8, max(len(m) for m in true_model_names))
    col_label_width = max(7, max(len(m) for m in selected_model_names) + 1)
    header = " " * (row_label_width + 6) + "Fitted"
    col_header = "True  " + " " * (row_label_width - 4) + "  " + "".join(
        name.rjust(col_label_width) for name in selected_model_names
    )
    print("\n" + header)
    print(col_header)
    for i, true_name in enumerate(true_model_names):
        row_values = "".join(f"{confusion_norm[i, j]:>{col_label_width}.2f}" for j in range(n_fit_models))
        print(f"      {true_name.ljust(row_label_width)}  {row_values}")

    fig_w = max(6, 1.1 * n_fit_models + 2)
    fig_h = max(4.5, 0.9 * n_true_models + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(confusion_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(n_fit_models))
    ax.set_yticks(range(n_true_models))
    ax.set_xticklabels(selected_model_names)
    ax.set_yticklabels(true_model_names)
    ax.set_xlabel('Fitted model')
    ax.set_ylabel('True (generating) model')
    ax.set_title(f'Model recovery: {data_str}')

    for i in range(n_true_models):
        for j in range(n_fit_models):
            if row_sums[i, 0] > 0:
                text = f'{confusion_norm[i, j]:.2f}\n({int(confusion[i, j])})'
                color = 'white' if confusion_norm[i, j] > 0.5 else 'black'
            else:
                text = '—'
                color = 'gray'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label='P(fit | true)')
    plt.tight_layout()

    out_path = f'fits_new/recovery/{data_str}/confusion_matrix.png'
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    data_str = 'set2'
    model_recovery(data_str, n_subjects=100)
    recovery_data_file = os.path.join('recovery', data_str, 'data.pkl')
    with open(recovery_data_file, 'rb') as f:
        recovery_data = pickle.load(f)
    lme, fitted_model_names, bms_result = run_fitting(recovery_data, f'recovery/{data_str}')

    plot_confusion_matrix(data_str, lme, fitted_model_names, model_idx=[0, 1, 2, 3, 4])
