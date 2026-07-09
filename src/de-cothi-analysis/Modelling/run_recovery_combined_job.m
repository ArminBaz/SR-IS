function run_recovery_combined_job(input)
% RUN_RECOVERY_COMBINED_JOB  Model recovery with combined human+rat parameter sampling.
%
%   input = [model_idx, sim_idx]
%   model_idx : 1=MF  2=MB  3=SR  4=Hybrid  5=SR-IS  (generating model)
%   sim_idx   : 1:20  (which sample — unique RNG seed per job)
%
%   Checkpoints saved to checkpoints_combined/{model}_sim{k}.mat

model_idx = input(1);
sim_idx   = input(2);

model_names = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};
model_name  = model_names{model_idx};

if ~exist('checkpoints_combined', 'dir'), mkdir('checkpoints_combined'); end

ckpt_file = sprintf('checkpoints_combined/%s_sim%d.mat', model_name, sim_idx);
if exist(ckpt_file, 'file')
    fprintf('Already done: %s sim%d\n', model_name, sim_idx);
    return;
end

fprintf('Starting: %s sim%d\n', model_name, sim_idx);

addpath('../Standard Functions (add to path)')
addpath('Model-free')
addpath('Model-based')
addpath('Hybrid SR+MB')
addpath('SR')

% -------------------------------------------------------------------------
%  Constants
% -------------------------------------------------------------------------
n_configs    = 25;
n_starts     = 10;
goal_state   = 34;
max_steps    = 500;
n_models     = 5;
n_params_vec = [2, 1, 2, 3, 3];
n_datapoints = n_configs * n_starts;

load('mazes.mat')
load('train_T.mat')

% -------------------------------------------------------------------------
%  Sample parameters (combined human+rat distribution)
% -------------------------------------------------------------------------
rng_seed = model_idx * 1000 + sim_idx;
[params_all, param_mean, param_std] = sample_model_params_combined(model_idx, 1, rng_seed);
params = params_all(1,:);

fprintf('  Sampled params: ');
fprintf('%.4f  ', params);
fprintf('\n  (mean: ');
fprintf('%.4f  ', param_mean);
fprintf('| std: ');
fprintf('%.4f  ', param_std);
fprintf(')\n');

% -------------------------------------------------------------------------
%  Simulate trajectories (use human participant 1 for start states)
% -------------------------------------------------------------------------
sim_dat = cell(1, n_configs, n_starts);

load('humans.mat'); dat_ref = humans;

switch model_idx

    % ---- MF ---------------------------------------------------------------
    case 1
        alpha = params(1);
        gamma = params(2);
        load('Model-free/PreQ.mat')
        Q = gamma .^ PreQ;   Q(isnan(Q)) = 0;

        for config = 1:n_configs
            A_allowed = map2allowed(mazes{config});
            for start = 1:n_starts
                real_traj = dat_ref{1, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    poss_a = find(A_allowed(state_id,:) == 1);
                    poss_next = zeros(1, length(poss_a));
                    for i = 1:length(poss_a)
                        [poss_next(i), ~] = action2state(state_id, poss_a(i), A_allowed);
                    end
                    vals   = max(Q(poss_next,:), [], 2);
                    logits = vals - max(vals);
                    probs  = exp(logits) / sum(exp(logits));
                    chosen_idx = randsample(length(poss_next), 1, true, probs);
                    next_state = poss_next(chosen_idx);
                    a_id       = poss_a(chosen_idx);
                    [~, reward] = action2state(state_id, a_id, A_allowed);
                    traj = [traj, next_state];
                    Q    = Q_update(Q, state_id, a_id, reward, next_state, gamma, alpha);
                    state_id = next_state;
                    if state_id == goal_state; break; end
                end
                sim_dat{1,config,start} = traj;
            end
        end

    % ---- MB ---------------------------------------------------------------
    case 2
        gamma     = params(1);
        goal_map  = zeros(10);   goal_map(4,4) = 1;
        blief_map = zeros(10);

        for config = 1:n_configs
            A_allowed = map2allowed(mazes{config});
            for start = 1:n_starts
                real_traj = dat_ref{1, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    blief_map    = update_bliefs(blief_map, state_id, mazes{config});
                    blief_sample = rand(10,10) < blief_map;
                    poss_a = find(A_allowed(state_id,:) == 1);
                    poss_next = zeros(1, length(poss_a));
                    V_mb      = zeros(1, length(poss_a));
                    for i = 1:length(poss_a)
                        [poss_next(i), ~] = action2state(state_id, poss_a(i), A_allowed);
                        if poss_next(i) == goal_state
                            V_mb(i) = 1;
                        else
                            [x_coord, y_coord] = state2coords(poss_next(i));
                            bmap_tmp = blief_sample;
                            bmap_tmp(y_coord, x_coord) = 0;
                            d = ASTARPATH(x_coord, y_coord, bmap_tmp, goal_map);
                            V_mb(i) = gamma^d * (d > 0);
                        end
                    end
                    probs      = exp(V_mb) / sum(exp(V_mb));
                    chosen_idx = randsample(length(poss_next), 1, true, probs);
                    next_state = poss_next(chosen_idx);
                    traj = [traj, next_state];
                    state_id = next_state;
                    if state_id == goal_state; break; end
                end
                sim_dat{1,config,start} = traj;
            end
        end

    % ---- SR ---------------------------------------------------------------
    case 3
        alpha  = params(1);
        gamma  = params(2);
        opt_M  = inv(eye(100) - gamma*T);
        opt_w  = zeros(100,1);   opt_w(goal_state) = 1;
        M = opt_M;   w = opt_w;

        for config = 1:n_configs
            A_allowed = map2allowed(mazes{config});
            for start = 1:n_starts
                real_traj = dat_ref{1, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    V = M * w;
                    poss_a = find(A_allowed(state_id,:) == 1);
                    poss_next = zeros(1, length(poss_a));
                    for i = 1:length(poss_a)
                        [poss_next(i), ~] = action2state(state_id, poss_a(i), A_allowed);
                    end
                    logits = V(poss_next);
                    logits = logits - max(logits);
                    probs  = exp(logits) / sum(exp(logits));
                    chosen_idx = randsample(length(poss_next), 1, true, probs);
                    next_state = poss_next(chosen_idx);
                    traj = [traj, next_state];
                    M    = TD_update(M, state_id, next_state, gamma, alpha);
                    state_id = next_state;
                    if state_id == goal_state; break; end
                end
                sim_dat{1,config,start} = traj;
            end
        end

    % ---- Hybrid -----------------------------------------------------------
    case 4
        alpha    = params(1);
        gamma    = params(2);
        w_mix    = params(3);
        opt_M  = inv(eye(100) - gamma*T);
        opt_w  = zeros(100,1);   opt_w(goal_state) = 1;
        M         = opt_M;
        sr_w      = opt_w;
        blief_map = zeros(10);

        for config = 1:n_configs
            A_allowed = map2allowed(mazes{config});
            for start = 1:n_starts
                real_traj = dat_ref{1, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    V = M * sr_w;
                    [poss_next, V_SR] = SoftmaxLike_V(V, state_id, A_allowed);
                    blief_map = update_bliefs(blief_map, state_id, mazes{config});
                    [~, V_MB] = SoftmaxMB_V(state_id, blief_map, gamma, A_allowed);
                    V_hybrid   = w_mix * V_SR + (1-w_mix) * V_MB;
                    probs      = exp(V_hybrid - max(V_hybrid)) / sum(exp(V_hybrid - max(V_hybrid)));
                    chosen_idx = randsample(length(poss_next), 1, true, probs);
                    next_state = poss_next(chosen_idx);
                    traj = [traj, next_state];
                    M    = TD_update(M, state_id, next_state, gamma, alpha);
                    state_id = next_state;
                    if state_id == goal_state; break; end
                end
                sim_dat{1,config,start} = traj;
            end
        end

    % ---- SR-IS ------------------------------------------------------------
    case 5
        alpha  = params(1);
        gamma  = params(2);
        lambda = params(3);
        const        = 0.0001;
        T_open       = create_transition_matrix_open();
        opt_M        = gamma * inv(eye(100) - gamma*T_open);
        reward_nt    = lambda * log(gamma);
        opt_w        = reward_nt * ones(100,1);   opt_w(goal_state) = 1;
        rmax         = 0.0;
        reward_t     = exp((opt_w(goal_state) - rmax) / lambda);
        terminals    = goal_state;
        nonterminals = setdiff(1:100, terminals);
        invT         = 1 / lambda;
        M = opt_M;

        for config = 1:n_configs
            T_maze    = maze_to_transition_matrix(mazes{config});
            A_allowed = map2allowed(mazes{config});
            P         = T_maze(nonterminals, terminals);
            for start = 1:n_starts
                real_traj = dat_ref{1, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    Z = zeros(100,1);
                    Z(nonterminals) = (gamma * M(nonterminals,nonterminals)) * P * reward_t;
                    Z(terminals)    = reward_t;
                    Z(Z < const)    = const;
                    logZ = log(Z) + rmax/lambda;
                    V    = logZ * lambda;
                    poss_a = find(A_allowed(state_id,:) == 1);
                    poss_next = zeros(1, length(poss_a));
                    for i = 1:length(poss_a)
                        [poss_next(i), ~] = action2state(state_id, poss_a(i), A_allowed);
                    end
                    logits = invT * V(poss_next);
                    logits = logits - max(logits);
                    probs  = exp(logits) / sum(exp(logits));
                    chosen_idx = randsample(length(poss_next), 1, true, probs);
                    next_state = poss_next(chosen_idx);
                    a_id       = poss_a(chosen_idx);
                    p_chosen   = probs(chosen_idx);
                    imp_samp   = ImportanceSampling(p_chosen, state_id, A_allowed, 10);
                    traj = [traj, next_state];
                    M    = TD_update_imp_stable(M, state_id, next_state, gamma, alpha, imp_samp);
                    state_id = next_state;
                    if state_id == goal_state; break; end
                end
                sim_dat{1,config,start} = traj;
            end
        end
end

fprintf('  Simulation done.\n');

% -------------------------------------------------------------------------
%  Fit all 5 models and save checkpoint
% -------------------------------------------------------------------------
sim_ll_r = fit_all_models(sim_dat);
BIC_r    = -2*sim_ll_r + n_params_vec' .* log(n_datapoints);
[~, winner] = min(BIC_r);
fprintf('  %s sim%d --> Winner: %s\n', model_name, sim_idx, model_names{winner});

save(ckpt_file, 'sim_ll_r', 'params');
fprintf('Done: %s sim%d\n', model_name, sim_idx);
end
