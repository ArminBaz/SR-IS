function run_recovery_job(input)
% RUN_RECOVERY_JOB  Run model recovery for one participant (called by CARC job scripts).

model_idx   = input(1);
species_idx = input(2);
r           = input(3);

model_names = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};
model_name  = model_names{model_idx};
if species_idx == 1; species = 'human'; else; species = 'rat'; end

% Skip if checkpoint already exists
ckpt_file = sprintf('checkpoints/%s_%s_%d.mat', model_name, species, r);
if exist(ckpt_file, 'file')
    fprintf('Already done: %s %s %d\n', model_name, species, r);
    return;
end

fprintf('Starting: %s %s %d\n', model_name, species, r);

% -------------------------------------------------------------------------
%  Paths — SR/ last so its functions take precedence on shared names
% -------------------------------------------------------------------------
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

% -------------------------------------------------------------------------
%  Load shared data
% -------------------------------------------------------------------------
load('mazes.mat')      % mazes
load('train_T.mat')    % T

if species_idx == 1
    load('humans.mat'); dat = humans;
else
    tmp = load('rat.mat');  dat = tmp.rat;
end

if ~exist('checkpoints', 'dir'), mkdir('checkpoints'); end

% -------------------------------------------------------------------------
%  Load participant parameters for the generating model
% -------------------------------------------------------------------------
switch model_idx
    case 1  % MF
        load('Model-free/PreQ.mat')
        if species_idx == 1
            load('Model-free/human_params.mat');
        else
            load('Model-free/rat_params.mat');
        end

    case 2  % MB
        if species_idx == 1
            load('Model-based/human_params.mat');
        else
            load('Model-based/rat_params.mat');
        end

    case 3  % SR
        if species_idx == 1
            load('SR/human_params.mat');
        else
            load('SR/rat_params.mat');
        end

    case 4  % Hybrid
        if species_idx == 1
            load('Hybrid SR+MB/human_Hybrid_llik_V2.mat');
        else
            load('Hybrid SR+MB/rat_Hybrid_llik_V2.mat');
        end

    case 5  % SR-IS
        if species_idx == 1
            load('SR/human_SR_IS_llik.mat');
        else
            load('SR/rat_SR_IS_llik.mat');
        end
end
params = ppt_params;

% -------------------------------------------------------------------------
%  Simulate trajectories for participant r
% -------------------------------------------------------------------------
sim_dat = cell(1, n_configs, n_starts);

switch model_idx

    % ---- MF (matches Q_lik.m) -------------------------------------------
    case 1
        alpha = params(r, 1);
        gamma = params(r, 2);
        Q = gamma .^ PreQ;   Q(isnan(Q)) = 0;

        for config = 1:n_configs
            A_allowed = map2allowed(mazes{config});
            for start = 1:n_starts
                real_traj = dat{r, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    poss_a    = find(A_allowed(state_id,:) == 1);
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

    % ---- MB (matches MB_lik.m) -------------------------------------------
    case 2
        gamma     = params(r, 1);
        goal_map  = zeros(10);   goal_map(4,4) = 1;
        blief_map = zeros(10);

        for config = 1:n_configs
            A_allowed = map2allowed(mazes{config});
            for start = 1:n_starts
                real_traj = dat{r, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    blief_map    = update_bliefs(blief_map, state_id, mazes{config});
                    blief_sample = rand(10,10) < blief_map;
                    poss_a    = find(A_allowed(state_id,:) == 1);
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

    % ---- SR (matches SR_lik.m) -------------------------------------------
    case 3
        alpha = params(r, 1);
        gamma = params(r, 2);
        opt_M = inv(eye(100) - gamma*T);
        opt_w = zeros(100,1);   opt_w(goal_state) = 1;
        M = opt_M;   w = opt_w;

        for config = 1:n_configs
            A_allowed = map2allowed(mazes{config});
            for start = 1:n_starts
                real_traj = dat{r, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    V = M * w;
                    poss_a    = find(A_allowed(state_id,:) == 1);
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

    % ---- Hybrid (matches Hybrid_lik_V.m) ----------------------------------
    case 4
        alpha = params(r, 1);
        gamma = params(r, 2);
        w_mix = params(r, 3);
        opt_M = inv(eye(100) - gamma*T);
        opt_w = zeros(100,1);   opt_w(goal_state) = 1;
        M         = opt_M;
        sr_w      = opt_w;
        blief_map = zeros(10);

        for config = 1:n_configs
            A_allowed = map2allowed(mazes{config});
            for start = 1:n_starts
                real_traj = dat{r, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    V = M * sr_w;
                    [poss_next, V_SR] = SoftmaxLike_V(V, state_id, A_allowed);
                    blief_map = update_bliefs(blief_map, state_id, mazes{config});
                    [~, V_MB] = SoftmaxMB_V(state_id, blief_map, gamma, A_allowed);
                    V_hybrid   = w_mix * V_SR + (1-w_mix) * V_MB;
                    probs      = exp(V_hybrid) / sum(exp(V_hybrid));
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

    % ---- SR-IS (matches SR_IS_lik.m) -------------------------------------
    case 5
        alpha  = params(r, 1);
        gamma  = params(r, 2);
        lambda = params(r, 3);
        T_open    = create_transition_matrix_open();
        opt_M     = gamma * inv(eye(100) - gamma*T_open);
        reward_nt = lambda * log(gamma);
        opt_w     = reward_nt * ones(100,1);   opt_w(goal_state) = 1;
        rmax      = 0.0;
        reward_t  = exp((opt_w(goal_state) - rmax) / lambda);
        terminals    = goal_state;
        nonterminals = setdiff(1:100, terminals);
        invT         = 1 / lambda;
        const        = 0.0001;
        M = opt_M;   w = opt_w;

        for config = 1:n_configs
            T_maze    = maze_to_transition_matrix(mazes{config});
            A_allowed = map2allowed(mazes{config});
            P         = T_maze(nonterminals, terminals);
            for start = 1:n_starts
                real_traj = dat{r, config, start};
                if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                state_id = real_traj(1);   traj = state_id;
                for step = 1:max_steps
                    Z = zeros(100,1);
                    Z(nonterminals) = (gamma * M(nonterminals,nonterminals)) * P * reward_t;
                    Z(terminals)    = reward_t;
                    Z(Z < const)    = const;
                    logZ = log(Z) + rmax/lambda;
                    V    = logZ * lambda;
                    poss_a    = find(A_allowed(state_id,:) == 1);
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

% -------------------------------------------------------------------------
%  Fit all 5 models to simulated data
% -------------------------------------------------------------------------
sim_ll_r = fit_all_models(sim_dat);

% -------------------------------------------------------------------------
%  BIC and save checkpoint
% -------------------------------------------------------------------------
BIC_r    = -2*sim_ll_r + n_params_vec' .* log(n_datapoints);
[~, winner] = min(BIC_r);
fprintf('  %s %s %d --> Winner: %s\n', model_name, species, r, model_names{winner});
save(ckpt_file, 'sim_ll_r');
fprintf('Done: %s %s %d\n', model_name, species, r);
end
