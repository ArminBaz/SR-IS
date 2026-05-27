% model_recovery.m
%
% Model recovery analysis for all candidate models.

clear; clc; close all;

%% -------------------------------------------------------------------------
%  Paths — SR/ last so its functions take precedence on shared names
% -------------------------------------------------------------------------
addpath('../Standard Functions (add to path)')
addpath('Model-free')
addpath('Model-based')
addpath('Hybrid SR+MB')
addpath('SR')

%% -------------------------------------------------------------------------
%  Shared data
% -------------------------------------------------------------------------
load('mazes.mat')     % mazes  {25 x 1}
load('train_T.mat')   % T      [100 x 100] transition matrix
load('humans.mat')    % humans {18 x 25 x 10}
tmp = load('rat.mat'); rat = tmp.rat;   % rat {9 x 25 x 10}

n_humans = size(humans, 1);   % 18
n_rats   = size(rat,    1);   % 9

%% -------------------------------------------------------------------------
%  Checkpoint directory
% -------------------------------------------------------------------------
if ~exist('checkpoints', 'dir'), mkdir('checkpoints'); end

%% -------------------------------------------------------------------------
%  Constants
% -------------------------------------------------------------------------
n_configs  = 25;
n_starts   = 10;
goal_state = 34;
max_steps  = 500;

n_models     = 5;
model_names  = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};
n_params_vec = [2, 1, 2, 3, 3];
n_datapoints = n_configs * n_starts;   % 250

%  Optimisation settings, bounds, and model fitting live in fit_all_models.m

%% =========================================================================
%  MF MODEL RECOVERY
% =========================================================================
fprintf('\n========== MF MODEL RECOVERY ==========\n');

load('Model-free/PreQ.mat')
load('Model-free/human_params.mat'); mf_params_humans = ppt_params;
load('Model-free/rat_params.mat');   mf_params_rats   = ppt_params;

sim_ll_mf_humans = zeros(n_models, n_humans);
sim_ll_mf_rats   = zeros(n_models, n_rats);

for species_idx = 1:2
    if species_idx == 1
        species = 'human'; dat = humans; params = mf_params_humans; n_ppt = n_humans;
    else
        species = 'rat';   dat = rat;    params = mf_params_rats;   n_ppt = n_rats;
    end

    for r = 1:n_ppt
        ckpt_file = sprintf('checkpoints/MF_%s_%d.mat', species, r);
        if exist(ckpt_file, 'file')
            load(ckpt_file, 'sim_ll_r');
            fprintf('  %s %d / %d  (checkpoint)\n', species, r, n_ppt);
        else
            fprintf('  %s %d / %d\n', species, r, n_ppt);

            alpha = params(r, 1);
            gamma = params(r, 2);
            Q = gamma .^ PreQ;   Q(isnan(Q)) = 0;

            sim_dat = cell(1, n_configs, n_starts);
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

            sim_ll_r = fit_all_models(sim_dat);

            BIC_r = -2*sim_ll_r + n_params_vec'.*log(n_datapoints);
            [~, winner] = min(BIC_r);
            fprintf('    --> Winner: %s\n', model_names{winner});
            save(ckpt_file, 'sim_ll_r');
        end
        if species_idx == 1; sim_ll_mf_humans(:,r) = sim_ll_r;
        else;                 sim_ll_mf_rats(:,r)   = sim_ll_r; end
    end
end

BIC_mf_humans = -2*sim_ll_mf_humans + n_params_vec'.*log(n_datapoints);
BIC_mf_rats   = -2*sim_ll_mf_rats   + n_params_vec'.*log(n_datapoints);
[~, best_model_mf_humans] = min(BIC_mf_humans,[],1);
[~, best_model_mf_rats]   = min(BIC_mf_rats,  [],1);
save('MF_model_recovery_humans.mat','sim_ll_mf_humans','BIC_mf_humans','best_model_mf_humans','mf_params_humans');
save('MF_model_recovery_rats.mat',  'sim_ll_mf_rats',  'BIC_mf_rats',  'best_model_mf_rats',  'mf_params_rats');
fprintf('\n--- MF recovery: Humans ---\n');
counts = histcounts(best_model_mf_humans, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_humans,100*counts(m)/n_humans); end
fprintf('\n--- MF recovery: Rats ---\n');
counts = histcounts(best_model_mf_rats, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_rats,100*counts(m)/n_rats); end

%% =========================================================================
%  MB MODEL RECOVERY
% =========================================================================
fprintf('\n========== MB MODEL RECOVERY ==========\n');

load('Model-based/human_params.mat'); mb_params_humans = ppt_params;
load('Model-based/rat_params.mat');   mb_params_rats   = ppt_params;

sim_ll_mb_humans = zeros(n_models, n_humans);
sim_ll_mb_rats   = zeros(n_models, n_rats);

for species_idx = 1:2
    if species_idx == 1
        species = 'human'; dat = humans; params = mb_params_humans; n_ppt = n_humans;
    else
        species = 'rat';   dat = rat;    params = mb_params_rats;   n_ppt = n_rats;
    end

    for r = 1:n_ppt
        ckpt_file = sprintf('checkpoints/MB_%s_%d.mat', species, r);
        if exist(ckpt_file, 'file')
            load(ckpt_file, 'sim_ll_r');
            fprintf('  %s %d / %d  (checkpoint)\n', species, r, n_ppt);
        else
            fprintf('  %s %d / %d\n', species, r, n_ppt);

            gamma     = params(r, 1);
            goal_map  = zeros(10);   goal_map(4,4) = 1;
            blief_map = zeros(10);   % persists across configs and starts

            sim_dat = cell(1, n_configs, n_starts);
            for config = 1:n_configs
                A_allowed = map2allowed(mazes{config});
                for start = 1:n_starts
                    real_traj = dat{r, config, start};
                    if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                    state_id = real_traj(1);   traj = state_id;
                    for step = 1:max_steps
                        % Update beliefs (matches MB_lik step order)
                        blief_map = update_bliefs(blief_map, state_id, mazes{config});

                        % Compute MB values via A* (mirrors SoftmaxMB_V)
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
                        probs = exp(V_mb) / sum(exp(V_mb));   % invT=1
                        chosen_idx = randsample(length(poss_next), 1, true, probs);
                        next_state = poss_next(chosen_idx);
                        traj = [traj, next_state];
                        state_id = next_state;
                        if state_id == goal_state; break; end
                    end
                    sim_dat{1,config,start} = traj;
                end
            end

            sim_ll_r = fit_all_models(sim_dat);

            BIC_r = -2*sim_ll_r + n_params_vec'.*log(n_datapoints);
            [~, winner] = min(BIC_r);
            fprintf('    --> Winner: %s\n', model_names{winner});
            save(ckpt_file, 'sim_ll_r');
        end
        if species_idx == 1; sim_ll_mb_humans(:,r) = sim_ll_r;
        else;                 sim_ll_mb_rats(:,r)   = sim_ll_r; end
    end
end

BIC_mb_humans = -2*sim_ll_mb_humans + n_params_vec'.*log(n_datapoints);
BIC_mb_rats   = -2*sim_ll_mb_rats   + n_params_vec'.*log(n_datapoints);
[~, best_model_mb_humans] = min(BIC_mb_humans,[],1);
[~, best_model_mb_rats]   = min(BIC_mb_rats,  [],1);
save('MB_model_recovery_humans.mat','sim_ll_mb_humans','BIC_mb_humans','best_model_mb_humans','mb_params_humans');
save('MB_model_recovery_rats.mat',  'sim_ll_mb_rats',  'BIC_mb_rats',  'best_model_mb_rats',  'mb_params_rats');
fprintf('\n--- MB recovery: Humans ---\n');
counts = histcounts(best_model_mb_humans, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_humans,100*counts(m)/n_humans); end
fprintf('\n--- MB recovery: Rats ---\n');
counts = histcounts(best_model_mb_rats, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_rats,100*counts(m)/n_rats); end

%% =========================================================================
%  SR MODEL RECOVERY
% =========================================================================
fprintf('\n========== SR MODEL RECOVERY ==========\n');

load('SR/human_params.mat'); sr_params_humans = ppt_params;
load('SR/rat_params.mat');   sr_params_rats   = ppt_params;

sim_ll_sr_humans = zeros(n_models, n_humans);
sim_ll_sr_rats   = zeros(n_models, n_rats);

for species_idx = 1:2
    if species_idx == 1
        species = 'human'; dat = humans; params = sr_params_humans; n_ppt = n_humans;
    else
        species = 'rat';   dat = rat;    params = sr_params_rats;   n_ppt = n_rats;
    end

    for r = 1:n_ppt
        ckpt_file = sprintf('checkpoints/SR_%s_%d.mat', species, r);
        if exist(ckpt_file, 'file')
            load(ckpt_file, 'sim_ll_r');
            fprintf('  %s %d / %d  (checkpoint)\n', species, r, n_ppt);
        else
            fprintf('  %s %d / %d\n', species, r, n_ppt);

            alpha = params(r, 1);
            gamma = params(r, 2);
            opt_M = inv(eye(100) - gamma*T);
            opt_w = zeros(100,1);   opt_w(goal_state) = 1;
            M = opt_M;   w = opt_w;

            sim_dat = cell(1, n_configs, n_starts);
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
                        % invT=1 (matches SR_lik.m SoftmaxLike 4-arg call)
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

            sim_ll_r = fit_all_models(sim_dat);

            BIC_r = -2*sim_ll_r + n_params_vec'.*log(n_datapoints);
            [~, winner] = min(BIC_r);
            fprintf('    --> Winner: %s\n', model_names{winner});
            save(ckpt_file, 'sim_ll_r');
        end
        if species_idx == 1; sim_ll_sr_humans(:,r) = sim_ll_r;
        else;                 sim_ll_sr_rats(:,r)   = sim_ll_r; end
    end
end

BIC_sr_humans = -2*sim_ll_sr_humans + n_params_vec'.*log(n_datapoints);
BIC_sr_rats   = -2*sim_ll_sr_rats   + n_params_vec'.*log(n_datapoints);
[~, best_model_sr_humans] = min(BIC_sr_humans,[],1);
[~, best_model_sr_rats]   = min(BIC_sr_rats,  [],1);
save('SR_model_recovery_humans.mat','sim_ll_sr_humans','BIC_sr_humans','best_model_sr_humans','sr_params_humans');
save('SR_model_recovery_rats.mat',  'sim_ll_sr_rats',  'BIC_sr_rats',  'best_model_sr_rats',  'sr_params_rats');
fprintf('\n--- SR recovery: Humans ---\n');
counts = histcounts(best_model_sr_humans, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_humans,100*counts(m)/n_humans); end
fprintf('\n--- SR recovery: Rats ---\n');
counts = histcounts(best_model_sr_rats, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_rats,100*counts(m)/n_rats); end

%% =========================================================================
%  HYBRID MODEL RECOVERY
% =========================================================================
fprintf('\n========== HYBRID MODEL RECOVERY ==========\n');

load('Hybrid SR+MB/human_Hybrid_llik_V2.mat'); hyb_params_humans = ppt_params;
load('Hybrid SR+MB/rat_Hybrid_llik_V2.mat');   hyb_params_rats   = ppt_params;

sim_ll_hyb_humans = zeros(n_models, n_humans);
sim_ll_hyb_rats   = zeros(n_models, n_rats);

for species_idx = 1:2
    if species_idx == 1
        species = 'human'; dat = humans; params = hyb_params_humans; n_ppt = n_humans;
    else
        species = 'rat';   dat = rat;    params = hyb_params_rats;   n_ppt = n_rats;
    end

    for r = 1:n_ppt
        ckpt_file = sprintf('checkpoints/Hybrid_%s_%d.mat', species, r);
        if exist(ckpt_file, 'file')
            load(ckpt_file, 'sim_ll_r');
            fprintf('  %s %d / %d  (checkpoint)\n', species, r, n_ppt);
        else
            fprintf('  %s %d / %d\n', species, r, n_ppt);

            alpha = params(r, 1);
            gamma = params(r, 2);
            w_mix = params(r, 3);
            opt_M = inv(eye(100) - gamma*T);
            opt_w = zeros(100,1);   opt_w(goal_state) = 1;
            M         = opt_M;
            sr_w      = opt_w;
            blief_map = zeros(10);   % persists across configs and starts

            sim_dat = cell(1, n_configs, n_starts);
            for config = 1:n_configs
                A_allowed = map2allowed(mazes{config});
                for start = 1:n_starts
                    real_traj = dat{r, config, start};
                    if isempty(real_traj); sim_dat{1,config,start} = []; continue; end
                    state_id = real_traj(1);   traj = state_id;
                    for step = 1:max_steps
                        % SR values
                        V = M * sr_w;
                        [poss_next, V_SR] = SoftmaxLike_V(V, state_id, A_allowed);

                        % MB values (includes stochastic belief sampling)
                        blief_map = update_bliefs(blief_map, state_id, mazes{config});
                        [~, V_MB] = SoftmaxMB_V(state_id, blief_map, gamma, A_allowed);

                        % Hybrid: mix and sample (invT=1, mirrors SoftmaxHybrid)
                        V_hybrid = w_mix * V_SR + (1-w_mix) * V_MB;
                        probs    = exp(V_hybrid) / sum(exp(V_hybrid));
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

            sim_ll_r = fit_all_models(sim_dat);

            BIC_r = -2*sim_ll_r + n_params_vec'.*log(n_datapoints);
            [~, winner] = min(BIC_r);
            fprintf('    --> Winner: %s\n', model_names{winner});
            save(ckpt_file, 'sim_ll_r');
        end
        if species_idx == 1; sim_ll_hyb_humans(:,r) = sim_ll_r;
        else;                 sim_ll_hyb_rats(:,r)   = sim_ll_r; end
    end
end

BIC_hyb_humans = -2*sim_ll_hyb_humans + n_params_vec'.*log(n_datapoints);
BIC_hyb_rats   = -2*sim_ll_hyb_rats   + n_params_vec'.*log(n_datapoints);
[~, best_model_hyb_humans] = min(BIC_hyb_humans,[],1);
[~, best_model_hyb_rats]   = min(BIC_hyb_rats,  [],1);
save('Hybrid_model_recovery_humans.mat','sim_ll_hyb_humans','BIC_hyb_humans','best_model_hyb_humans','hyb_params_humans');
save('Hybrid_model_recovery_rats.mat',  'sim_ll_hyb_rats',  'BIC_hyb_rats',  'best_model_hyb_rats',  'hyb_params_rats');
fprintf('\n--- Hybrid recovery: Humans ---\n');
counts = histcounts(best_model_hyb_humans, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_humans,100*counts(m)/n_humans); end
fprintf('\n--- Hybrid recovery: Rats ---\n');
counts = histcounts(best_model_hyb_rats, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_rats,100*counts(m)/n_rats); end

%% =========================================================================
%  SR-IS MODEL RECOVERY
% =========================================================================
fprintf('\n========== SR-IS MODEL RECOVERY ==========\n');

load('SR/human_SR_IS_llik.mat'); sris_params_humans = ppt_params;
load('SR/rat_SR_IS_llik.mat');   sris_params_rats   = ppt_params;

sim_ll_sris_humans = zeros(n_models, n_humans);
sim_ll_sris_rats   = zeros(n_models, n_rats);

const = 0.0001;

for species_idx = 1:2
    if species_idx == 1
        species = 'human'; dat = humans; params = sris_params_humans; n_ppt = n_humans;
    else
        species = 'rat';   dat = rat;    params = sris_params_rats;   n_ppt = n_rats;
    end

    for r = 1:n_ppt
        ckpt_file = sprintf('checkpoints/SR-IS_%s_%d.mat', species, r);
        if exist(ckpt_file, 'file')
            load(ckpt_file, 'sim_ll_r');
            fprintf('  %s %d / %d  (checkpoint)\n', species, r, n_ppt);
        else
            fprintf('  %s %d / %d\n', species, r, n_ppt);

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
            invT      = 1 / lambda;

            sim_dat = cell(1, n_configs, n_starts);
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
                        % Soft-max value function (matches SR_IS_lik exactly)
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
                        M    = TD_update_imp(M, state_id, next_state, gamma, alpha, imp_samp);
                        state_id = next_state;
                        if state_id == goal_state; break; end
                    end
                    sim_dat{1,config,start} = traj;
                end
            end

            sim_ll_r = fit_all_models(sim_dat);

            BIC_r = -2*sim_ll_r + n_params_vec'.*log(n_datapoints);
            [~, winner] = min(BIC_r);
            fprintf('    --> Winner: %s\n', model_names{winner});
            save(ckpt_file, 'sim_ll_r');
        end
        if species_idx == 1; sim_ll_sris_humans(:,r) = sim_ll_r;
        else;                 sim_ll_sris_rats(:,r)   = sim_ll_r; end
    end
end

BIC_sris_humans = -2*sim_ll_sris_humans + n_params_vec'.*log(n_datapoints);
BIC_sris_rats   = -2*sim_ll_sris_rats   + n_params_vec'.*log(n_datapoints);
[~, best_model_sris_humans] = min(BIC_sris_humans,[],1);
[~, best_model_sris_rats]   = min(BIC_sris_rats,  [],1);
save('SR-IS_model_recovery_humans.mat','sim_ll_sris_humans','BIC_sris_humans','best_model_sris_humans','sris_params_humans');
save('SR-IS_model_recovery_rats.mat',  'sim_ll_sris_rats',  'BIC_sris_rats',  'best_model_sris_rats',  'sris_params_rats');
fprintf('\n--- SR-IS recovery: Humans ---\n');
counts = histcounts(best_model_sris_humans, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_humans,100*counts(m)/n_humans); end
fprintf('\n--- SR-IS recovery: Rats ---\n');
counts = histcounts(best_model_sris_rats, 0.5:(n_models+0.5));
for m=1:n_models; fprintf('  %-8s: %2d/%2d (%.0f%%)\n',model_names{m},counts(m),n_rats,100*counts(m)/n_rats); end
