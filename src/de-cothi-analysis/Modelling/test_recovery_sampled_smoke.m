% test_recovery_sampled_smoke.m
% Run from Modelling/ directory.
%
% 1. Prints sampled parameters for all 5 models x 2 species.
% 2. Runs ONE full simulation + fit (MF, humans, sim 1) to verify the pipeline.

clear; clc;

addpath('../Standard Functions (add to path)')
addpath('Model-free')
addpath('Model-based')
addpath('Hybrid SR+MB')
addpath('SR')

model_names  = {'MF', 'MB', 'SR', 'Hybrid', 'SR-IS'};
param_labels = {{'alpha','gamma'}, {'gamma'}, {'alpha','gamma'}, ...
                {'alpha','gamma','w_mix'}, {'alpha','gamma','lambda'}};
species_names = {'human', 'rat'};

% -------------------------------------------------------------------------
%  Print sampled parameters for all models x species
% -------------------------------------------------------------------------
fprintf('=== Sampled parameters (1 draw per model x species) ===\n\n');

for model_idx = 1:5
    for sp = 1:2
        rng_seed = model_idx * 1000 + sp * 100 + 1;
        [p_samp, p_med, p_sem] = sample_model_params(model_idx, sp, 1, rng_seed);
        labels = param_labels{model_idx};

        fprintf('%s | %s\n', model_names{model_idx}, species_names{sp});
        for k = 1:length(labels)
            fprintf('  %-8s  median=%.4f  sem=%.4f  sampled=%.4f\n', ...
                labels{k}, p_med(k), p_sem(k), p_samp(k));
        end
        fprintf('\n');
    end
end

% -------------------------------------------------------------------------
%  Pipeline smoke test: MF, humans, sim 1
%  (runs a short version — only 2 configs, 3 starts, 50 max steps)
% -------------------------------------------------------------------------
fprintf('=== Pipeline smoke test: MF human sim1 ===\n');

model_idx   = 1;
species_idx = 1;
sim_idx     = 1;
rng_seed    = model_idx * 1000 + species_idx * 100 + sim_idx;

[params_all, ~, ~] = sample_model_params(model_idx, species_idx, 1, rng_seed);
params = params_all(1,:);
alpha = params(1);   gamma = params(2);
fprintf('Params: alpha=%.4f  gamma=%.4f\n', alpha, gamma);

load('mazes.mat')
load('humans.mat')
load('Model-free/PreQ.mat')

goal_state = 34;
max_steps  = 50;    % short for smoke test
n_configs_test = 2;
n_starts_test  = 3;

Q = gamma .^ PreQ;   Q(isnan(Q)) = 0;
sim_dat = cell(1, 25, 10);   % keep full size so fit_all_models works

for config = 1:n_configs_test
    A_allowed = map2allowed(mazes{config});
    for start = 1:n_starts_test
        real_traj = humans{1, config, start};
        if isempty(real_traj); continue; end
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
        sim_dat{1, config, start} = traj;
    end
end
fprintf('Simulation done (%d configs x %d starts).\n', n_configs_test, n_starts_test);

fprintf('Fitting all models to smoke-test data...\n');
t0 = tic;
sim_ll_r = fit_all_models(sim_dat);
fprintf('Fitting done in %.1fs.\n', toc(t0));

n_params_vec = [2, 1, 2, 3, 3];
n_datapoints = 25 * 10;
BIC_r = -2*sim_ll_r + n_params_vec' .* log(n_datapoints);
[~, winner] = min(BIC_r);
model_names_all = {'MF','MB','SR','Hybrid','SR-IS'};
fprintf('LLs:    '); fprintf('%8.1f  ', sim_ll_r); fprintf('\n');
fprintf('BICs:   '); fprintf('%8.1f  ', BIC_r);    fprintf('\n');
fprintf('Winner: %s\n', model_names_all{winner});
fprintf('\nSmoke test passed.\n');
