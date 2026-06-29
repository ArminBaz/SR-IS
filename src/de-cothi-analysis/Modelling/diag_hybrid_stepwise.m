% diag_hybrid_stepwise.m
% Run from Modelling/ directory.
%
% Steps through human participant 1's trajectories trial by trial.
% At each decision prints V_SR and V_MB for every available action,
% marks the chosen action, and shows P(chosen) under SR-only vs MB-only.
% Accumulates per-step log-likelihood difference to show where SR beats MB.

clear; clc;

addpath('../Standard Functions (add to path)')
addpath('Model-free')
addpath('Model-based')
addpath('Hybrid SR+MB')
addpath('SR')

load('mazes.mat')
load('train_T.mat')
load('humans.mat')
load('Hybrid SR+MB/human_Hybrid_llik_V2.mat')

r         = 1;
alpha     = ppt_params(r, 1);
gamma     = ppt_params(r, 2);
goal_state = 34;

fprintf('Human %d  alpha=%.4f  gamma=%.4f\n\n', r, alpha, gamma);

opt_M     = inv(eye(100) - gamma*T);
opt_w     = zeros(100,1);   opt_w(goal_state) = 1;
M         = opt_M;
sr_w      = opt_w;
blief_map = zeros(10);

% Accumulators
ll_SR = 0;   ll_MB = 0;
n_steps = 0;

% Per-step delta log-likelihood (SR - MB): positive = SR better that step
delta_ll  = [];
step_info = struct('config',{},'start',{},'step',{},...
                   'state',{},'chosen',{},...
                   'V_SR',{},'V_MB',{},...
                   'p_SR',{},'p_MB',{},'delta',{});

% Only print first N_PRINT steps to keep output manageable
N_PRINT = 60;
n_printed = 0;

% Accumulators: raw and normalised
ll_SR_raw = 0;   ll_MB_raw = 0;
ll_SR_n   = 0;   ll_MB_n   = 0;
delta_ll_raw = [];
delta_ll_n   = [];

for config = 1:25
    A_allowed = map2allowed(mazes{config});
    for start = 1:10
        traj = humans{r, config, start};
        if isempty(traj); continue; end
        state_id = traj(1);

        for k = 1:(length(traj)-1)
            next_state_id = traj(1+k);

            % --- SR values (raw and normalised by global max) ---
            V        = M * sr_w;
            max_V    = max(V);
            poss_a   = find(A_allowed(state_id,:) == 1);
            poss_next = zeros(1, length(poss_a));
            for i = 1:length(poss_a)
                [poss_next(i), ~] = action2state(state_id, poss_a(i), A_allowed);
            end
            V_SR     = V(poss_next)';
            V_SR_n   = V_SR / max(max_V, 1e-8);   % normalised

            % --- MB values ---
            blief_map = update_bliefs(blief_map, state_id, mazes{config});
            [~, V_MB] = SoftmaxMB_V(state_id, blief_map, gamma, A_allowed);
            V_MB      = V_MB(:);

            % --- Softmax probabilities: raw ---
            probs_SR_raw = exp(V_SR   - max(V_SR))   / sum(exp(V_SR   - max(V_SR)));
            probs_MB_raw = exp(V_MB   - max(V_MB))   / sum(exp(V_MB   - max(V_MB)));

            % --- Softmax probabilities: normalised SR, raw MB ---
            probs_SR_norm = exp(V_SR_n - max(V_SR_n)) / sum(exp(V_SR_n - max(V_SR_n)));
            probs_MB_norm = probs_MB_raw;   % MB unchanged

            % --- Chosen action ---
            chosen_idx = find(poss_next == next_state_id, 1);

            p_SR_raw = probs_SR_raw(chosen_idx);
            p_MB_raw = probs_MB_raw(chosen_idx);
            p_SR_nm  = probs_SR_norm(chosen_idx);
            p_MB_nm  = probs_MB_norm(chosen_idx);

            d_raw = log(max(p_SR_raw,1e-12)) - log(max(p_MB_raw,1e-12));
            d_n   = log(max(p_SR_nm, 1e-12)) - log(max(p_MB_nm, 1e-12));

            ll_SR_raw = ll_SR_raw + log(max(p_SR_raw,1e-12));
            ll_MB_raw = ll_MB_raw + log(max(p_MB_raw,1e-12));
            ll_SR_n   = ll_SR_n   + log(max(p_SR_nm, 1e-12));
            ll_MB_n   = ll_MB_n   + log(max(p_MB_nm, 1e-12));
            n_steps = n_steps + 1;

            delta_ll_raw(end+1) = d_raw;
            delta_ll_n(end+1)   = d_n;

            % --- Print ---
            if n_printed < N_PRINT
                fprintf('Config %2d | Start %2d | Step %2d | State %3d → %3d\n', ...
                    config, start, k, state_id, next_state_id);
                fprintf('  %-6s  %-8s %-8s %-8s   %-8s %-8s %-8s %-8s\n', ...
                    'Next','V_SR','V_SR_n','V_MB', ...
                    'p_SR','p_SR_n','p_MB','delta_n');
                for i = 1:length(poss_next)
                    cs = '';  if poss_next(i) == next_state_id; cs = ' <--'; end
                    fprintf('  %-6d  %-8.4f %-8.4f %-8.4f   %-8.4f %-8.4f %-8.4f %-8.4f%s\n', ...
                        poss_next(i), V_SR(i), V_SR_n(i), V_MB(i), ...
                        probs_SR_raw(i), probs_SR_norm(i), probs_MB_raw(i), ...
                        log(max(probs_SR_norm(i),1e-12))-log(max(probs_MB_norm(i),1e-12)), cs);
                end
                fprintf('  raw:  logP(SR)=%+.4f logP(MB)=%+.4f  delta=%+.4f\n', ...
                    log(max(p_SR_raw,1e-12)), log(max(p_MB_raw,1e-12)), d_raw);
                fprintf('  norm: logP(SR)=%+.4f logP(MB)=%+.4f  delta=%+.4f\n\n', ...
                    log(max(p_SR_nm,1e-12)), log(max(p_MB_nm,1e-12)), d_n);
                n_printed = n_printed + 1;
            end

            % Update SR
            M = TD_update(M, state_id, next_state_id, gamma, alpha);
            state_id = next_state_id;
            if state_id == goal_state; break; end
        end
    end
end

% -------------------------------------------------------------------------
%  Summary
% -------------------------------------------------------------------------
fprintf('=== SUMMARY over %d steps ===\n\n', n_steps);

fprintf('  --- RAW (original model) ---\n');
fprintf('  Total LL  SR=%.2f  MB=%.2f  SR-MB=%.2f (%+.4f/step)\n', ...
    ll_SR_raw, ll_MB_raw, ll_SR_raw-ll_MB_raw, (ll_SR_raw-ll_MB_raw)/n_steps);
fprintf('  SR>MB: %d (%.1f%%)   MB>SR: %d (%.1f%%)\n', ...
    sum(delta_ll_raw>0), 100*mean(delta_ll_raw>0), ...
    sum(delta_ll_raw<0), 100*mean(delta_ll_raw<0));
fprintf('  Percentiles: 5th=%.4f 25th=%.4f 50th=%.4f 75th=%.4f 95th=%.4f\n\n', ...
    prctile(delta_ll_raw,[5 25 50 75 95]));

fprintf('  --- NORMALISED (V_SR / max(V), V_MB unchanged) ---\n');
fprintf('  Total LL  SR=%.2f  MB=%.2f  SR-MB=%.2f (%+.4f/step)\n', ...
    ll_SR_n, ll_MB_n, ll_SR_n-ll_MB_n, (ll_SR_n-ll_MB_n)/n_steps);
fprintf('  SR>MB: %d (%.1f%%)   MB>SR: %d (%.1f%%)\n', ...
    sum(delta_ll_n>0), 100*mean(delta_ll_n>0), ...
    sum(delta_ll_n<0), 100*mean(delta_ll_n<0));
fprintf('  Percentiles: 5th=%.4f 25th=%.4f 50th=%.4f 75th=%.4f 95th=%.4f\n\n', ...
    prctile(delta_ll_n,[5 25 50 75 95]));

% -------------------------------------------------------------------------
%  Plot
% -------------------------------------------------------------------------
figure('Units','inches','Position',[1 1 14 8],'Color','w');

subplot(2,2,1)
plot(delta_ll_raw,'Color',[0.4 0.4 0.4],'LineWidth',0.5); hold on;
yline(0,'k--','LineWidth',1); yline(mean(delta_ll_raw),'r-','LineWidth',1.5);
title('Raw: \Delta LL per step (SR - MB)');
xlabel('Step'); ylabel('\Delta log-likelihood');
set(gca,'FontSize',11,'Box','on');

subplot(2,2,2)
plot(delta_ll_n,'Color',[0.4 0.4 0.4],'LineWidth',0.5); hold on;
yline(0,'k--','LineWidth',1); yline(mean(delta_ll_n),'r-','LineWidth',1.5);
title('Normalised: \Delta LL per step (SR - MB)');
xlabel('Step'); ylabel('\Delta log-likelihood');
set(gca,'FontSize',11,'Box','on');

subplot(2,2,3)
histogram(delta_ll_raw,50,'FaceColor',[0.3 0.3 0.3],'EdgeColor','none');
xline(0,'k--','LineWidth',1.5); xline(mean(delta_ll_raw),'r-','LineWidth',1.5);
title(sprintf('Raw  mean=%.4f/step',mean(delta_ll_raw)));
xlabel('\Delta LL (SR-MB)'); ylabel('Count');
set(gca,'FontSize',11,'Box','on');

subplot(2,2,4)
histogram(delta_ll_n,50,'FaceColor',[0.3 0.3 0.3],'EdgeColor','none');
xline(0,'k--','LineWidth',1.5); xline(mean(delta_ll_n),'r-','LineWidth',1.5);
title(sprintf('Normalised  mean=%.4f/step',mean(delta_ll_n)));
xlabel('\Delta LL (SR-MB)'); ylabel('Count');
set(gca,'FontSize',11,'Box','on');

sgtitle(sprintf('Human %d: Raw vs Normalised (alpha=%.3f, gamma=%.3f)', ...
    r, alpha, gamma),'FontSize',12);
