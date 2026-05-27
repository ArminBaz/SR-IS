clear; clc;

addpath('../Standard Functions (add to path)')
addpath('Model-free')
addpath('Model-based')
addpath('Hybrid SR+MB')
addpath('SR')

load('mazes.mat')
load('train_T.mat')
load('humans.mat')
load('Model-free/PreQ.mat')
load('Model-free/human_params.mat')
tmp = load('rat.mat'); rat = tmp.rat;

% Simulate MF human 1
alpha = ppt_params(1,1);
gamma = ppt_params(1,2);
Q = gamma .^ PreQ;   Q(isnan(Q)) = 0;

sim_dat = cell(1, 25, 10);
for config = 1:25
    A_allowed = map2allowed(mazes{config});
    for start = 1:10
        real_traj = humans{1, config, start};
        if isempty(real_traj); continue; end
        state_id = real_traj(1);   traj = state_id;
        for step = 1:500
            poss_a    = find(A_allowed(state_id,:) == 1);
            poss_next = zeros(1, length(poss_a));
            for i = 1:length(poss_a)
                [poss_next(i), ~] = action2state(state_id, poss_a(i), A_allowed);
            end
            vals   = max(Q(poss_next,:), [], 2);
            logits = vals - max(vals);
            probs  = exp(logits) / sum(exp(logits));
            chosen_idx  = randsample(length(poss_next), 1, true, probs);
            next_state  = poss_next(chosen_idx);
            a_id        = poss_a(chosen_idx);
            [~, reward] = action2state(state_id, a_id, A_allowed);
            traj     = [traj, next_state];
            Q        = Q_update(Q, state_id, a_id, reward, next_state, gamma, alpha);
            state_id = next_state;
            if state_id == 34; break; end
        end
        sim_dat{1, config, start} = traj;
    end
end
fprintf('Simulation done.\n');

% Test SR-IS fitting with NaN guard
fprintf('Fitting SR-IS...\n');
t0 = tic;

A=[]; b=[]; Aeq=[]; beq=[]; nonlcon=[];
fit_opts_IS = optimset('Display','off','MaxFunEvals',100,'MaxIter',100);
x0_IS = [0.5, 0.5, 0.5];
lb_IS = [0, 0, 0];
ub_IS = [1, 1, 100];

try
    [~, ll] = fmincon(@(x) sr_is_safe(x, sim_dat), ...
        x0_IS, A,b,Aeq,beq, lb_IS, ub_IS, nonlcon, fit_opts_IS);
    sris_ll = -ll;
catch
    sris_ll = -1e8;
end

fprintf('SR-IS ll = %.4f  (%.1fs)\n', sris_ll, toc(t0));

% -------------------------------------------------------------------------
function val = sr_is_safe(x, dat)
ll = SR_IS_lik(x, dat);
if isnan(ll) || isinf(ll)
    val = 1e8;
else
    val = -ll;
end
end
