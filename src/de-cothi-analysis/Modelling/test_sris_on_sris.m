clear; clc;

addpath('../Standard Functions (add to path)')
addpath('Model-free')
addpath('Model-based')
addpath('Hybrid SR+MB')
addpath('SR')

load('mazes.mat')
load('train_T.mat')
load('humans.mat')
load('SR/human_SR_IS_llik.mat')
tmp = load('rat.mat'); rat = tmp.rat;

r = 1;
alpha  = ppt_params(r, 1);
gamma  = ppt_params(r, 2);
lambda = ppt_params(r, 3);
fprintf('SR-IS human 1 params: alpha=%.3f  gamma=%.3f  lambda=%.3f\n', alpha, gamma, lambda);

% Simulate SR-IS human 1
n_configs  = 25;
n_starts   = 10;
goal_state = 34;
max_steps  = 500;
const      = 0.0001;

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

sim_dat = cell(1, n_configs, n_starts);
for config = 1:n_configs
    T_maze    = maze_to_transition_matrix(mazes{config});
    A_allowed = map2allowed(mazes{config});
    P         = T_maze(nonterminals, terminals);
    for start = 1:n_starts
        real_traj = humans{r, config, start};
        if isempty(real_traj); continue; end
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
            M    = TD_update_imp(M, state_id, next_state, gamma, alpha, imp_samp);
            state_id = next_state;
            if state_id == goal_state; break; end
        end
        sim_dat{1, config, start} = traj;
    end
end
fprintf('Simulation done.\n');

% Test at true parameters — old vs stable
ll_old   = SR_IS_lik([alpha, gamma, lambda], sim_dat);
ll_stable = SR_IS_lik_stable([alpha, gamma, lambda], sim_dat);
fprintf('SR_IS_lik        at true params: %.4f\n', ll_old);
fprintf('SR_IS_lik_stable at true params: %.4f\n', ll_stable);

% Test at x0 — old vs stable
ll_x0_old    = SR_IS_lik([0.5, 0.5, 0.5], sim_dat);
ll_x0_stable = SR_IS_lik_stable([0.5, 0.5, 0.5], sim_dat);
fprintf('SR_IS_lik        at x0=[0.5,0.5,0.5]: %.4f\n', ll_x0_old);
fprintf('SR_IS_lik_stable at x0=[0.5,0.5,0.5]: %.4f\n', ll_x0_stable);

% Fit using stable version
fprintf('Fitting SR-IS with stable update...\n');
t0 = tic;
A=[]; b=[]; Aeq=[]; beq=[]; nonlcon=[];
fit_opts_IS = optimset('Display','off','MaxFunEvals',100,'MaxIter',100);
x0_IS = [0.5, 0.5, 0.5]; lb_IS = [0,0,0]; ub_IS = [1,1,100];
try
    [xfit, ll] = fmincon(@(x) -SR_IS_lik_stable(x, sim_dat), ...
        x0_IS, A,b,Aeq,beq, lb_IS, ub_IS, nonlcon, fit_opts_IS);
    sris_ll = -ll;
catch
    sris_ll = -1e8;
    xfit = nan(1,3);
end
fprintf('SR-IS fitted ll = %.4f  (%.1fs)\n', sris_ll, toc(t0));
fprintf('Fitted params: alpha=%.3f  gamma=%.3f  lambda=%.3f\n', xfit(1), xfit(2), xfit(3));
