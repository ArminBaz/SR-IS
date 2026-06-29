load('rat.mat')
load('humans.mat')

% Parameters: [alpha, gamma, w]
% alpha: SR learning rate
% gamma: shared discount factor
% w: mixing weight (w*V_SR + (1-w)*V_MB)
x0 = [0.5, 0.5, 0.5];
lb = [0, 0, 0];
ub = [1, 0.9, 1];
A = []; b = []; Aeq = []; beq = [];

for sp = 2:2
    if sp == 1
        dat = humans;
        savefile = 'human_Hybrid_llik.mat';
        species = 'human';
    else
        dat = rat;
        savefile = 'rat_Hybrid_llik_V2.mat';
        species = 'rat';
    end

    n_ppt = size(dat,1);
    n_params = length(x0);
    ppt_params = zeros(n_ppt, n_params);
    ll = 0;
    Hybrid_llik = cell(n_ppt, 25, 10);

    fprintf('\n===== %s (n=%d) =====\n', species, n_ppt);

    for i = 1:n_ppt
        fprintf('\n--- Participant %d ---\n', i);
        t0 = tic;
        fun = @(x)-Hybrid_lik_V(x, dat(i,:,:));
        options = optimset('Display','iter');
        [x, f_val] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, [], options);

        ppt_params(i,:) = x;
        ll = f_val + ll;

        [~, participant_llik] = Hybrid_lik_V(x, dat(i,:,:));
        Hybrid_llik(i,:,:) = participant_llik;

        fprintf('  alpha=%.4f  gamma=%.4f  w=%.4f  (%.1fs)\n', ...
            x(1), x(2), x(3), toc(t0));
    end

    save(savefile, 'ppt_params', 'll', 'Hybrid_llik');
    fprintf('\nSaved %s\n', savefile);
end
