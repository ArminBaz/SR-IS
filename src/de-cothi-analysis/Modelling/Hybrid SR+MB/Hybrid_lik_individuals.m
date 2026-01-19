load('rat.mat')
load('humans.mat')

% Parameters: [alpha, gamma, w]
% alpha: SR learning rate
% gamma: shared discount factor
% w: mixing weight (w*P_SR + (1-w)*P_MB)
x0 = [0.5, 0.5, 0.5];
lb = [0, 0, 0];
ub = [1, 1, 1];
A = []; b = []; Aeq = []; beq = [];

% dat = humans;
dat = rat;

n_ppt = size(dat,1);
n_params = length(x0);
ppt_params = zeros(n_ppt, n_params);
ll = 0;
Hybrid_llik = cell(n_ppt, 25, 10);

for i = 1:n_ppt
    fun = @(x)-Hybrid_lik(x, dat(i,:,:));
    % fun = @(x)-Hybrid_lik_V(x, dat(i,:,:));
    % options = optimset('Display','iter','PlotFcns',@optimplotfval);
    options = optimset('Display','iter');
    [x, f_val] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, [], options);

    ppt_params(i,:) = x;
    ll = f_val + ll;

    [~, participant_llik] = Hybrid_lik(x, dat(i,:,:));
    % [~, participant_llik] = Hybrid_lik_V(x, dat(i,:,:));
    
    Hybrid_llik(i,:,:) = participant_llik;
end

% Save final results
% save('human_Hybrid_llik_V2.mat', 'ppt_params', 'll', 'Hybrid_llik');
save('rat_Hybrid_llik.mat', 'ppt_params', 'll', 'Hybrid_llik');