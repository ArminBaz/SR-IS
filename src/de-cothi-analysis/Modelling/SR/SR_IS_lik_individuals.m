load('rat.mat')
load('humans.mat')

x0 = [0.5, 0.5, 0.5];
lb = [0, 0, 0];
ub = [1.0, 1.0, 100.0];

A = []; b = []; Aeq = []; beq = [];

% dat = humans;
dat = rat;
n_ppt = size(dat, 1);
n_params = length(x0);
ppt_params = zeros(n_ppt, n_params);
SR_IS_llik = cell(n_ppt, 25, 10);
ll = 0;

% Fit all participants
for i = 1:n_ppt
    fprintf('\nFitting participant %d/%d\n', i, n_ppt);
    
    % Set up optimization function
    fun = @(x)-SR_IS_lik(x, dat(i,:,:));
    % options = optimset('Display', 'iter', 'MaxFunEvals', 200, 'MaxIter', 100);
    options = optimset('Display','iter','PlotFcns',@optimplotfval);
    % Run optimization
    [x, f_val] = fmincon(fun, x0, A, b, Aeq, beq, lb, ub, [], options);
    
    % Store results
    ppt_params(i,:) = x;
    ll = f_val + ll;
    
    % Calculate likelihoods with best parameters
    [~, participant_llik] = SR_IS_lik(x, dat(i,:,:));
    SR_IS_llik(i,:,:) = participant_llik;
    
    fprintf('Participant %d completed. Alpha=%.3f, Gamma=%.3f, LL=%.3f\n', i, x(1), x(2), -f_val);
end

% Save final results
% save('human_SR_IS_llik.mat', 'ppt_params', 'll', 'SR_IS_llik');
save('rat_SR_IS_llik.mat', 'ppt_params', 'll', 'SR_IS_llik');