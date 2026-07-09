function [params_sampled, param_mean, param_std] = sample_model_params_combined(model_idx, n_samples, rng_seed)
% SAMPLE_MODEL_PARAMS_COMBINED  Sample parameters from combined human+rat distributions.
%
%   model_idx  : 1=MF  2=MB  3=SR  4=Hybrid  5=SR-IS
%   n_samples  : number of parameter vectors to draw
%   rng_seed   : RNG seed for reproducibility

if nargin < 3; rng_seed = 42; end

bounds = {[0 0; 1 1], ...          % MF:    alpha, gamma
          [0; 1], ...               % MB:    gamma
          [0 0; 1 10], ...          % SR:    alpha, gamma
          [0 0 0; 1 1 1], ...       % Hybrid: alpha, gamma, w
          [0 0 0; 1 1 100]};        % SR-IS:  alpha, gamma, lambda

lb = bounds{model_idx}(1,:);
ub = bounds{model_idx}(2,:);

switch model_idx
    case 1  % MF
        h = load('Model-free/human_params.mat', 'ppt_params');
        r = load('Model-free/rat_params.mat',   'ppt_params');
    case 2  % MB
        h = load('Model-based/human_params.mat', 'ppt_params');
        r = load('Model-based/rat_params.mat',   'ppt_params');
    case 3  % SR
        h = load('SR/human_params.mat', 'ppt_params');
        r = load('SR/rat_params.mat',   'ppt_params');
    case 4  % Hybrid
        h = load('Hybrid SR+MB/human_Hybrid_llik.mat', 'ppt_params');
        r = load('Hybrid SR+MB/rat_Hybrid_llik.mat',   'ppt_params');
    case 5  % SR-IS
        h = load('SR/human_SR_IS_llik_V2.mat', 'ppt_params');
        r = load('SR/rat_SR_IS_llik_V2.mat',   'ppt_params');
end

ppt_params = [h.ppt_params; r.ppt_params];

param_mean = mean(ppt_params, 1);
param_std  = std(ppt_params, 0, 1);

rng(rng_seed);
raw = normrnd(repmat(param_mean, n_samples, 1), repmat(param_std, n_samples, 1));
params_sampled = max(repmat(lb, n_samples, 1), min(repmat(ub, n_samples, 1), raw));

% SR-IS: sample lambda in log-space
if model_idx == 5
    lambda_vals = ppt_params(:, 3);
    lambda_vals = lambda_vals(lambda_vals > 0);
    log_lambda  = log(lambda_vals);
    log_mean    = mean(log_lambda);
    log_std     = std(log_lambda);
    lambda_samp = exp(normrnd(log_mean, log_std, n_samples, 1));
    lambda_samp = min(lambda_samp, ub(3));
    params_sampled(:, 3) = lambda_samp;
    param_mean(3) = exp(log_mean);
    param_std(3)  = log_std;
end
end
