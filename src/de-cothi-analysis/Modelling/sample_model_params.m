function [params_sampled, param_median, param_sem] = sample_model_params(model_idx, species_idx, n_samples, rng_seed)
% SAMPLE_MODEL_PARAMS  Sample parameter vectors from fitted population distributions.
%
%   [params_sampled, param_median, param_sem] = sample_model_params(model_idx, species_idx, n_samples, rng_seed)
%
%   model_idx  : 1=MF  2=MB  3=SR  4=Hybrid  5=SR-IS
%   species_idx: 1=human  2=rat
%   n_samples  : number of parameter vectors to draw
%   rng_seed   : RNG seed for reproducibility
%
%   params_sampled : [n_samples x n_params] clamped to valid bounds
%   param_median   : [1 x n_params] median across participants
%   param_sem      : [1 x n_params] median-based SEM (1.2533*std/sqrt(n))

if nargin < 4; rng_seed = 42; end

% Parameter bounds per model  [lb; ub]
bounds = {[0 0; 1 1], ...          % MF:    alpha, gamma
          [0; 1], ...               % MB:    gamma
          [0 0; 1 10], ...          % SR:    alpha, gamma
          [0 0 0; 1 1 1], ...          % Hybrid: alpha, gamma, w
          [0 0 0; 1 1 100]};        % SR-IS: alpha, gamma, lambda

lb = bounds{model_idx}(1,:);
ub = bounds{model_idx}(2,:);

% Load fitted parameters
switch model_idx
    case 1  % MF
        if species_idx == 1
            load('Model-free/human_params.mat', 'ppt_params');
        else
            load('Model-free/rat_params.mat',   'ppt_params');
        end
    case 2  % MB
        if species_idx == 1
            load('Model-based/human_params.mat', 'ppt_params');
        else
            load('Model-based/rat_params.mat',   'ppt_params');
        end
    case 3  % SR
        if species_idx == 1
            load('SR/human_params.mat', 'ppt_params');
        else
            load('SR/rat_params.mat',   'ppt_params');
        end
    case 4  % Hybrid
        if species_idx == 1
            load('Hybrid SR+MB/human_Hybrid_llik.mat', 'ppt_params');
        else
            load('Hybrid SR+MB/rat_Hybrid_llik.mat',   'ppt_params');
        end
    case 5  % SR-IS
        if species_idx == 1
            load('SR/human_SR_IS_llik.mat', 'ppt_params');
        else
            load('SR/rat_SR_IS_llik.mat',   'ppt_params');
        end
end

n_ppt      = size(ppt_params, 1);
param_median = median(ppt_params, 1);
param_sem    = 1.2533 * std(ppt_params, 0, 1) / sqrt(n_ppt);

% Sample and clamp
rng(rng_seed);
raw = normrnd(repmat(param_median, n_samples, 1), repmat(param_sem, n_samples, 1));
params_sampled = max(repmat(lb, n_samples, 1), min(repmat(ub, n_samples, 1), raw));

% SR-IS: resample lambda (col 3) in log-space — lambda is strictly positive
% and spans orders of magnitude, so normal sampling produces near-zero values.
if model_idx == 5
    lambda_vals = ppt_params(:, 3);
    lambda_vals = lambda_vals(lambda_vals > 0);   % exclude any zero fits
    log_lambda  = log(lambda_vals);
    log_med     = median(log_lambda);
    log_sem     = 1.2533 * std(log_lambda) / sqrt(numel(log_lambda));
    lambda_samp = exp(normrnd(log_med, log_sem, n_samples, 1));
    lambda_samp = min(lambda_samp, ub(3));         % clamp to upper bound
    params_sampled(:, 3) = lambda_samp;
    % Update reported median/sem to reflect log-space distribution
    param_median(3) = exp(log_med);
    param_sem(3)    = log_sem;   % reported in log units for transparency
end
end
