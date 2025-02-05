function [new_M] = TD_trace_update_imp(M, E, state, next_state, gamma, alpha, imp_samp)
%TD_UPDATE Summary of this function goes here
%   Detailed explanation goes here
new_M = M + alpha * E' * (imp_samp * ((1:100 == state) + gamma * M(next_state,:)) - M(state,:));
% Calculate the difference
% diff = new_M - M;
% max_diff = max(abs(diff(:)));
% fprintf('Max difference in M: %.6e\n , imp_samp: %d', max_diff, imp_samp);
% Check inputs for NaN
if any(isnan(M(:))) || any(isnan(E(:))) || isnan(imp_samp)
    fprintf('NaN detected in input\n');
    if any(isnan(M(:)))
        fprintf('NaN in M matrix\n');
    end
    if any(isnan(E(:)))
        fprintf('NaN in E vector\n');
    end
    if isnan(imp_samp)
        fprintf('NaN in imp_samp\n');
    end
end
end
