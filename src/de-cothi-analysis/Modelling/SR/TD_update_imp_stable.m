function [new_M] = TD_update_imp_stable(M, state, next_state, gamma, alpha, imp_samp)
% TD_UPDATE_IMP_STABLE  Numerically stable importance-sampled TD update.

new_M = M;
one_hot        = zeros(1, 100);
one_hot(state) = 1;
target         = one_hot + gamma * M(next_state, :);
new_M(state,:) = (1 - alpha) * M(state, :) + alpha * imp_samp * target;
end
