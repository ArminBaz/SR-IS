function [new_M] = TD_update_imp(M, state, next_state, gamma, alpha, imp_samp)
%TD_UPDATE Summary of this function goes here
%   Detailed explanation goes here
new_M = M;
new_M(state,:) = M(state,:) + alpha*(imp_samp * (1*(1:100 == state) + gamma * M(next_state,:)) - M(state,:));
end

