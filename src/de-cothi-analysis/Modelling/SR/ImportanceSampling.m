function w = ImportanceSampling(p, state_id, A_allowed, max_imp)
% Importance Sampling function
% Calculate the probability of selecting next state from current state
s_prob = 1/(sum(A_allowed(state_id,:)==1));
w = s_prob / (p+eps);
w = min(max(w, s_prob), max_imp);
end
