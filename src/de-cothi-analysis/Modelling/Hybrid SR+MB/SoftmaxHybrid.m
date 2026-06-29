function [p] = SoftmaxHybrid(V_SR, V_MB, w, poss_next, next_state_id)
p_SR = exp(V_SR - max(V_SR)) / sum(exp(V_SR - max(V_SR)));
p_MB = exp(V_MB - max(V_MB)) / sum(exp(V_MB - max(V_MB)));
probs = w * p_SR + (1-w) * p_MB;
p = probs(poss_next == next_state_id);
end
