function [p] = SoftmaxHybrid(V_SR, V_MB, w, poss_next, next_state_id)
%SOFTMAXHYBRID Combines SR and MB values and returns action probability
%   V_SR: SR values for each possible next state
%   V_MB: MB values for each possible next state
%   w: mixing weight (w*V_SR + (1-w)*V_MB)
%   poss_next: possible next state IDs
%   next_state_id: the actually chosen next state
%   Returns p: probability of choosing next_state_id

% Combine values
V_hybrid = w * V_SR + (1-w) * V_MB;

% Apply softmax
invT = 1;
probs = exp(invT * V_hybrid) / sum(exp(invT * V_hybrid));

% Get probability of chosen action
p = probs(poss_next == next_state_id);

end
