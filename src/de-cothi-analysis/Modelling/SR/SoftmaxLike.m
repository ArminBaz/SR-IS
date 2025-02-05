function [ action, p ] = SoftmaxLike(V, state_id, next_state_id, A_allowed, invT)
% function p = SoftmaxLike(V, state_id, next_state_id, A_allowed)
%GREEDYCHOOSE Summary of this function goes here
%   Detailed explanation goes here
poss_a = find(A_allowed(state_id,:) == 1);
poss_next = [];
% invT = 1;
for i = 1:length(poss_a)
    [next, ~] = action2state(state_id,poss_a(i), A_allowed);
    poss_next = [poss_next, next];
end
V = V - max(V);
w = exp(invT*V(poss_next))+eps;
w = w/sum(w);
p = w(poss_next == next_state_id);
% w = (exp(invT * V(poss_next))+eps) / sum(exp(invT * V(poss_next))+eps);
% p = w(poss_next == next_state_id);
action = poss_a(poss_next == next_state_id);
end