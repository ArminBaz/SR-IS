function [poss_next, V_SR] = SoftmaxLike_V(V, state_id, A_allowed)
%SOFTMAXLIKE_V Returns SR values for possible next states
%   V: full value vector (100x1)
%   state_id: current state
%   A_allowed: allowed actions matrix
%   Returns poss_next: possible next state IDs
%   Returns V_SR: SR values for each possible next state

poss_a = find(A_allowed(state_id,:) == 1);
poss_next = [];
for i = 1:length(poss_a)
    [next, ~] = action2state(state_id, poss_a(i), A_allowed);
    poss_next = [poss_next, next];
end

V_SR = V(poss_next)';

end