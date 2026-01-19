function [poss_next, V_MB] = SoftmaxMB_V(state_id, blief_map, gamma, A_allowed)
%SOFTMAXMB_V Returns MB values for possible next states
%   state_id: current state
%   blief_map: belief map of obstacles
%   gamma: discount factor
%   A_allowed: allowed actions matrix
%   Returns poss_next: possible next state IDs
%   Returns V_MB: MB values for each possible next state

goal = zeros(10); goal(4,4) = 1;
blief_map_sample = rand(10,10) < blief_map;

poss_a = find(A_allowed(state_id,:) == 1);
poss_next = [];
V_MB = [];

for i = 1:length(poss_a)
    [next, ~] = action2state(state_id, poss_a(i), A_allowed);
    poss_next = [poss_next, next];
    if (next == 34)
        V_MB = [V_MB, 1];
    else
        [x,y] = state2coords(next);
        blief_map_temp = blief_map_sample;
        blief_map_temp(y,x) = 0;
        d = ASTARPATH(x,y,blief_map_temp,goal);
        if (d > 0)
            V_MB = [V_MB, gamma^(d)];
        else
            V_MB = [V_MB, 0];
        end
    end
end

end