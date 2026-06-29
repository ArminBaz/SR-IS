function [ new_map ] = update_bliefs( blief_map, state, map)
%UPDATE_BLIEFS

alpha=1;
vis=1;

[state_x, state_y] = state2coords(state);

ids = zeros(10); ids(state_y,state_x) = 1;
ids = bwdist(ids) <= vis;

blief_map(ids) = blief_map(ids) + alpha * ((map(ids) == -1) - blief_map(ids));

new_map = blief_map;
end

