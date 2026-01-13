function transition_matrix = create_transition_matrix_open(N)
    % Create a transition matrix for a random walk on an NxN grid
    % N defaults to 10 if not provided
    
    if nargin < 1
        N = 10;
    end
    
    total_states = N * N;
    transition_matrix = zeros(total_states, total_states);
    
    % Helper function: convert (i,j) coordinates to linear index
    % MATLAB uses 1-based indexing
    function idx = coord_to_index(i, j)
        idx = (i-1) * N + j;
    end
    
    % Helper function: convert linear index to (i,j) coordinates
    function [i, j] = index_to_coord(idx)
        i = floor((idx-1) / N) + 1;
        j = mod(idx-1, N) + 1;
    end
    
    % For each state (position on the grid)
    for idx = 1:total_states
        [i, j] = index_to_coord(idx);
        
        % Find valid neighbors (up, down, left, right)
        neighbors = [];
        
        % Up
        if i > 1
            neighbors = [neighbors, coord_to_index(i-1, j)];
        end
        
        % Down
        if i < N
            neighbors = [neighbors, coord_to_index(i+1, j)];
        end
        
        % Left
        if j > 1
            neighbors = [neighbors, coord_to_index(i, j-1)];
        end
        
        % Right
        if j < N
            neighbors = [neighbors, coord_to_index(i, j+1)];
        end
        
        % Uniform transition probability to each valid neighbor
        if ~isempty(neighbors)
            prob = 1.0 / length(neighbors);
            transition_matrix(idx, neighbors) = prob;
        end
    end
    
    TERM_IDX = 34;
    transition_matrix(TERM_IDX, :) = 0;
    transition_matrix(TERM_IDX, TERM_IDX) = 1;
end