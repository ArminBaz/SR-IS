function T = maze_to_transition_matrix(maze)
    % Convert a maze to a transition matrix
    
    working_maze = maze;
    working_maze(working_maze > 0) = 0;
    [rows, cols] = size(working_maze);
    
    % Create state mapping using a matrix instead of dictionaries
    state_map = -ones(rows, cols);  % -1 for walls
    state_idx = 0;
    
    for i = 1:rows
        for j = 1:cols
            if working_maze(i, j) ~= -1
                state_map(i, j) = state_idx;
                state_idx = state_idx + 1;
            end
        end
    end
    
    % Initialize transition matrix (hardcoded 100 states)
    T = zeros(100, 100);
    
    % Possible moves: up, down, left, right
    moves = [-1, 0; 1, 0; 0, -1; 0, 1];
    
    % For each cell in the maze
    for i = 1:rows
        for j = 1:cols
            if state_map(i, j) >= 0  % If it's an accessible state
                state = state_map(i, j);
                accessible_neighbors = [];
                
                % Check all 4 directions
                for m = 1:4
                    ni = i + moves(m, 1);
                    nj = j + moves(m, 2);
                    
                    % Check if neighbor is valid and accessible
                    if ni >= 1 && ni <= rows && nj >= 1 && nj <= cols
                        if state_map(ni, nj) >= 0
                            accessible_neighbors = [accessible_neighbors, state_map(ni, nj)];
                        end
                    end
                end
                
                % Set uniform transition probabilities
                if ~isempty(accessible_neighbors)
                    prob = 1.0 / length(accessible_neighbors);
                    T(state + 1, accessible_neighbors + 1) = prob;
                end
            end
        end
    end
    
    T(34, 34) = 1.0;
end