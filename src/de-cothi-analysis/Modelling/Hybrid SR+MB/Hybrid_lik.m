function [ll, llik_out] = Hybrid_lik(x, dat)
%HYBRID_LIK Hybrid SR+MB likelihood function
%   Combines SR and MB models with probability mixing
%   Parameters: x(1) = alpha (SR learning rate)
%               x(2) = gamma (shared discount factor)
%               x(3) = w (mixing weight: w*P_SR + (1-w)*P_MB)

alpha = x(1);
gamma = x(2);
w = x(3);

n_samples = size(dat,1);
llik = cell(n_samples,25,10);

load('mazes.mat')
load('train_T.mat')

% Initialize SR model
opt_M = inv(eye(100) - gamma*T);
opt_w = zeros(100,1);
opt_w(34) = 1;
mazes = mazes;

for r = 1:n_samples
    % SR initialization
    M = opt_M;
    sr_w = opt_w;

    % MB initialization
    blief_map = zeros(10);

    for config = 1:25
        A_allowed = map2allowed(mazes{config});

        for start = 1:10
            lik = 1;
            traj = dat{r, config, start};

            if (~isempty(traj))
                state_id = traj(1);

                for k = 1:(length(traj)-1)
                    next_state_id = traj(1+k);

                    % SR: compute value and get probability
                    V = M * sr_w;
                    [~, p_SR] = SoftmaxLike(V, state_id, next_state_id, A_allowed);

                    % MB: update beliefs and get probability
                    blief_map = update_bliefs(blief_map, state_id, mazes{config});
                    p_MB = SoftmaxMB(state_id, next_state_id, blief_map, gamma, A_allowed);

                    % Hybrid probability
                    p_hybrid = w * p_SR + (1-w) * p_MB;
                    lik = [lik, p_hybrid];

                    % SR: update M matrix
                    [next_state_id_sr, ~] = action2state(state_id, find(A_allowed(state_id,:) == 1, 1), A_allowed);
                    % Get actual next state for SR update
                    poss_a = find(A_allowed(state_id,:) == 1);
                    for a_idx = 1:length(poss_a)
                        [ns, ~] = action2state(state_id, poss_a(a_idx), A_allowed);
                        if ns == next_state_id
                            break
                        end
                    end
                    M = TD_update(M, state_id, next_state_id, gamma, alpha);

                    state_id = next_state_id;
                    if (state_id == 34)
                        break
                    end
                end
            end
            llik{r, config, start} = log(lik);
        end
    end
end

ll = sum(cellfun(@sum, llik), 'all');
llik_out = llik;
end
