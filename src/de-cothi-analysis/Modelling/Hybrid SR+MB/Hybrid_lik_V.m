function [ll, llik_out] = Hybrid_lik_V(x, dat)
%HYBRID_LIK_V Hybrid SR+MB likelihood function with VALUE-level mixing
%   Combines SR and MB models with value mixing
%   Parameters: x(1) = alpha (SR learning rate)
%               x(2) = gamma (shared discount factor)
%               x(3) = w (mixing weight: w*V_SR + (1-w)*V_MB)

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

                    % SR: compute values for possible next states
                    V = M * sr_w;
                    [poss_next, V_SR] = SoftmaxLike_V(V, state_id, A_allowed);

                    % MB: update beliefs and get values for possible next states
                    blief_map = update_bliefs(blief_map, state_id, mazes{config});
                    [~, V_MB] = SoftmaxMB_V(state_id, blief_map, gamma, A_allowed);

                    % Hybrid: combine values and get probability via softmax
                    p_hybrid = SoftmaxHybrid(V_SR, V_MB, w, poss_next, next_state_id);
                    lik = [lik, p_hybrid];

                    % SR: update M matrix
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
