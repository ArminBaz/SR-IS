function [ll, llik_out] = SR_IS_lik(x,dat)
alpha=x(1); gamma=x(2); lambda=x(3);
const = 0.0001;

n_samples = size(dat,1);
llik = cell(n_samples,25,10);
load('mazes.mat')
load('train_T.mat')

% opt_M = inv(eye(100)-gamma*T);
% opt_w = zeros(100,1);

T_open = create_transition_matrix_open();
I = eye(100);
opt_M = inv(I - gamma * T_open + 1e-10 * I);
opt_M = gamma*inv(eye(100)-gamma*T_open);
reward_nt = lambda * log(gamma);
opt_w = reward_nt * ones(100,1);
opt_w(34) = 1;
rmax = 0.0;
reward_t = exp((opt_w(34)-rmax) / lambda);
terminals = 34;
nonterminals = setdiff(1:100, terminals);

for r = 1:n_samples
    M = opt_M;
    w = opt_w;
    for config = 1:25
        T_maze = maze_to_transition_matrix(mazes{config});
        A_allowed = map2allowed(mazes{config});
        P = T_maze(nonterminals, terminals);
        
        for start = 1:10
            lik = 1;
            traj = dat{r,config,start};
            
            if(~isempty(traj))
                state_id = traj(1);
                for k = 1:(length(traj)-1)
                    Z = zeros(100, 1);
                    Z(nonterminals) = (gamma * M(nonterminals, nonterminals)) * P * reward_t;
                    Z(terminals) = reward_t;
                    Z(Z < const) = const;
                    logZ = log(Z);
                    logZ = logZ + rmax/lambda;
                    V = logZ * lambda;
                    % V = M*w;
                    invT = 1/lambda;
                    [a_id, p] = SoftmaxLike(V, state_id, traj(1+k), A_allowed, invT);
                    lik = [lik, p];
                    
                    imp_samp = ImportanceSampling(p, state_id, A_allowed, 10);
                    if isnan(imp_samp)
                        disp("imp samp becomes nan");
                    end
                    
                    [next_state_id, ~] = action2state(state_id, a_id, A_allowed);
                    M = TD_update_imp(M, state_id, next_state_id, gamma, alpha, imp_samp);

                    if isnan(M)
                        disp("M becomes nan");
                    end

                    state_id = next_state_id;
                    if(state_id == 34)
                        break
                    end
                end
            end
            llik{r,config,start} = log(lik);
        end
    end
end

ll = sum(cellfun(@sum,llik),'all');
llik_out = llik;
end