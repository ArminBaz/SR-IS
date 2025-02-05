function [ll] = SR_lik_imp(x,dat)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

alpha=0.6; gamma = x(1); invT = x(2);
n_samples = size(dat,1);
llik = cell(n_samples,25,10);
acts = cell(n_samples,25,10);

load('mazes.mat')
load('train_T.mat')
opt_M = inv(eye(100)-gamma*T);
opt_w = zeros(100,1);
opt_w(34) = 1;
mazes = mazes;
E = zeros(100);

%% Run Testing
for r = 1:n_samples

    M = opt_M;
    w = opt_w;
    for config = 1:25
        A_allowed = map2allowed(mazes{config});
        
        for start = 1:10            
            lik = 1;
            
            traj = dat{r,config,start};
            if(~isempty(traj))
                
                state_id = traj(1);
                for k = 1:(length(traj)-1)
                    V = M*w;
                    [a_id, p] = SoftmaxLike(V, state_id, traj(1+k), A_allowed, invT);
                    lik = [lik, p];
                    imp_samp = ImportanceSampling(p, state_id, A_allowed, 10);
                    if isnan(imp_samp)
                        disp("imp samp becomes nan");
                    end
                    % observe the next state and next reward ** there is no reward matrix
                    [next_state_id, ~] = action2state(state_id, a_id, A_allowed);

                    % update the Q matrix using the Q-learning rule
                    M = TD_update_imp(M,state_id, next_state_id, gamma, alpha, imp_samp);
                    if isnan(M)
                        disp("M becomes nan");
                    end
                    
                    state_id = next_state_id;
                    if(state_id == 34)
                        break
                    end
                end
            end
            llik{r,config,start} = sum(log(lik));            
        end
    end
end

ll = sum(cellfun(@sum,llik),'all');
end