%% SR-TD With Importance Sampling
% Modified the original code to include importance sampling

load('human_params.mat')
load('humans.mat')
ppts_traj = humans;

% load('rat_params.mat')
% load('rat.mat')
% ppts_traj = rat;

K = 45;

n_samples = 100;
n_ppt = length(ppt_params);
success = nan(n_samples*n_ppt,25,10);
trajectories = cell(n_samples*n_ppt,25,10);
lambda = 0.5;

load('mazes.mat')
A_space = ["N",0,-1;"E",1,0;"S",0,1;"W",-1,0];

%% Training
load('train_T.mat')
opt_w = zeros(100,1);
opt_w(34) = 1;

%% Run Testing
for ppt = 1:n_ppt
    alpha = 0.7;
    gamma = 0.2;
    % alpha = 0.8;
    % gamma = 0.1;
    invT = 1;
    opt_M = inv(eye(100)-gamma*T);
    
    for s = 1:n_samples
        agent_idx = (ppt-1)*n_samples+s;
        display(agent_idx/(n_samples*n_ppt));
        old_M = opt_M;
        w = opt_w;
        
        for config = 1:25
            A_allowed = map2allowed(mazes{config});
            M = old_M;
            
            epsilon = 0.11;
            % Simulate behaviour
            for start = 1:10
                old_E = zeros(1,100);
                E = zeros(1,100);
                epsilon = epsilon-0.01;
                [state_y, state_x] = find(mazes{config} == start);
                state_id = coords2state(state_x,state_y);
                traj = state_id;
                
                for k = 1:K   % number of timepoints left
                    V = M*w;
                    rndm = rand; % get 1 uniform random number
                    x = sum(rndm >= cumsum([0, 1-epsilon, epsilon])); % check it to be in which probability area
                    %explore or exploit
                    if x == 1   % exploit
                        current_action = A_space(GreedyChoose(V, state_id, A_allowed), 1);
                    else        % explore
                        current_action = A_space(datasample(find(A_allowed(state_id,:) == 1),1),1); % choose 1 action randomly (uniform random distribution)
                    end
                    % action chosen, now find reward
                    a_id = find(A_space(:,1) == current_action); % idx of the chosen action
                    % observe the next state and next reward ** there is no reward matrix
                    [next_state_id, reward] = action2state(state_id, a_id, A_allowed);
                    % update the SR matrix using TD rule
                    
                    E(state_id) = E(state_id) + 1;
                    
                    % Armin addition : Importance sampling
                    [~,p] = SoftmaxLike(V, state_id, next_state_id, A_allowed, invT);
                    imp_samp = ImportanceSampling(p, state_id, A_allowed, 20);

                    M = TD_trace_update_imp(M, E, state_id, next_state_id, gamma, alpha, imp_samp);
                    % M = TD_update_imp(M, state_id, next_state_id, gamma, alpha, imp_samp);
                    E = gamma*lambda*E;

                    % update the stored trajectories
                    traj = [traj, next_state_id];
                    % if reached terminal state
                    if (next_state_id == 34)
                        success(agent_idx,config,start) = 1;
                        trajectories{agent_idx,config,start} = traj;
                        break
                    elseif (k ==K)
                        trajectories{agent_idx,config,start} = traj;
                        success(agent_idx,config,start) = 0;
                        break
                    else
                        state_id = next_state_id;
                    end
                end

                ppt_traj = ppts_traj{ppt,config,start};
                if (~isempty(ppt_traj))
                    time_steps_left = length(ppt_traj)-1;
                    % Update M based on true trajectory
                    for t = 1:time_steps_left
                        V = old_M*w;
                        if isnan(V)
                            disp("V is nan")
                        end
                        state = ppt_traj(t);
                        next_state = ppt_traj(t+1);
                
                        % Armin addition: importance sampling
                        [~,p] = SoftmaxLike(V, state, next_state, A_allowed, invT);
                        imp_samp = ImportanceSampling(p, state_id, A_allowed, 20);
                        

                        old_E(state) = old_E(state) + 1;
                        old_M = TD_trace_update_imp(old_M, old_E, state, next_state, gamma, alpha, imp_samp);
                        % old_M = TD_update_imp(old_M, state, next_state, gamma, alpha, imp_samp);
                        old_E = gamma*lambda*old_E;

                        if(next_state == 34)
                            break
                        end
                    end
                end
            end
        end
    end
end

SR_imp_succ = success;
SR_imp = trajectories;