using MixedModels
using RLSR

include("regression_sim_models.jl")

""" These functions take the current reward state, as well as left/right indices of the currently best rewards
So if policy_side_2 = 1, then on the right island, the left boat (1) is better than the right (2)
"""
function onpolicy_swap!(rewards, policy_side_1, policy_side_2)
    rewards[policy_side_1], rewards[2 + policy_side_2] = rewards[2 + policy_side_2], rewards[policy_side_1]
    rewards[3 - policy_side_1], rewards[5 - policy_side_2] = rewards[5 - policy_side_2], rewards[3 - policy_side_1]
    (policy_side_1, policy_side_2)
end

function offpolicy_swap!(rewards, policy_side_1, policy_side_2)
    rewards[policy_side_1], rewards[5 - policy_side_2] = rewards[5 - policy_side_2], rewards[policy_side_1]
    rewards[3 - policy_side_1], rewards[2 + policy_side_2] = rewards[2 + policy_side_2], rewards[3 - policy_side_1]
    (3 - policy_side_1, 3 - policy_side_2)
end

function within_swap!(rewards, policy_side_1, policy_side_2)
    rewards[1], rewards[2] = rewards[2], rewards[1]
    rewards[3], rewards[4] = rewards[4], rewards[3]
    (3 - policy_side_1, 3 - policy_side_2)
end


# Simple helper function to initialize models with perfect reward knowledge
function initialize_perfect_rewards!(model, rewards)
    # For SR model - set perfect reward knowledge
    if hasfield(typeof(model), :R)
        # Map the 4 boat rewards to states 4,5,6,7
        model.R[4] = (rewards[1] - 0.5) * 2  # Convert back to [-1,1] scale if needed
        model.R[5] = (rewards[2] - 0.5) * 2
        model.R[6] = (rewards[3] - 0.5) * 2  
        model.R[7] = (rewards[4] - 0.5) * 2
        
        # Update V and Q based on perfect rewards
        model.V[:] = model.M * model.R
        model.Q[:] = model.γ * model.V
    end
end


function sim_blockwise_TD0TD1SRMB_twobeta_subject(βTD0_1, βTD0_2, βTD1_1, βTD1_2, βSR_1, βSR_2, βMB_1, βMB_2, βBoat, island_bias, boat_bias, αHome, αAway, αM, γ, s, use_sr_importance_sampling, sr_λ, c; seed=nothing, perfect_rewards=false)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    rewards_rec = zeros(4, 400)
    rec_ind = 1
    rewards = vcat(shuffle([shuffle([0.15, 0.85]), shuffle([0.325, 0.675])])...)
    policy_side_1 = rewards[1] > rewards[2] ? 1 : 2    
    policy_side_2 = rewards[3] > rewards[4] ? 1 : 2
    block_lens = shuffle([8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12])
    block_types = vcat(0, shuffle([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    env = make_env(rewards)
    model_TD0 = MFTDModel(env; α=αAway, λ=0.0, γ)
    model_TD1 = MFTDModel(env; α=αAway, λ=1.0, γ)
    model_SR = SRModel(env; α=αAway, αM, γ, λ=sr_λ, c)
    model_MB = MBModel(env; α=αAway, γ)


    # Initialize with perfect reward knowledge if requested
    if perfect_rewards
        # Convert rewards to the scale your models expect
        model_rewards = (rewards .- 0.5) .* 2  # Convert [0,1] to [-1,1] if needed
        
        # Initialize SR model with perfect rewards
        initialize_perfect_rewards!(model_SR, rewards)
        
        println("Initialized with perfect rewards: $(rewards)")
    end

    agent = TD0TD1SRMBBiasWeightedAgent(env, model_TD0, model_TD1, model_SR, model_MB, βTD0_1, βTD1_1, βSR_1, βMB_1, βBoat, island_bias, boat_bias, use_sr_importance_sampling)
    agentRecord = RLSR.Record(agent, sum(block_lens))
    episodeRecord = Vector{Episode}()
    rwd_swap_types = []
    for i in eachindex(block_lens)
        block_type = block_types[i]
        if (i == 1)
            agent.policy.βTD0 = βTD0_1
            agent.policy.βTD1 = βTD1_1
            agent.policy.βSR = βSR_1
            agent.policy.βMB = βMB_1
            push!(rwd_swap_types, repeat(["start"], block_lens[i]*2))
        elseif block_type == 0
            agent.policy.βTD0 = βTD0_1
            agent.policy.βTD1 = βTD1_1
            agent.policy.βSR = βSR_1
            agent.policy.βMB = βMB_1
            policy_side_1, policy_side_2 = onpolicy_swap!(rewards, policy_side_1, policy_side_2)
            push!(rwd_swap_types, repeat(["good"], block_lens[i]*2))
        elseif block_type == 1
            agent.policy.βTD0 = βTD0_2
            agent.policy.βTD1 = βTD1_2
            agent.policy.βSR = βSR_2
            agent.policy.βMB = βMB_2
            policy_side_1, policy_side_2 = offpolicy_swap!(rewards, policy_side_1, policy_side_2)
            push!(rwd_swap_types, repeat(["bad"], block_lens[i]*2))
        else
            agent.policy.βTD0 = βTD0_2
            agent.policy.βTD1 = βTD1_2
            agent.policy.βSR = βSR_2
            agent.policy.βMB = βMB_2
            policy_side_1, policy_side_2 = within_swap!(rewards, policy_side_1, policy_side_2)
            push!(rwd_swap_types, repeat(["within"], block_lens[i]*2))
        end
        update_rewards!(agent, rewards)
        trials = make_alternate_starts(block_lens[i]*2, [4,5,6,7])

        for trial in trials
            if trial == 1
                agent.model.TD0Model.α = αAway
                agent.model.TD1Model.α = αAway
                agent.model.SRModel.α = αAway
                agent.model.MBModel.α = αAway
            else
                agent.model.TD0Model.α = αHome
                agent.model.TD1Model.α = αHome
                agent.model.SRModel.α = αHome
                agent.model.MBModel.α = αHome
            end
            rewards_rec[:, rec_ind] .= rewards
            rec_ind += 1
            run_trials!(agent, [trial], agentRecord, episodeRecord)
        end
    end
    df = RunToDataFrame(episodeRecord, subject=s)
    df[!, "M_2_4"] .= agentRecord.M[:,2,4]
    df[!, "M_2_5"] .= agentRecord.M[:,2,5]
    df[!, "M_3_6"] .= agentRecord.M[:,3,6]
    df[!, "M_3_7"] .= agentRecord.M[:,3,7]
    df[!, "R_4"] .= (agentRecord.QTD0[:,4] .+ 1.0) ./ 2.0
    df[!, "R_5"] .= (agentRecord.QTD0[:,5] .+ 1.0) ./ 2.0
    df[!, "R_6"] .= (agentRecord.QTD0[:,6] .+ 1.0) ./ 2.0
    df[!, "R_7"] .= (agentRecord.QTD0[:,7] .+ 1.0) ./ 2.0

    df[!, "rwd_swap_type"] .= vcat(rwd_swap_types...)
    df[!, "p_reward_1"] .= rewards[1, :]
    df[!, "p_reward_2"] .= rewards[2, :]
    df[!, "p_reward_3"] .= rewards[3, :]
    df[!, "p_reward_4"] .= rewards[4, :]
    df.rewscaled .= df.reward
    df.reward .= (df.reward .+ 1.0) ./ 2.0
    df.rewardₜ₋₁ = (df.rewardₜ₋₁ ./ 2) .+ 0.5  # Set reward to 0/1
    df.endStateSiblingPriorRewardₜ₋₁ = (df.endStateSiblingPriorRewardₜ₋₁ ./ 2) .+ 0.5
    return df
end

function sim_blockwise_TD0TD1SRMB_subject(βTD0, βTD1, βSR, βMB, βBoat, island_bias, boat_bias, αHome, αAway, αM, γ, s, use_sr_importance_sampling, sr_λ, c; seed=nothing)
    return sim_blockwise_TD0TD1SRMB_twobeta_subject(βTD0, βTD0, βTD1, βTD1, βSR, βSR, βMB, βMB, βBoat, island_bias, boat_bias, αHome, αAway, αM, γ, s, use_sr_importance_sampling, sr_λ, c; seed)
end

# function sim_blockwise_LRL(;βLRL, βBoat, island_bias, boat_bias, α, αT, log_λ, σβLRL, σβBoat, σisland_bias, σboat_bias, σα, σαT, σlog_λ, c, nsims)
## Armin addition
function sim_blockwise_LRL(;βLRL, βBoat, α, αT, s_λ, σβLRL, σβBoat, σα, σαT, c, nsims)
    dfs = []
    for s in 1:nsims
        Random.seed!(s)
        rewards = vcat(shuffle([shuffle([0.15, 0.85]), shuffle([0.325, 0.675])])...)
        policy_side_1 = rewards[1] > rewards[2] ? 1 : 2    
        policy_side_2 = rewards[3] > rewards[4] ? 1 : 2
        block_lens = shuffle([8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12])
        block_types = vcat(0, shuffle([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        env = make_env(rewards)
        s_βLRL = rand(Normal(βLRL, σβLRL))
        s_βBoat = rand(Normal(βBoat, σβBoat))
        # s_island_bias = rand(Normal(island_bias, σisland_bias))
        # s_boat_bias = rand(Normal(boat_bias, σboat_bias))
        s_α = unitnorm(rand(Normal(α, σα)))
        s_αT = unitnorm(rand(Normal(αT, σαT)))
        # s_log_λ = rand(Normal(log_λ, σlog_λ))
        # s_λ = exp(s_log_λ)
        agent = LRLTwoStepSoftmax(env; α=s_α, αT=s_αT, λ=s_λ, c=c, β1=s_βLRL, β2=s_βBoat)
        agentRecord = RLSR.Record(agent, sum(block_lens))
        episodeRecord = Vector{Episode}()
        rwd_swap_types = []
        for i in 1:length(block_lens)
            block_type = block_types[i]
            if (i == 1)
                push!(rwd_swap_types, repeat(["start"], block_lens[i]*2))
            elseif block_type == 0
                policy_side_1, policy_side_2 = onpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["good"], block_lens[i]*2))
            elseif block_type == 1
                policy_side_1, policy_side_2 = offpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["bad"], block_lens[i]*2))
            else
                policy_side_1, policy_side_2 = within_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["within"], block_lens[i]*2))
            end
            update_rewards!(agent, rewards)
            trials = make_alternate_starts(block_lens[i]*2, [4,5,6,7])
            for trial in trials
                run_trials!(agent, [trial], agentRecord, episodeRecord)
            end
            # run_trials!(agent, block_lens[i]*2, agentRecord, episodeRecord)
        end
        df = RunToDataFrame(episodeRecord, subject=s)
        df[!, "M_2_4"] .= agentRecord.T[:,2,4]
        df[!, "M_2_5"] .= agentRecord.T[:,2,5]
        df[!, "M_3_6"] .= agentRecord.T[:,3,6]
        df[!, "M_3_7"] .= agentRecord.T[:,3,7]
        df[!, "R_4"] .= (agentRecord.R[:,4] .+ 1.0) ./ 2.0
        df[!, "R_5"] .= (agentRecord.R[:,5] .+ 1.0) ./ 2.0
        df[!, "R_6"] .= (agentRecord.R[:,6] .+ 1.0) ./ 2.0
        df[!, "R_7"] .= (agentRecord.R[:,7] .+ 1.0) ./ 2.0

        df[!, "rwd_swap_type"] .= vcat(rwd_swap_types...)
        df[!, "sub"] .= s
        df[!, "subject"] .= lpad(s, 3, "0")
        df.rewscaled .= df.reward
        df.reward .= (df.reward .+ 1.0) ./ 2.0
        df.rewardₜ₋₁ = (df.rewardₜ₋₁ ./ 2) .+ 0.5  # Set reward to 0/1
        df.endStateSiblingPriorRewardₜ₋₁ = (df.endStateSiblingPriorRewardₜ₋₁ ./ 2) .+ 0.5
        push!(dfs, df)
    end
    vcat(dfs...)
end

# function sim_blockwise_LRL(;βLRL, βBoat, island_bias, boat_bias, α, αT, log_λ, σβLRL, σβBoat, σisland_bias, σboat_bias, σα, σαT, σlog_λ, c, nsims)
#     dfs = []
#     for s in 1:nsims
#         Random.seed!(s)
#         rewards = vcat(shuffle([shuffle([0.15, 0.85]), shuffle([0.325, 0.675])])...)
#         policy_side_1 = rewards[1] > rewards[2] ? 1 : 2    
#         policy_side_2 = rewards[3] > rewards[4] ? 1 : 2
#         block_lens = shuffle([8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12])
#         block_types = vcat(0, shuffle([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
#         env = make_env(rewards)
#         s_βLRL = rand(Normal(βLRL, σβLRL))
#         s_βBoat = rand(Normal(βBoat, σβBoat))
#         # s_island_bias = rand(Normal(island_bias, σisland_bias))
#         # s_boat_bias = rand(Normal(boat_bias, σboat_bias))
#         s_α = unitnorm(rand(Normal(α, σα)))
#         s_αT = unitnorm(rand(Normal(αT, σαT)))
#         s_log_λ = rand(Normal(log_λ, σlog_λ))
#         s_λ = exp(s_log_λ)
#         agent = LRLTwoStepSoftmax(env; α=s_α, αT=s_αT, λ=s_λ, c=c, β1=s_βLRL, β2=s_βBoat)
#         agentRecord = RLSR.Record(agent, sum(block_lens))
#         episodeRecord = Vector{Episode}()
#         rwd_swap_types = []
#         for i in 1:length(block_lens)
#             block_type = block_types[i]
#             if (i == 1)
#                 push!(rwd_swap_types, repeat(["start"], block_lens[i]*2))
#             elseif block_type == 0
#                 policy_side_1, policy_side_2 = onpolicy_swap!(rewards, policy_side_1, policy_side_2)
#                 push!(rwd_swap_types, repeat(["good"], block_lens[i]*2))
#             elseif block_type == 1
#                 policy_side_1, policy_side_2 = offpolicy_swap!(rewards, policy_side_1, policy_side_2)
#                 push!(rwd_swap_types, repeat(["bad"], block_lens[i]*2))
#             else
#                 policy_side_1, policy_side_2 = within_swap!(rewards, policy_side_1, policy_side_2)
#                 push!(rwd_swap_types, repeat(["within"], block_lens[i]*2))
#             end
#             update_rewards!(agent, rewards)
#             run_trials!(agent, block_lens[i]*2, agentRecord, episodeRecord)
#         end
#         df = RunToDataFrame(episodeRecord, subject=s)
#         df[!, "M_2_4"] .= agentRecord.T[:,2,4]
#         df[!, "M_2_5"] .= agentRecord.T[:,2,5]
#         df[!, "M_3_6"] .= agentRecord.T[:,3,6]
#         df[!, "M_3_7"] .= agentRecord.T[:,3,7]
#         df[!, "R_4"] .= (agentRecord.R[:,4] .+ 1.0) ./ 2.0
#         df[!, "R_5"] .= (agentRecord.R[:,5] .+ 1.0) ./ 2.0
#         df[!, "R_6"] .= (agentRecord.R[:,6] .+ 1.0) ./ 2.0
#         df[!, "R_7"] .= (agentRecord.R[:,7] .+ 1.0) ./ 2.0

#         df[!, "rwd_swap_type"] .= vcat(rwd_swap_types...)
#         df[!, "sub"] .= s
#         df[!, "subject"] .= lpad(s, 3, "0")
#         df.rewscaled .= df.reward
#         df.reward .= (df.reward .+ 1.0) ./ 2.0
#         df.rewardₜ₋₁ = (df.rewardₜ₋₁ ./ 2) .+ 0.5  # Set reward to 0/1
#         df.endStateSiblingPriorRewardₜ₋₁ = (df.endStateSiblingPriorRewardₜ₋₁ ./ 2) .+ 0.5
#         push!(dfs, df)
#     end
#     vcat(dfs...)
# end

function sim_blockwise_TD1(;βTD1, βBoat, island_bias, boat_bias, α, αM, γ, σβTD1, σβBoat, σisland_bias, σboat_bias, σα, σαM, σγ, nsims)
    dfs = []
    for s in 1:nsims
        Random.seed!(s)
        rewards = vcat(shuffle([shuffle([0.15, 0.85]), shuffle([0.325, 0.675])])...)
        policy_side_1 = rewards[1] > rewards[2] ? 1 : 2    
        policy_side_2 = rewards[3] > rewards[4] ? 1 : 2
        block_lens = shuffle([8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12])
        block_types = vcat(0, shuffle([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        env = make_env(rewards)
        s_βTD1 = rand(Normal(βTD1, σβTD1))
        s_βBoat = rand(Normal(βBoat, σβBoat))
        # s_island_bias = rand(Normal(island_bias, σisland_bias))
        # s_boat_bias = rand(Normal(boat_bias, σboat_bias))
        s_α = unitnorm(rand(Normal(α, σα)))
        s_αM = unitnorm(rand(Normal(αM, σαM)))
        s_γ = unitnorm(rand(Normal(γ, σγ)))
        model_TD0 = MFTDModel(env; α=s_α, λ=0.0, γ=s_γ)
        model_TD1 = MFTDModel(env; α=s_α, λ=1.0, γ=s_γ)
        model_SR = SRModel(env; α=s_α, αM=s_αM, γ=s_γ, λ=1.0)
        model_MB = MBModel(env; α=s_α, γ=s_γ)
        agent = TD0TD1SRMBWeightedAgent(env, model_TD0, model_TD1, model_SR, model_MB, 0.0, s_βTD1, 0.0, 0.0, s_βBoat)
        agentRecord = RLSR.Record(agent, sum(block_lens))
        episodeRecord = Vector{Episode}()
        rwd_swap_types = []
        for i in 1:length(block_lens)
            block_type = block_types[i]
            if (i == 1)
                push!(rwd_swap_types, repeat(["start"], block_lens[i]*2))
            elseif block_type == 0
                policy_side_1, policy_side_2 = onpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["good"], block_lens[i]*2))
            elseif block_type == 1
                policy_side_1, policy_side_2 = offpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["bad"], block_lens[i]*2))
            else
                policy_side_1, policy_side_2 = within_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["within"], block_lens[i]*2))
            end
            update_rewards!(agent, rewards)
            run_trials!(agent, block_lens[i]*2, agentRecord, episodeRecord)
        end
        df = RunToDataFrame(episodeRecord, subject=s)
        df[!, "M_2_4"] .= agentRecord.M[:,2,4]
        df[!, "M_2_5"] .= agentRecord.M[:,2,5]
        df[!, "M_3_6"] .= agentRecord.M[:,3,6]
        df[!, "M_3_7"] .= agentRecord.M[:,3,7]
        df[!, "R_4"] .= (agentRecord.QTD0[:,4] .+ 1.0) ./ 2.0
        df[!, "R_5"] .= (agentRecord.QTD0[:,5] .+ 1.0) ./ 2.0
        df[!, "R_6"] .= (agentRecord.QTD0[:,6] .+ 1.0) ./ 2.0
        df[!, "R_7"] .= (agentRecord.QTD0[:,7] .+ 1.0) ./ 2.0

        df[!, "rwd_swap_type"] .= vcat(rwd_swap_types...)
        df[!, "sub"] .= s
        df[!, "subject"] .= lpad(s, 3, "0")
        df.rewscaled .= df.reward
        df.reward .= (df.reward .+ 1.0) ./ 2.0
        df.rewardₜ₋₁ = (df.rewardₜ₋₁ ./ 2) .+ 0.5  # Set reward to 0/1
        df.endStateSiblingPriorRewardₜ₋₁ = (df.endStateSiblingPriorRewardₜ₋₁ ./ 2) .+ 0.5
        push!(dfs, df)
    end
    vcat(dfs...)
end

function sim_blockwise_TD0(;βTD0, βBoat, island_bias, boat_bias, α, αM, γ, σβTD0, σβBoat, σisland_bias, σboat_bias, σα, σαM, σγ, nsims)
    dfs = []
    for s in 1:nsims
        Random.seed!(s)
        rewards = vcat(shuffle([shuffle([0.15, 0.85]), shuffle([0.325, 0.675])])...)
        policy_side_1 = rewards[1] > rewards[2] ? 1 : 2    
        policy_side_2 = rewards[3] > rewards[4] ? 1 : 2
        block_lens = shuffle([8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12])
        block_types = vcat(0, shuffle([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        env = make_env(rewards)
        s_βTD0 = rand(Normal(βTD0, σβTD0))
        s_βBoat = rand(Normal(βBoat, σβBoat))
        # s_island_bias = rand(Normal(island_bias, σisland_bias))
        # s_boat_bias = rand(Normal(boat_bias, σboat_bias))
        s_α = unitnorm(rand(Normal(α, σα)))
        s_αM = unitnorm(rand(Normal(αM, σαM)))
        s_γ = unitnorm(rand(Normal(γ, σγ)))
        model_TD0 = MFTDModel(env; α=s_α, λ=0.0, γ=s_γ)
        model_TD1 = MFTDModel(env; α=s_α, λ=1.0, γ=s_γ)
        model_SR = SRModel(env; α=s_α, αM=s_αM, γ=s_γ, λ=1.0)
        model_MB = MBModel(env; α=s_α, γ=s_γ)
        agent = TD0TD1SRMBWeightedAgent(env, model_TD0, model_TD1, model_SR, model_MB, s_βTD0, 0.0, 0.0, 0.0, s_βBoat)
        agentRecord = RLSR.Record(agent, sum(block_lens))
        episodeRecord = Vector{Episode}()
        rwd_swap_types = []
        for i in 1:length(block_lens)
            block_type = block_types[i]
            if (i == 1)
                push!(rwd_swap_types, repeat(["start"], block_lens[i]*2))
            elseif block_type == 0
                policy_side_1, policy_side_2 = onpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["good"], block_lens[i]*2))
            elseif block_type == 1
                policy_side_1, policy_side_2 = offpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["bad"], block_lens[i]*2))
            else
                policy_side_1, policy_side_2 = within_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["within"], block_lens[i]*2))
            end
            update_rewards!(agent, rewards)
            run_trials!(agent, block_lens[i]*2, agentRecord, episodeRecord)
        end
        df = RunToDataFrame(episodeRecord, subject=s)
        df[!, "M_2_4"] .= agentRecord.M[:,2,4]
        df[!, "M_2_5"] .= agentRecord.M[:,2,5]
        df[!, "M_3_6"] .= agentRecord.M[:,3,6]
        df[!, "M_3_7"] .= agentRecord.M[:,3,7]
        df[!, "R_4"] .= (agentRecord.QTD0[:,4] .+ 1.0) ./ 2.0
        df[!, "R_5"] .= (agentRecord.QTD0[:,5] .+ 1.0) ./ 2.0
        df[!, "R_6"] .= (agentRecord.QTD0[:,6] .+ 1.0) ./ 2.0
        df[!, "R_7"] .= (agentRecord.QTD0[:,7] .+ 1.0) ./ 2.0

        df[!, "rwd_swap_type"] .= vcat(rwd_swap_types...)
        df[!, "sub"] .= s
        df[!, "subject"] .= lpad(s, 3, "0")
        df.rewscaled .= df.reward
        df.reward .= (df.reward .+ 1.0) ./ 2.0
        df.rewardₜ₋₁ = (df.rewardₜ₋₁ ./ 2) .+ 0.5  # Set reward to 0/1
        df.endStateSiblingPriorRewardₜ₋₁ = (df.endStateSiblingPriorRewardₜ₋₁ ./ 2) .+ 0.5
        push!(dfs, df)
    end
    vcat(dfs...)
end

function sim_blockwise_MB(;βMB, βBoat, island_bias, boat_bias, α, αM, γ, σβMB, σβBoat, σisland_bias, σboat_bias, σα, σαM, σγ, nsims)
    dfs = []
    for s in 1:nsims
        Random.seed!(s)
        rewards = vcat(shuffle([shuffle([0.15, 0.85]), shuffle([0.325, 0.675])])...)
        policy_side_1 = rewards[1] > rewards[2] ? 1 : 2    
        policy_side_2 = rewards[3] > rewards[4] ? 1 : 2
        block_lens = shuffle([8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12])
        block_types = vcat(0, shuffle([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        env = make_env(rewards)
        s_βMB = rand(Normal(βMB, σβMB))
        s_βBoat = rand(Normal(βBoat, σβBoat))
        # s_island_bias = rand(Normal(island_bias, σisland_bias))
        # s_boat_bias = rand(Normal(boat_bias, σboat_bias))
        s_α = unitnorm(rand(Normal(α, σα)))
        s_αM = unitnorm(rand(Normal(αM, σαM)))
        s_γ = unitnorm(rand(Normal(γ, σγ)))
        model_TD0 = MFTDModel(env; α=s_α, λ=0.0, γ=s_γ)
        model_TD1 = MFTDModel(env; α=s_α, λ=1.0, γ=s_γ)
        model_SR = SRModel(env; α=s_α, αM=s_αM, γ=s_γ, λ=1.0)
        model_MB = MBModel(env; α=s_α, γ=s_γ)
        agent = TD0TD1SRMBWeightedAgent(env, model_TD0, model_TD1, model_SR, model_MB, 0.0, 0.0, 0.0, s_βMB, s_βBoat)
        agentRecord = RLSR.Record(agent, sum(block_lens))
        episodeRecord = Vector{Episode}()
        rwd_swap_types = []
        for i in 1:length(block_lens)
            block_type = block_types[i]
            if (i == 1)
                push!(rwd_swap_types, repeat(["start"], block_lens[i]*2))
            elseif block_type == 0
                policy_side_1, policy_side_2 = onpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["good"], block_lens[i]*2))
            elseif block_type == 1
                policy_side_1, policy_side_2 = offpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["bad"], block_lens[i]*2))
            else
                policy_side_1, policy_side_2 = within_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["within"], block_lens[i]*2))
            end
            update_rewards!(agent, rewards)
            run_trials!(agent, block_lens[i]*2, agentRecord, episodeRecord)
        end
        df = RunToDataFrame(episodeRecord, subject=s)
        df[!, "M_2_4"] .= agentRecord.M[:,2,4]
        df[!, "M_2_5"] .= agentRecord.M[:,2,5]
        df[!, "M_3_6"] .= agentRecord.M[:,3,6]
        df[!, "M_3_7"] .= agentRecord.M[:,3,7]
        df[!, "R_4"] .= (agentRecord.QTD0[:,4] .+ 1.0) ./ 2.0
        df[!, "R_5"] .= (agentRecord.QTD0[:,5] .+ 1.0) ./ 2.0
        df[!, "R_6"] .= (agentRecord.QTD0[:,6] .+ 1.0) ./ 2.0
        df[!, "R_7"] .= (agentRecord.QTD0[:,7] .+ 1.0) ./ 2.0

        df[!, "rwd_swap_type"] .= vcat(rwd_swap_types...)
        df[!, "sub"] .= s
        df[!, "subject"] .= lpad(s, 3, "0")
        df.rewscaled .= df.reward
        df.reward .= (df.reward .+ 1.0) ./ 2.0
        df.rewardₜ₋₁ = (df.rewardₜ₋₁ ./ 2) .+ 0.5  # Set reward to 0/1
        df.endStateSiblingPriorRewardₜ₋₁ = (df.endStateSiblingPriorRewardₜ₋₁ ./ 2) .+ 0.5
        push!(dfs, df)
    end
    vcat(dfs...)
end

function sim_blockwise_SR(;βSR, βBoat, island_bias, boat_bias, α, αM, γ, σβSR, σβBoat, σisland_bias, σboat_bias, σα, σαM, σγ, nsims)
    dfs = []
    for s in 1:nsims
        Random.seed!(s)
        rewards = vcat(shuffle([shuffle([0.15, 0.85]), shuffle([0.325, 0.675])])...)
        policy_side_1 = rewards[1] > rewards[2] ? 1 : 2    
        policy_side_2 = rewards[3] > rewards[4] ? 1 : 2
        block_lens = shuffle([8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12])
        block_types = vcat(0, shuffle([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        env = make_env(rewards)
        s_βSR = rand(Normal(βSR, σβSR))
        s_βBoat = rand(Normal(βBoat, σβBoat))
        # s_island_bias = rand(Normal(island_bias, σisland_bias))
        # s_boat_bias = rand(Normal(boat_bias, σboat_bias))
        s_α = unitnorm(rand(Normal(α, σα)))
        s_αM = unitnorm(rand(Normal(αM, σαM)))
        s_γ = unitnorm(rand(Normal(γ, σγ)))
        model_TD0 = MFTDModel(env; α=s_α, λ=0.0, γ=s_γ)
        model_TD1 = MFTDModel(env; α=s_α, λ=1.0, γ=s_γ)
        model_SR = SRModel(env; α=s_α, αM=s_αM, γ=s_γ, λ=1.0)
        model_MB = MBModel(env; α=s_α, γ=s_γ)
        agent = TD0TD1SRMBWeightedAgent(env, model_TD0, model_TD1, model_SR, model_MB, 0.0, 0.0, s_βSR, 0.0, s_βBoat)
        agentRecord = RLSR.Record(agent, sum(block_lens))
        episodeRecord = Vector{Episode}()
        rwd_swap_types = []
        for i in 1:length(block_lens)
            block_type = block_types[i]
            if (i == 1)
                push!(rwd_swap_types, repeat(["start"], block_lens[i]*2))
            elseif block_type == 0
                policy_side_1, policy_side_2 = onpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["good"], block_lens[i]*2))
            elseif block_type == 1
                policy_side_1, policy_side_2 = offpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["bad"], block_lens[i]*2))
            else
                policy_side_1, policy_side_2 = within_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["within"], block_lens[i]*2))
            end
            update_rewards!(agent, rewards)
            run_trials!(agent, block_lens[i]*2, agentRecord, episodeRecord)
        end
        df = RunToDataFrame(episodeRecord, subject=s)
        df[!, "M_2_4"] .= agentRecord.M[:,2,4]
        df[!, "M_2_5"] .= agentRecord.M[:,2,5]
        df[!, "M_3_6"] .= agentRecord.M[:,3,6]
        df[!, "M_3_7"] .= agentRecord.M[:,3,7]
        df[!, "R_4"] .= (agentRecord.QTD0[:,4] .+ 1.0) ./ 2.0
        df[!, "R_5"] .= (agentRecord.QTD0[:,5] .+ 1.0) ./ 2.0
        df[!, "R_6"] .= (agentRecord.QTD0[:,6] .+ 1.0) ./ 2.0
        df[!, "R_7"] .= (agentRecord.QTD0[:,7] .+ 1.0) ./ 2.0

        df[!, "rwd_swap_type"] .= vcat(rwd_swap_types...)
        df[!, "sub"] .= s
        df[!, "subject"] .= lpad(s, 3, "0")
        df.rewscaled .= df.reward
        df.reward .= (df.reward .+ 1.0) ./ 2.0
        df.rewardₜ₋₁ = (df.rewardₜ₋₁ ./ 2) .+ 0.5  # Set reward to 0/1
        df.endStateSiblingPriorRewardₜ₋₁ = (df.endStateSiblingPriorRewardₜ₋₁ ./ 2) .+ 0.5
        push!(dfs, df)
    end
    vcat(dfs...)
end

function sim_blockwise_LRL_twoλ(;βLRL, βBoat, island_bias, boat_bias, α, αT, log_λ1, log_λ2, σβLRL, σβBoat, σisland_bias, σboat_bias, σα, σαT, σlog_λ1, σlog_λ2, nsims, c=0.1)
    dfs = []
    for s in 1:nsims
        Random.seed!(s)
        rewards = vcat(shuffle([shuffle([0.15, 0.85]), shuffle([0.325, 0.675])])...)
        policy_side_1 = rewards[1] > rewards[2] ? 1 : 2    
        policy_side_2 = rewards[3] > rewards[4] ? 1 : 2
        block_lens = shuffle([8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 12])
        block_types = vcat(0, shuffle([1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
        env = make_env(rewards)
        s_βLRL = rand(Normal(βLRL, σβLRL))
        s_βBoat = rand(Normal(βBoat, σβBoat))
        # s_island_bias = rand(Normal(island_bias, σisland_bias))
        # s_boat_bias = rand(Normal(boat_bias, σboat_bias))
        s_α = unitnorm(rand(Normal(α, σα)))
        s_αT = unitnorm(rand(Normal(αT, σαT)))
        s_log_λ1 = rand(Normal(log_λ1, σlog_λ1))
        s_log_λ2 = rand(Normal(log_λ2, σlog_λ2))
        s_λ1 = exp(s_log_λ1)
        s_λ2 = exp(s_log_λ2)
        c = c
        # agent = TD0TD1SRMBWeightedAgent(env, TD0, TD1, SR, MB, βTD0, βTD1, βSR, βMB, βBoat)
        agent = LRLTwoStepSoftmax(env; α=s_α, αT=s_αT, λ=s_λ1, c=c, β1=s_βLRL, β2=s_βBoat)
        agentRecord = RLSR.Record(agent, sum(block_lens))
        episodeRecord = Vector{Episode}()
        rwd_swap_types = []
        for i in 1:length(block_lens)
            block_type = block_types[i]
            if (i == 1)
                push!(rwd_swap_types, repeat(["start"], block_lens[i]*2))
                agent.model.λ = s_λ1
            elseif block_type == 0
                policy_side_1, policy_side_2 = onpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["good"], block_lens[i]*2))
                agent.model.λ = s_λ1
            elseif block_type == 1
                policy_side_1, policy_side_2 = offpolicy_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["bad"], block_lens[i]*2))
                agent.model.λ = s_λ2
            else
                policy_side_1, policy_side_2 = within_swap!(rewards, policy_side_1, policy_side_2)
                push!(rwd_swap_types, repeat(["within"], block_lens[i]*2))
                agent.model.λ = s_λ2
            end
            update_rewards!(agent, rewards)
            run_trials!(agent, block_lens[i]*2, agentRecord, episodeRecord)
        end
        df = RunToDataFrame(episodeRecord, subject=s)
        df[!, "M_2_4"] .= agentRecord.T[:,2,4]
        df[!, "M_2_5"] .= agentRecord.T[:,2,5]
        df[!, "M_3_6"] .= agentRecord.T[:,3,6]
        df[!, "M_3_7"] .= agentRecord.T[:,3,7]
        df[!, "R_4"] .= (agentRecord.R[:,4] .+ 1.0) ./ 2.0
        df[!, "R_5"] .= (agentRecord.R[:,5] .+ 1.0) ./ 2.0
        df[!, "R_6"] .= (agentRecord.R[:,6] .+ 1.0) ./ 2.0
        df[!, "R_7"] .= (agentRecord.R[:,7] .+ 1.0) ./ 2.0

        df[!, "rwd_swap_type"] .= vcat(rwd_swap_types...)
        df[!, "sub"] .= s
        df.rewscaled .= df.reward
        df.reward .= (df.reward .+ 1.0) ./ 2.0
        df.rewardₜ₋₁ = (df.rewardₜ₋₁ ./ 2) .+ 0.5  # Set reward to 0/1
        df.endStateSiblingPriorRewardₜ₋₁ = (df.endStateSiblingPriorRewardₜ₋₁ ./ 2) .+ 0.5
        push!(dfs, df)
    end
    vcat(dfs...)
end

function filter_df(df)
    df_reg = copy(df)

    df_reg = @chain df_reg begin
        @subset(:trial .> 1) # Ignore the first trial (no previous state info)
        @subset(:state1ₜ₋₁ .> 3) # End states
        dropmissing(:parentPriorMoveToEndStateₜ₋₁)
        dropmissing(:endStateSiblingPriorRewardₜ₋₁)
        dropmissing(:rewardₜ₋₁)
        @transform(:endStateSiblingPriorNoRewardₜ₋₁ = :endStateSiblingPriorRewardₜ₋₁ .== 0)
    end

    return df_reg
end


"""
Add a regressor for previous boat-only trials

For the last 10 boat-only traversals
"""
function add_boatonly_island_reg!(df)
    prev_island_outcomes = Array{Union{Missing, Bool}}(missing, 11, 2)
    for i in 1:10
        df[:, "lag$(i)_boatonly_same_reg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
        df[:, "lag$(i)_boatonly_opp_reg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_island_outcomes .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            if !ismissing(df[t, :endBranchLeftₜ₋₁])
                island = df[t, :endBranchLeftₜ₋₁] ? 1 : 2
                opp_island = df[t, :endBranchLeftₜ₋₁] ? 2 : 1
                for i in 1:10
                    df[t, "lag$(i)_boatonly_same_reg"] = prev_island_outcomes[i+1, island]
                    df[t, "lag$(i)_boatonly_opp_reg"] = !prev_island_outcomes[i+1, opp_island]
                end
            end
        else
            island = df[t, :endBranchLeft] ? 1 : 2
            reward = df[t, :reward] .> 0
            for i in 11:-1:2
                prev_island_outcomes[i, island] = prev_island_outcomes[i-1, island] 
            end
            prev_island_outcomes[1, island] = reward
        end
    end
end

"""
Previous N reward outcomes at the neighbor of the current boat.
So if we're currently at boat 2, last 10 reward outcomes at boat 1.

SR shows MB effect because neighbor is correlated w/ policy.
See how last N trials at neighbor influence current choice

Note that lag1 should be the same as our current regressor
"""
function add_lags_at_neighborboat!(df)
    prev_boat_outcomes = Array{Union{Missing, Bool}}(missing, 10, 4)
    for i in 1:10
        df[:, "lag$(i)_neighborboat_reg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_boat_outcomes .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            boat = df[t, :state3] - 3
            reward = df[t, :reward] .> 0
            if !ismissing(df[t, :endStateₜ₋₁])
                prev_boat = df[t, :endStateₜ₋₁] - 3
                sib = 0
                if prev_boat == 1
                    sib = 2
                elseif prev_boat == 2
                    sib = 1
                elseif prev_boat == 3
                    sib = 4
                elseif prev_boat == 4
                    sib = 3
                end
                for i in 1:10
                    df[t, "lag$(i)_neighborboat_reg"] = !prev_boat_outcomes[i, sib]
                end
            end
            for i in 10:-1:2
                prev_boat_outcomes[i, boat] = prev_boat_outcomes[i-1, boat] 
            end
            prev_boat_outcomes[1, boat] = reward
        else
            boat = df[t, :state1] - 3
            reward = df[t, :reward] .> 0
            for i in 10:-1:2
                prev_boat_outcomes[i, boat] = prev_boat_outcomes[i-1, boat] 
            end
            prev_boat_outcomes[1, boat] = reward
        end
    end
end

"""
Lagged regressor for SR interation
"""
function add_lags_policy!(df)
    prev_boat_choices = Array{Union{Missing, Int}}(missing, 11, 2)
    for i in 1:10
        df[:, "lag$(i)_policy_reg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_boat_choices .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            if !ismissing(df[t, :endStateₜ₋₁])
                prev_branch = df[t, :endBranchLeftₜ₋₁] ? 1 : 2
                for i in 1:10
                    df[t, "lag$(i)_policy_reg"] = prev_boat_choices[i, prev_branch] .== df[t, :endStateₜ₋₁]
                end
            end
            branch = df[t, :endBranchLeft] ? 1 : 2
            for i in 11:-1:2
                prev_boat_choices[i, branch] = prev_boat_choices[i-1, branch] 
            end
            prev_boat_choices[1, branch] = df[t, :endState]
        end
    end
end

"""
Last N boat trials, +/- outcomes support current choice
"""
function add_lags_boattrial_choicereg!(df)
    prev_boat_outcomes = Array{Union{Missing, Bool}}(missing, 11)
    for i in 1:10
        df[:, "lag$(i)_boattrial_choicereg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_boat_outcomes .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            if !ismissing(df[t, :endBranchLeftₜ₋₁])
                prev_branch = df[t, :endBranchLeftₜ₋₁]
                for i in 1:10
                    df[t, "lag$(i)_boattrial_choicereg"] = prev_boat_outcomes[i+1] .== prev_branch
                end
            end
        else
            reward = df[t, :reward] .> 0
            branch = df[t, :endBranchLeft]
            for i in 11:-1:2
                prev_boat_outcomes[i] = prev_boat_outcomes[i-1] 
            end
            prev_boat_outcomes[1] = (branch & reward) | (!branch & !reward)
        end
    end
end
function add_lags_traversaltrial_choicereg!(df)
    prev_traversal_outcomes = Array{Union{Missing, Bool}}(missing, 11)
    for i in 1:10
        df[:, "lag$(i)_traversaltrial_choicereg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_traversal_outcomes .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            if !ismissing(df[t, :endBranchLeftₜ₋₁])
                prev_branch = df[t, :endBranchLeftₜ₋₁]
                for i in 1:10
                    df[t, "lag$(i)_traversaltrial_choicereg"] = prev_traversal_outcomes[i+1] .== prev_branch
                end
            end
            reward = df[t, :reward] .> 0
            branch = df[t, :endBranchLeft]
            for i in 11:-1:2
                prev_traversal_outcomes[i] = prev_traversal_outcomes[i-1] 
            end
            prev_traversal_outcomes[1] = (branch & reward) | (!branch & !reward)
        end
    end
end

function add_lags_at_sameboat!(df)
    prev_boat_outcomes = Array{Union{Missing, Bool}}(missing, 11, 4)
    for i in 1:10
        df[:, "lag$(i)_sameboat_reg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_boat_outcomes .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            boat = df[t, :state3] - 3
            reward = df[t, :reward] .> 0
            if !ismissing(df[t, :endStateₜ₋₁])
                prev_boat = df[t, :endStateₜ₋₁] - 3
                for i in 1:10
                    df[t, "lag$(i)_sameboat_reg"] = prev_boat_outcomes[i+1, prev_boat]
                end
            end
            for i in 11:-1:2
                prev_boat_outcomes[i, boat] = prev_boat_outcomes[i-1, boat] 
            end
            prev_boat_outcomes[1, boat] = reward
        else
            boat = df[t, :state1] - 3
            reward = df[t, :reward] .> 0
            for i in 11:-1:2
                prev_boat_outcomes[i, boat] = prev_boat_outcomes[i-1, boat] 
            end
            prev_boat_outcomes[1, boat] = reward
        end
    end
end

function add_lags_at_boats_oppisland!(df)
    prev_boat_outcomes = Array{Union{Missing, Bool}}(missing, 10, 4)
    for i in 1:10
        df[:, "lag$(i)_oppislandsameboat_reg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
        df[:, "lag$(i)_oppislandoppboat_reg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
        df[:, "lag$(i)_oppislandavgboat_reg"] = Vector{Union{Missing, Float64}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_boat_outcomes .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            boat = df[t, :state3] - 3
            reward = df[t, :reward] .> 0
            if !ismissing(df[t, :endStateₜ₋₁])
                prevboat = df[t, :endStateₜ₋₁] - 3
                if prevboat == 1
                    oppislandsameboat = 3
                    oppislandoppboat = 4
                elseif prevboat == 2
                    oppislandsameboat = 4
                    oppislandoppboat = 3
                elseif prevboat == 3
                    oppislandsameboat = 1
                    oppislandoppboat = 2
                elseif prevboat == 4
                    oppislandsameboat = 2
                    oppislandoppboat = 1
                end
                for i in 1:10
                    df[t, "lag$(i)_oppislandsameboat_reg"] = prev_boat_outcomes[i, oppislandsameboat]
                    df[t, "lag$(i)_oppislandoppboat_reg"] = prev_boat_outcomes[i, oppislandoppboat]
                    df[t, "lag$(i)_oppislandavgboat_reg"] = mean([prev_boat_outcomes[i, oppislandsameboat], prev_boat_outcomes[i, oppislandoppboat]])
                end
            end
            for i in 10:-1:2
                prev_boat_outcomes[i, boat] = prev_boat_outcomes[i-1, boat] 
            end
            prev_boat_outcomes[1, boat] = reward
        else
            boat = df[t, :state1] - 3
            reward = df[t, :reward] .> 0
            for i in 10:-1:2
                prev_boat_outcomes[i, boat] = prev_boat_outcomes[i-1, boat] 
            end
            prev_boat_outcomes[1, boat] = reward
        end
    end
end

"""Island choice autorgressive behavior"""
function add_lags_choice_autoreg!(df)
    prev_island_choices = Array{Union{Missing, Bool}}(missing, 10)
    for i in 1:10
        df[:, "lag$(i)_choice_autoreg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_island_choices .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            island_choice = df[t, :endBranchLeft]
            prev_branch = df[t, :endBranchLeftₜ₋₁]
            for i in 1:10
                df[t, "lag$(i)_choice_autoreg"] = prev_island_choices[i] .== prev_branch
            end
            for i in 10:-1:2
                prev_island_choices[i] = prev_island_choices[i-1] 
            end
            prev_island_choices[1] = island_choice
        end
    end
end

function add_lags_boat_autoreg!(df)
    prev_branches = Array{Union{Missing, Bool}}(missing, 11)
    for i in 1:10
        df[:, "lag$(i)_boat_autoreg"] = Vector{Union{Missing, Bool}}(missing, nrow(df))
    end
    prev_trial = -1
    for t in 1:nrow(df)
        if df[t, :trial] < prev_trial
            prev_trial = -1
            prev_branches .= missing
        end
        prev_trial = df[t, :trial]

        if df[t, :state1] == 1
            for i in 1:10
                df[t, "lag$(i)_boat_autoreg"] = prev_branches[i+1] .== prev_branches[1]
            end
        else 
            branch = df[t, :endBranchLeft]
            for i in 11:-1:2
                prev_branches[i] = prev_branches[i-1] 
            end
            prev_branches[1] = branch
        end
    end
end

function add_all_regressors!(df)
    add_lags_at_boats_oppisland!(df)
    add_lags_choice_autoreg!(df)
    add_lags_boat_autoreg!(df)
    add_lags_at_sameboat!(df)
    add_lags_at_neighborboat!(df)
    add_lags_policy!(df)
    add_boatonly_island_reg!(df)
    add_lags_boattrial_choicereg!(df)
    add_lags_traversaltrial_choicereg!(df)
end