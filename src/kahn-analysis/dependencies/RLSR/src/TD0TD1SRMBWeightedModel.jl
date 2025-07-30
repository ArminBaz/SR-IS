struct TD0TD1SRMBWeightedStateModel <: AbstractWeightedStateModel
    TD0Model::MFTDModel
    TD1Model::MFTDModel
    SRModel::SRModel
    MBModel::MBModel

    V::Vector{Float64}
    QTD0::Vector{Float64}
    QTD1::Vector{Float64}
    QSR::Vector{Float64}
    QMB::Vector{Float64}
    use_sr_importance_sampling::Bool
end
function TD0TD1SRMBWeightedStateModel(TD0Model, TD1Model, SRModel, MBModel, use_sr_importance_sampling=false)
    TD0TD1SRMBWeightedStateModel(
        TD0Model, TD1Model, SRModel, MBModel,
        zeros(length(TD0Model.V)),
        zeros(length(TD0Model.V)),
        zeros(length(TD0Model.V)),
        zeros(length(TD0Model.V)),
        zeros(length(TD0Model.V)),
        use_sr_importance_sampling
    )
    
end
function model_name(agent::StateAgent{E, M, P}; kwargs...) where {E, M <: TD0TD1SRMBWeightedStateModel, P <: PolicyTD0TD1SRMBTwoStepSoftmax}
    policy_str = ": [βTD0: $(agent.policy.βTD0) βTD1: $(agent.policy.βTD1) βSR: $(agent.policy.βSR) βMB: $(agent.policy.βMB)] βBoat: $(agent.policy.βBoat)]"
    model_name(agent.model; kwargs...) .* policy_str
end
function model_name(model::M; kwargs...) where {M <: TD0TD1SRMBWeightedStateModel}
    "TD0TD1SRMBWeightedStateModel"
end

function TD0TD1SRMBWeightedAgent(env, TD0Model, TD1Model, SRModel, MBModel, βTD0, βTD1, βSR, βMB, βBoat, use_sr_importance_sampling=false)
    model = TD0TD1SRMBWeightedStateModel(TD0Model, TD1Model, SRModel, MBModel, use_sr_importance_sampling)
    policy = PolicyTD0TD1SRMBTwoStepSoftmax(βTD0, βTD1, βSR, βMB, βBoat)
    StateAgent(env, model, policy)
end

function TD0TD1SRMBBiasWeightedAgent(env, TD0Model, TD1Model, SRModel, MBModel, βTD0, βTD1, βSR, βMB, βBoat, Bias1, Bias2, use_sr_importance_sampling=false)
    model = TD0TD1SRMBWeightedStateModel(TD0Model, TD1Model, SRModel, MBModel, use_sr_importance_sampling)
    policy = PolicyTD0TD1SRMBBiasTwoStepSoftmax(βTD0, βTD1, βSR, βMB, βBoat, Bias1, Bias2)
    StateAgent(env, model, policy)
end

####################     ARMIN ADDITION     ####################
# Helper function to calculate action probability from policy
function calculate_action_probability(policy::PolicyTD0TD1SRMBTwoStepSoftmax, model::TD0TD1SRMBWeightedStateModel, s::Int, s′::Int, neighbors::Vector{Int})
    if s == 1
        neighbor_values = exp.(policy.βTD0 * model.QTD0[neighbors] + 
                             policy.βTD1 * model.QTD1[neighbors] + 
                             policy.βSR * model.QSR[neighbors] + 
                             policy.βMB * model.QMB[neighbors])
    else
        neighbor_values = exp.(policy.βBoat * model.QTD0[neighbors])
    end
    
    softmax_probs = neighbor_values ./ sum(neighbor_values)
    action_idx = findfirst(neighbors .== s′)
    return softmax_probs[action_idx]
end

function calculate_action_probability(policy::PolicyTD0TD1SRMBBiasTwoStepSoftmax, model::TD0TD1SRMBWeightedStateModel, s::Int, s′::Int, neighbors::Vector{Int})
    if s == 1
        neighbor_values = exp.(policy.βTD0 * model.QTD0[neighbors] + 
                             policy.βTD1 * model.QTD1[neighbors] + 
                             policy.βSR * model.QSR[neighbors] + 
                             policy.βMB * model.QMB[neighbors])
        
        if policy.Prev1 > 0
            prev_idx = findfirst(neighbors .== policy.Prev1)
            if !isnothing(prev_idx)
                neighbor_values[prev_idx] += policy.Bias1
            end
        end
    else
        neighbor_values = exp.(policy.βBoat * model.QTD0[neighbors])
        
        if s == 2 && policy.Prev22 > 0
            prev_idx = findfirst(neighbors .== policy.Prev22)
            if !isnothing(prev_idx)
                neighbor_values[prev_idx] += policy.Bias1
            end
        elseif s == 3 && policy.Prev23 > 0
            prev_idx = findfirst(neighbors .== policy.Prev23)
            if !isnothing(prev_idx)
                neighbor_values[prev_idx] += policy.Bias1
            end
        end
    end
    
    softmax_probs = neighbor_values ./ sum(neighbor_values)
    action_idx = findfirst(neighbors .== s′)
    return softmax_probs[action_idx]
end

# Fallback for other policy types (returns uniform probability)
function calculate_action_probability(policy, model, s::Int, s′::Int, neighbors::Vector{Int})
    return 1.0 / length(neighbors)
end

# Helper function to update SR with importance sampling
function update_sr_with_importance_sampling!(SR, s::Int, reward::Real, s′::Int, action_prob::Float64, neighbors::Vector{Int}, agent_policy, agent_model, max_weight::Float64=10.0)
    #  Non-traversal trial
    # if s ∈ [4, 5, 6, 7]
    #     if s in [4,5]
    #         curr_state = 2
    #         new_neighbors = [4, 5]
    #     elseif s in [6, 7]
    #         curr_state = 3
    #         new_neighbors = [6, 7]
    #     end
    #     next_state = s′-4
    #     new_action_prob = calculate_action_probability(agent_policy, agent_model, curr_state, next_state, new_neighbors)
    #     action_prob = new_action_prob
    #     neighbors = new_neighbors
    # end

    ##  Calculate importance sampling weight
    n_actions = length(neighbors)
    default_policy_prob = 1.0 / n_actions
    
    # # Avoid division by zero
    action_prob = max(action_prob, 1e-10)  # Minimum probability threshold
    
    w_raw = default_policy_prob / action_prob
    w = min(w_raw, max_weight)  # Clip the weight
    # w = 1

    ##  TD Update
    target = SR.ident[s, :] + SR.γ .* SR.DR[s′, :]
    # δM = target - SR.M[s, :]                             ## No IS
    # δM_weighted = w .* target - SR.DR[s, :]              ## IS

    #  TRACE UPDATE
    # SR.trace[:] = SR.trace .* (SR.γ * SR.λ)
    # SR.trace[s] = SR.trace[s] + 1
    # for sx in 1:length(SR)
    #     # SR.M[sx, :] = SR.M[sx, :] + SR.αM * SR.trace[sx] .* δM               ## No IS
    #     SR.M[sx, :] = SR.M[sx, :] + SR.αM * SR.trace[sx] .* δM_weighted      ## IS
    # end

    #  NO TRACE LOOP (1- α)*DR[s] + α*DR[s]*target*w
    SR.DR[s, :] = (1 - SR.αDR) .* SR.DR[s, :] .+ SR.αDR .* target .* w

    ##  Update R
    # SR.R[s] = (SR.α * reward) + ((1 - SR.α) * SR.R[s])
    # Only update if s′ is terminal
    # if s′ ∈ [8, 9, 10, 11]
    #     SR.R[s′] = (SR.α * reward) + ((1 - SR.α) * SR.R[s])
    # end
    if s in [8, 9, 10, 11]
        SR.R[s] = (SR.α * reward) + ((1 - SR.α) * SR.R[s])
        # SR.R[s] = reward
    end
    if s′ in [8, 9, 10, 11]
        SR.R[s′] = (SR.α * reward) + ((1 - SR.α) * SR.R[s′])
        # SR.R[s′] = reward
    end

    ##  Update Values
    SR.e_R = exp.(SR.R[SR.terminals]./SR.λ)

    SR.z_hat .= SR.DR * SR.P
    SR.e_V[SR.nonterminals] .= SR.z_hat[SR.nonterminals, :] * SR.e_R
    SR.e_V[SR.terminals] .= SR.e_R

    SR.V .= log.(SR.e_V) .* SR.λ
    SR.Q .= SR.V
    # println("update")
    # println(SR.e_V)
    # println(SR.V)

    # Update V and Q
    # SR.V = SR.Z
    # SR.Q[:] = SR.γ * SR.V
end

# Modified update_model_step! function with optional importance sampling
function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: TD0TD1SRMBWeightedStateModel, P}
    # println("s: $(s) | s′:$(s′)")
    # Update TD0, TD1, and MB models normally
    # update_model_step!(StateAgent(agent.env, agent.model.TD0Model, agent.policy), s, reward, s′)
    # update_model_step!(StateAgent(agent.env, agent.model.TD1Model, agent.policy), s, reward, s′)
    # update_model_step!(StateAgent(agent.env, agent.model.MBModel, agent.policy), s, reward, s′)

    # Update SR model with optional importance sampling
    if agent.model.use_sr_importance_sampling
        # Calculate action probability and apply importance sampling
        neighbors = find_neighbors(agent.env, s)
        action_prob = calculate_action_probability(agent.policy, agent.model, s, s′, neighbors)
        update_sr_with_importance_sampling!(agent.model.SRModel, s, reward, s′, action_prob, neighbors, agent.policy, agent.model)
    else
        # Regular SR update
        update_model_step!(StateAgent(agent.env, agent.model.SRModel, agent.policy), s, reward, s′)
    end
    
    # Update combined Q-values
    # agent.model.QTD0[:] = agent.model.TD0Model.Q
    # agent.model.QTD1[:] = agent.model.TD1Model.Q
    # agent.model.QSR[:] = agent.model.SRModel.Q
    # agent.model.QMB[:] = agent.model.MBModel.Q
    # agent.model.V[:] = agent.model.TD0Model.V

    agent.model.QSR[:] = agent.model.SRModel.Q
    agent.model.QTD0[:] .= 0
    agent.model.QTD1[:] .= 0
    agent.model.QMB[:] .= 0
    agent.model.V[:] .= 0
end

# Also update the blind step function
function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: TD0TD1SRMBWeightedStateModel, P}
    # Update TD0, TD1, and MB models normally
    update_model_step_blind!(StateAgent(agent.env, agent.model.TD0Model, agent.policy), s, s′)
    update_model_step_blind!(StateAgent(agent.env, agent.model.TD1Model, agent.policy), s, s′)
    update_model_step_blind!(StateAgent(agent.env, agent.model.MBModel, agent.policy), s, s′)
    
    # Update SR model with optional importance sampling
    if agent.model.use_sr_importance_sampling
        neighbors = find_neighbors(agent.env, s)
        # For blind updates, we don't have reward, so we calculate action prob and apply IS to M update only
        action_prob = calculate_action_probability(agent.policy, agent.model, s, s′, neighbors)
        
        # Simplified importance sampling update for blind case
        SR = agent.model.SRModel
        SR.trace[:] = SR.trace .* (SR.γ * SR.λ)
        SR.trace[s] = SR.trace[s] + 1
        
        n_actions = length(neighbors)
        default_policy_prob = 1.0 / n_actions
        w = default_policy_prob / action_prob
        
        δM = SR.ident[s, :] + SR.γ .* SR.M[s′, :] - SR.M[s, :]
        for sx in 1:length(SR)
            SR.M[sx, :] = SR.M[sx, :] + (SR.αM * SR.trace[sx] * w) .* δM
        end
        
        SR.V[:] = SR.M * SR.R
        SR.Q[:] = SR.γ * SR.V
    else
        # Regular SR update
        update_model_step_blind!(StateAgent(agent.env, agent.model.SRModel, agent.policy), s, s′)
    end
    
    # Update combined Q-values
    agent.model.QTD0[:] = agent.model.TD0Model.Q
    agent.model.QTD1[:] = agent.model.TD1Model.Q
    agent.model.QSR[:] = agent.model.SRModel.Q
    agent.model.QMB[:] = agent.model.MBModel.Q
    agent.model.V[:] = agent.model.TD0Model.V
end

##################     ARMIN ADDITION END     ##################


function update_model_start!(agent::StateAgent{E, M, P}) where {E, M <: TD0TD1SRMBWeightedStateModel, P}
    # update_model_start!(StateAgent(agent.env, agent.model.TD0Model, agent.policy))
    # update_model_start!(StateAgent(agent.env, agent.model.TD1Model, agent.policy))
    update_model_start!(StateAgent(agent.env, agent.model.SRModel, agent.policy))
    # update_model_start!(StateAgent(agent.env, agent.model.MBModel, agent.policy))
    # agent.model.QTD0[:] = agent.model.TD0Model.Q
    # agent.model.QTD1[:] = agent.model.TD1Model.Q
    # agent.model.QSR[:] = agent.model.SRModel.Q
    # agent.model.QMB[:] = agent.model.MBModel.Q
    # agent.model.V[:] = agent.model.TD0Model.V
    
    agent.model.QTD0[:] .= 0
    agent.model.QTD1[:] .= 0
    agent.model.QSR[:] = agent.model.SRModel.Q
    agent.model.QMB[:] .= 0
    agent.model.V[:] .= 0
end

# function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: TD0TD1SRMBWeightedStateModel, P}
#     update_model_step!(StateAgent(agent.env, agent.model.TD0Model, agent.policy), s, reward, s′)
#     update_model_step!(StateAgent(agent.env, agent.model.TD1Model, agent.policy), s, reward, s′)
#     update_model_step!(StateAgent(agent.env, agent.model.SRModel, agent.policy), s, reward, s′)
#     update_model_step!(StateAgent(agent.env, agent.model.MBModel, agent.policy), s, reward, s′)
#     agent.model.QTD0[:] = agent.model.TD0Model.Q
#     agent.model.QTD1[:] = agent.model.TD1Model.Q
#     agent.model.QSR[:] = agent.model.SRModel.Q
#     agent.model.QMB[:] = agent.model.MBModel.Q
#     agent.model.V[:] = agent.model.TD0Model.V
# end

# function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: TD0TD1SRMBWeightedStateModel, P}
#     update_model_step_blind!(StateAgent(agent.env, agent.model.TD0Model, agent.policy), s, s′)
#     update_model_step_blind!(StateAgent(agent.env, agent.model.TD1Model, agent.policy), s, s′)
#     update_model_step_blind!(StateAgent(agent.env, agent.model.SRModel, agent.policy), s, s′)
#     update_model_step_blind!(StateAgent(agent.env, agent.model.MBModel, agent.policy), s, s′)
#     agent.model.QTD0[:] = agent.model.TD0Model.Q
#     agent.model.QTD1[:] = agent.model.TD1Model.Q
#     agent.model.QSR[:] = agent.model.SRModel.Q
#     agent.model.QMB[:] = agent.model.MBModel.Q
#     agent.model.V[:] = agent.model.TD0Model.V
# end

function update_model_end!(agent::StateAgent{E, M, P}, episode::Episode) where {E, M <: TD0TD1SRMBWeightedStateModel, P}
    # update_model_end!(StateAgent(agent.env, agent.model.TD0Model, agent.policy), episode)
    # update_model_end!(StateAgent(agent.env, agent.model.TD1Model, agent.policy), episode)
    update_model_end!(StateAgent(agent.env, agent.model.SRModel, agent.policy), episode)
    # update_model_end!(StateAgent(agent.env, agent.model.MBModel, agent.policy), episode)
    # agent.model.QTD0[:] = agent.model.TD0Model.Q
    # agent.model.QTD1[:] = agent.model.TD1Model.Q
    agent.model.QSR[:] = agent.model.SRModel.Q
    # agent.model.QMB[:] = agent.model.MBModel.Q
    # agent.model.V[:] = agent.model.TD0Model.V

    agent.model.QTD0[:] .= 0
    agent.model.QTD1[:] .= 0
    agent.model.QMB[:] .= 0
    agent.model.V[:] .= 0
end

# Snapshot code
struct TD0TD1SRMBWeightedStateModelSnapshot <: AbstractModelSnapshot
    V::Vector{Float64}
    QTD0::Vector{Float64}
    QTD1::Vector{Float64}
    QSR::Vector{Float64}
    QMB::Vector{Float64}
    M::Matrix{Float64}
    
end
function TD0TD1SRMBWeightedStateModelSnapshot(model::TD0TD1SRMBWeightedStateModel)
    TD0TD1SRMBWeightedStateModelSnapshot(
        copy(model.V),
        copy(model.QTD0),
        copy(model.QTD1),
        copy(model.QSR),
        copy(model.QMB),
        copy(model.SRModel.M))
end
mutable struct TD0TD1SRMBWeightedStateModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    V::Matrix{Float64}
    QTD0::Matrix{Float64}
    QTD1::Matrix{Float64}
    QSR::Matrix{Float64}
    QMB::Matrix{Float64}
    M::Array{Float64, 3}
    n::Int
end
function TD0TD1SRMBWeightedStateModelRecord(agent::StateAgent{E,M,P}, maxsize::Int)::TD0TD1SRMBWeightedStateModelRecord where {E <: AbstractEnv, M <: TD0TD1SRMBWeightedStateModel, P <: AbstractPolicy}
    TD0TD1SRMBWeightedStateModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env), length(agent.env)),
        0)
end
Base.firstindex(record::TD0TD1SRMBWeightedStateModelRecord) = 1
Base.lastindex(record::TD0TD1SRMBWeightedStateModelRecord) = length(record)
Base.length(record::TD0TD1SRMBWeightedStateModelRecord) = record.n
function Base.push!(record::TD0TD1SRMBWeightedStateModelRecord, model::TD0TD1SRMBWeightedStateModel)
    record.n += 1
    (sx, sy) = size(record.V)
    if record.n > sx
        new_V = zeros(sx * 2, sy)
        new_V[1:sx, :] .= record.V
        record.V = new_V

        new_QTD0 = zeros(sx * 2, sy)
        new_QTD0[1:sx, :] .= record.QTD0
        record.QTD0 = new_QTD0

        new_QTD1 = zeros(sx * 2, sy)
        new_QTD1[1:sx, :] .= record.QTD1
        record.QTD1 = new_QTD1

        new_QSR = zeros(sx * 2, sy)
        new_QSR[1:sx, :] .= record.QSR
        record.QSR = new_QSR

        new_QMB = zeros(sx * 2, sy)
        new_QMB[1:sx, :] .= record.QMB
        record.QMB = new_QMB

        new_M = zeros(sx * 2, sy, sy)
        new_M[1:sx, :, :] .= record.M
        record.M = new_M
    end
    record.V[record.n, :] = model.V[:]
    record.QTD0[record.n, :] = model.QTD0[:]
    record.QTD1[record.n, :] = model.QTD1[:]
    record.QSR[record.n, :] = model.QSR[:]
    record.QMB[record.n, :] = model.QMB[:]
    record.M[record.n, :, :] = model.SRModel.M[:, :]
end
function Base.iterate(record::TD0TD1SRMBWeightedStateModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (TD0TD1SRMBWeightedStateModelSnapshot(
            record.V[state, :],
            record.QTD0[state, :],
            record.QTD1[state, :],
            record.QSR[state, :],
            record.QMB[state, :],
            record.M[state, :, :]), state+1)
    end
end
function Base.getindex(record::TD0TD1SRMBWeightedStateModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    TD0TD1SRMBWeightedStateModelSnapshot(
        record.V[i, :], 
        record.QTD0[i, :],
        record.QTD1[i, :],
        record.QSR[i, :],
        record.QMB[i, :],
        record.M[i, :, :])
end
Base.getindex(record::TD0TD1SRMBWeightedStateModelRecord, I) = TD0TD1SRMBWeightedStateModelRecord(
    record.env,
    record.policy,
    record.V[I, :],
    record.QTD0[I, :],
    record.QTD1[I, :],
    record.QSR[I, :],
    record.QMB[I, :],
    record.M[I, I, :],
    length(I))

function Record(agent::StateAgent{E, M, P}, maxsize::Int)::TD0TD1SRMBWeightedStateModelRecord where {E, M <: TD0TD1SRMBWeightedStateModel, P}
    TD0TD1SRMBWeightedStateModelRecord(agent, maxsize)
end