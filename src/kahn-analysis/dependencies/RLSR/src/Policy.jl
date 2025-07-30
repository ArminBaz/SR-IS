struct Policy_ϵ_Greedy <: AbstractPolicy
    ϵ::Float64
end
function Policy_ϵ_Greedy(; ϵ)
    Policy_ϵ_Greedy(ϵ)
end
function sample_successor(env::AbstractEnv, model::AbstractStateModel, policy::Policy_ϵ_Greedy, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        weights = model.Q[neighbors]
        if rand() < policy.ϵ || sum(weights) == 0
            rand(neighbors)
        else
            rand(neighbors[weights .== maximum(weights)])
        end
    end
end
function sample_successor(env::AbstractEnv, model::AbstractActionModel, policy::Policy_ϵ_Greedy, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        edges = [(s, n) for n in neighbors]
        weights = [model.Q[edge_to_ind(env, e)] for e in edges]
        if rand() < policy.ϵ || sum(weights) == 0
            rand(neighbors)
        else
            rand(neighbors[weights .== maximum(weights)])
        end
    end
end
function policy_name(policy::Policy_ϵ_Greedy) "ϵ-Greedy" end

struct PolicyGreedy <: AbstractPolicy
end
function sample_successor(env::AbstractEnv, model::AbstractStateModel, ::PolicyGreedy, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        weights = model.Q[neighbors]
        rand(neighbors[weights .== maximum(weights)])
    end
end
function sample_successor(env::AbstractEnv, model::AbstractActionModel, ::PolicyGreedy, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        edges = [(s, n) for n in neighbors]
        weights = [model.Q[edge_to_ind(env, e)] for e in edges]
        rand(neighbors[weights .== maximum(weights)])
    end
end
function policy_name(policy::PolicyGreedy) "Greedy" end

struct PolicySoftmax <: AbstractPolicy
    β::Float64
end
function PolicySoftmax(; β)
    PolicySoftmax(β)
end
function sample_successor(env::AbstractEnv, model::AbstractStateModel, policy::PolicySoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        neighbor_values = exp.(policy.β * model.Q[neighbors])
        weights = Weights(neighbor_values ./ sum(neighbor_values))
        sample(neighbors, weights)
    end
end
function sample_successor(env::AbstractEnv, model::AbstractActionModel, policy::PolicySoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        edges = [(s, n) for n in neighbors]
        raw_weights = [model.Q[edge_to_ind(env, e)] for e in edges]
        neighbor_values = exp.(policy.β * raw_weights)
        weights = Weights(neighbor_values ./ sum(neighbor_values))
        sample(neighbors, weights)
    end
end
function policy_name(policy::PolicySoftmax) "Softmax" end

struct PolicyTwoStepSoftmax <: AbstractPolicy
    β1::Float64
    β2::Float64
end
function PolicyTwoStepSoftmax(; β1, β2)
    PolicyTwoStepSoftmax(β1, β2)
end
function sample_successor(env::AbstractEnv, model::AbstractStateModel, policy::PolicyTwoStepSoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        if (s == 1)
            neighbor_values = exp.(policy.β1 * model.Q[neighbors])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        else
            neighbor_values = exp.(policy.β2 * model.Q[neighbors])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        end
    end
end
function sample_successor(env::AbstractEnv, model::AbstractActionModel, policy::PolicyTwoStepSoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        edges = [(s, n) for n in neighbors]
        raw_weights = [model.Q[edge_to_ind(env, e)] for e in edges]
        if (s == 1)
            neighbor_values = exp.(policy.β1 * raw_weights)
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        else
            neighbor_values = exp.(policy.β2 * raw_weights)
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        end
    end
end
function policy_name(policy::PolicyTwoStepSoftmax) "Softmax" end


struct PolicyTD0TD1SRMBTwoStepSoftmax <: AbstractPolicy
    βTD0::Float64
    βTD1::Float64
    βSR::Float64
    βMB::Float64
    βBoat::Float64
end
function sample_successor(env::AbstractEnv, model::AbstractStateModel, policy::PolicyTD0TD1SRMBTwoStepSoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        if (s == 1)
            neighbor_values = exp.(policy.βTD0 * model.QTD0[neighbors] + policy.βTD1 * model.QTD1[neighbors] + policy.βSR * model.QSR[neighbors] + policy.βMB * model.QMB[neighbors])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        else
            neighbor_values = exp.(policy.βBoat * model.QTD0[neighbors])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        end
    end
end
function sample_successor(env::AbstractEnv, model::AbstractActionModel, policy::PolicyTD0TD1SRMBTwoStepSoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        edges = [(s, n) for n in neighbors]
        inds = [edge_to_ind(env, e) for e in edges]
        if (s == 1)
            neighbor_values = exp.(policy.βTD0 * model.QTD0[inds] + policy.βTD1 * model.QTD1[inds] + policy.βSR * model.QSR[inds] + policy.βMB * model.QMB[inds])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        else
            neighbor_values = exp.(policy.βBoat * model.QTD0[inds])
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            sample(neighbors, weights)
        end
    end
end
function policy_name(policy::PolicyTD0TD1SRMBTwoStepSoftmax) "TD0TD1SRMBwoStepSoftmax" end

mutable struct PolicyTD0TD1SRMBBiasTwoStepSoftmax <: AbstractPolicy
    βTD0::Float64
    βTD1::Float64
    βSR::Float64
    βMB::Float64
    βBoat::Float64
    Bias1::Float64
    Bias2::Float64
    Prev1::Int
    Prev22::Int
    Prev23::Int
end
function PolicyTD0TD1SRMBBiasTwoStepSoftmax(βTD0, βTD1, βSR, βMB, βBoat, Bias1, Bias2)
    PolicyTD0TD1SRMBBiasTwoStepSoftmax(βTD0, βTD1, βSR, βMB, βBoat, Bias1, Bias2, 0, 0, 0)
end
function sample_successor(env::AbstractEnv, model::AbstractStateModel, policy::PolicyTD0TD1SRMBBiasTwoStepSoftmax, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        if (s == 1)
            neighbor_values = exp.(policy.βTD0 * model.QTD0[neighbors] + policy.βTD1 * model.QTD1[neighbors] + policy.βSR * model.QSR[neighbors] + policy.βMB * model.QMB[neighbors])
            if (policy.Prev1 > 0)
                neighbor_values[findfirst(neighbors .== policy.Prev1)] += policy.Bias1
            end
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            s′ = sample(neighbors, weights)
            policy.Prev1 = s′
        else
            neighbor_values = exp.(policy.βBoat * model.QTD0[neighbors])
            if (s == 2)
                if (policy.Prev22 > 0)
                    neighbor_values[findfirst(neighbors .== policy.Prev22)] += policy.Bias1
                end
            elseif (s == 3)
                if (policy.Prev23 > 0)
                    neighbor_values[findfirst(neighbors .== policy.Prev23)] += policy.Bias1
                end
            end
            weights = Weights(neighbor_values ./ sum(neighbor_values))
            s′ = sample(neighbors, weights)
            if (s == 2)
                policy.Prev22 = s′
            elseif (s == 3)
                policy.Prev23 = s′
            end
        end
        s′
    end
end
# function sample_successor(env::AbstractEnv, model::AbstractActionModel, policy::PolicyTD0TD1SRMBBiasTwoStepSoftmax, s::Int)::Union{Int, Nothing}
#     neighbors = find_neighbors(env, s)
#     if isempty(neighbors)
#         nothing
#     else
#         edges = [(s, n) for n in neighbors]
#         inds = [edge_to_ind(env, e) for e in edges]
#         if (s == 1)
#             neighbor_values = exp.(policy.βTD0 * model.QTD0[inds] + policy.βTD1 * model.QTD1[inds] + policy.βSR * model.QSR[inds] + policy.βMB * model.QMB[inds])
#             weights = Weights(neighbor_values ./ sum(neighbor_values))
#             sample(neighbors, weights)
#         else
#             neighbor_values = exp.(policy.βBoat * model.QTD0[inds])
#             weights = Weights(neighbor_values ./ sum(neighbor_values))
#             sample(neighbors, weights)
#         end
#     end
# end
function policy_name(policy::PolicyTD0TD1SRMBBiasTwoStepSoftmax) "TD0TD1SRMBBiaswoStepSoftmax" end

# For Linear RL
struct PolicyLRLOnPolicy <: AbstractPolicy
end
function sample_successor(env::AbstractEnv, model::AbstractStateModel, policy::PolicyLRLOnPolicy, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        neighbor_values = model.T_policy[s, neighbors]
        weights = Weights(neighbor_values ./ sum(neighbor_values))
        sample(neighbors, weights)
    end
end

struct PolicyLRLOnPolicy_ϵ_Greedy <: AbstractPolicy
    ϵ::Float64
end
function PolicyLRLOnPolicy_ϵ_Greedy(; ϵ)
    PolicyLRLOnPolicy_ϵ_Greedy(ϵ)
end
function sample_successor(env::AbstractEnv, model::AbstractStateModel, policy::PolicyLRLOnPolicy_ϵ_Greedy, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    elseif rand() < policy.ϵ
        rand(neighbors)
    else
        neighbor_values = model.T_policy[s, neighbors]
        weights = Weights(neighbor_values ./ sum(neighbor_values))
        sample(neighbors, weights)
    end
end