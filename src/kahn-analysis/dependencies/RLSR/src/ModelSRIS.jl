# 1. Create SR-IS Model (extends the existing SRModel)
abstract type AbstractSR <: AbstractStateModel end
mutable struct SRISModel <: AbstractSR
    M::Matrix{Float64}
    R::Vector{Float64}
    V::Vector{Float64}
    Q::Vector{Float64}
    α::Float64
    αM::Float64
    γ::Float64
    λ::Float64
    trace::Vector{Float64}
    ident::Matrix{Float64}
    # New fields for IS
    last_policy_probs::Vector{Float64}  # Store policy probabilities from last decision
    last_action_idx::Int                 # Which action was taken
end

# Constructor - same as SRModel but with IS fields
function SRISModel(env, α, αM, γ, λ)
    n = length(env)
    # Initialize M by inverting the transition matrix
    M = inv(I(n) - γ * stochastic_matrix(env))
    # Set M to zero for the dummy terminal states
    view(M, :, env.terminal_states) .= 0
    # Ident is used every step, save ourselves from reallocating
    ident = Matrix{Float64}(I, n, n)
    trace = zeros(n)
    SRISModel(M, zeros(n), zeros(n), zeros(n), α, αM, γ, λ, trace, ident, [1.0], 1)
end

# Alternative constructor with keyword arguments
function SRISModel(env; α, αM, γ, λ)
    SRISModel(env, α, αM, γ, λ)
end

# Model name function for compatibility
function model_name(model::SRISModel) 
    "SR-IS" 
end

# 2. Modified Policy that stores probabilities
mutable struct PolicyTwoStepSoftmaxIS <: AbstractPolicy
    β1::Float64
    β2::Float64
    last_probs::Vector{Float64}  # Store the last computed probabilities
    last_action_idx::Int         # Store which action was taken
end

function PolicyTwoStepSoftmaxIS(β1, β2)
    PolicyTwoStepSoftmaxIS(β1, β2, [1.0], 1)
end

# Modified sample_successor that saves probabilities
function sample_successor(env::AbstractEnv, model::SRISModel, policy::PolicyTwoStepSoftmaxIS, s::Int)::Union{Int, Nothing}
    neighbors = find_neighbors(env, s)
    if isempty(neighbors)
        nothing
    else
        β = (s == 1) ? policy.β1 : policy.β2
        neighbor_values = exp.(β * model.Q[neighbors])
        probs = neighbor_values ./ sum(neighbor_values)
        
        # Store probabilities for IS
        policy.last_probs = probs
        
        # Sample action
        action_idx = sample(1:length(neighbors), Weights(probs))
        policy.last_action_idx = action_idx
        
        # Also store in model for update
        model.last_policy_probs = probs
        model.last_action_idx = action_idx
        
        neighbors[action_idx]
    end
end

function policy_name(policy::PolicyTwoStepSoftmaxIS) 
    "Two-Step Softmax IS" 
end

# 3. Update functions for SR-IS
function update_model_start!(agent::StateAgent{E, M, P}) where {E, M <: SRISModel, P}
    agent.model.trace .= 0
end

# Blind update (without rewards) - used during learning
function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: SRISModel, P}
    SR = agent.model

    # Update eligibility trace
    SR.trace[:] = SR.trace .* (SR.γ * SR.λ)
    SR.trace[s] = SR.trace[s] + 1

    # Calculate importance weight only for choice states
    is_weight = 1.0
    if length(find_neighbors(agent.env, s)) > 1  # Choice point
        n_actions = length(SR.last_policy_probs)
        uniform_prob = 1.0 / n_actions
        policy_prob = SR.last_policy_probs[SR.last_action_idx]
        is_weight = min(uniform_prob / max(policy_prob, 0.01), 5.0)  # Clip to prevent extreme weights
    end

    # Update M with IS weighting
    δM = SR.ident[s, :] + SR.γ .* SR.M[s′, :] - SR.M[s, :]
    for sx in 1:length(SR)
        SR.M[sx, :] = SR.M[sx, :] + (SR.αM * SR.trace[sx] * is_weight) .* δM
    end

    # Update V and Q
    SR.V[:] = SR.M * SR.R
    SR.Q[:] = SR.γ * SR.V
end

# Full update (with rewards)
function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: SRISModel, P}
    SR = agent.model
    
    # Update eligibility trace
    SR.trace[:] = SR.trace .* (SR.γ * SR.λ)
    SR.trace[s] = SR.trace[s] + 1
    
    # Calculate importance weight only for choice states
    is_weight = 1.0
    if length(find_neighbors(agent.env, s)) > 1  # Choice point
        n_actions = length(SR.last_policy_probs)
        uniform_prob = 1.0 / n_actions
        policy_prob = SR.last_policy_probs[SR.last_action_idx]
        is_weight = min(uniform_prob / max(policy_prob, 0.01), 5.0)  # Clip to prevent extreme weights
    end
    
    # Update M with IS weighting
    δM = SR.ident[s, :] + SR.γ .* SR.M[s′, :] - SR.M[s, :]
    for sx in 1:length(SR)
        SR.M[sx, :] = SR.M[sx, :] + (SR.αM * SR.trace[sx] * is_weight) .* δM
    end
    
    # Update R with IS weighting
    SR.R[s] = SR.R[s] + SR.α * is_weight * (reward - SR.R[s])
    
    # Update V and Q
    SR.V[:] = SR.M * SR.R
    SR.Q[:] = SR.γ * SR.V
end

function update_model_end!(::StateAgent{E, M, P}, ::Episode) where {E, M <: SRISModel, P} 
    # No end-of-episode updates needed
end

# 4. Snapshot and recording functionality for SR-IS
struct SRISModelSnapshot <: AbstractModelSnapshot
    V::Vector{Float64}
    M::Matrix{Float64}
end

function SRISModelSnapshot(model::SRISModel)
    SRISModelSnapshot(copy(model.V), copy(model.M))
end

mutable struct SRISModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    V::Matrix{Float64}
    M::Array{Float64, 3}
    n::Int
end

function SRISModelRecord(agent::StateAgent{E,M,P}, maxsize::Int)::SRISModelRecord where {E <: AbstractEnv, M <: SRISModel, P <: AbstractPolicy}
    SRISModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env), length(agent.env)),
        0)
end

Base.firstindex(record::SRISModelRecord) = 1
Base.lastindex(record::SRISModelRecord) = length(record)
Base.length(record::SRISModelRecord) = record.n

function Base.push!(record::SRISModelRecord, model::SRISModel)
    record.n += 1
    (sx, sy) = size(record.V)
    if record.n > sx
        new_V = zeros(sx * 2, sy)
        view(new_V, 1:sx, :) .= record.V
        record.V = new_V

        new_M = zeros(sx * 2, sy, sy)
        new_M[1:sx, :, :] .= record.M
        record.M = new_M
    end
    record.V[record.n, :] = model.V[:]
    record.M[record.n, :, :] = model.M[:, :]
end

function Base.iterate(record::SRISModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (SRISModelSnapshot(record.V[state, :], record.M[state, :, :]), state+1)
    end
end

function Base.getindex(record::SRISModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    SRISModelSnapshot(record.V[i, :], record.M[i, :, :])
end

Base.getindex(record::SRISModelRecord, I) = SRISModelRecord(record.env, record.policy, record.V[I, :], record.M[I, I, :], length(I))

function Record(agent::StateAgent{E, M, P}, maxsize::Int)::SRISModelRecord where {E, M <: SRISModel, P}
    SRISModelRecord(agent, maxsize)
end

# 5. Create SR-IS agent constructors
function SRISTwoStepSoftmax(env; α, αM, γ, λ, β1, β2)
    SR = SRISModel(env; α=α, αM=αM, γ=γ, λ=λ)
    policy = PolicyTwoStepSoftmaxIS(β1, β2)
    StateAgent(env, SR, policy)
end