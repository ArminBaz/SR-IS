"""

e^v/λ = MP e^r/λ

We can pass in `c` as a step cost

For Linear RL, we need to keep track of the following:
e_V: exp(v/λ)
V: v
Q: v + step cost
R: Reward, as well as step cost (negative) at each point
T: Transition Matrix for π_d
T_policy: Transition matrix for π
D: This is the equivalent of M, for all (terminal & non-terminal) states
z_hat: This is DP
α: Learning rate for reward
αT: Learning rate for transitions
λ: Control Cost Scaling
"""
abstract type AbstractLRL <: AbstractStateModel end
mutable struct LRLModel <: AbstractLRL
    e_V::Vector{Float64}
    V::Vector{Float64}
    Q::Vector{Float64}
    R::Vector{Float64}
    T::Matrix{Float64}
    T_policy::Matrix{Float64}
    D::Matrix{Float64}
    z_hat::Matrix{Float64}
    α::Float64
    α2::Float64
    αT::Float64
    λ::Float64
end
function LRLModel(env, α, α2, αT, λ, c)
    n = length(env)
    terminals = findall(env.terminal_states)
    nonterminals = findall(.!env.terminal_states)

    # Initial calculation:
    R = zeros(n)
    R[nonterminals] .= -c

    T = stochastic_matrix(env)
    T_policy = zeros(n, n)

    D = zeros(n, n)
    z_hat = zeros(n, length(terminals))

    e_V = zeros(n)
    V = zeros(n)
    Q = zeros(n)

    LRL = LRLModel(e_V, V, Q, R, T, T_policy, D, z_hat, α, α2, αT, λ)

    recompute_z!(env, LRL)
    recompute_V!(env, LRL)
    recompute_policy!(LRL)

    return LRL
end
LRLModel(env, α, αT, λ, c)::LRLModel = LRLModel(env, α, α, αT, λ, c)
function LRLModel(env; α, α2=nothing, αT, λ, c)::LRLModel
    if isnothing(α2)
        α2 = α
    end
    LRLModel(env, α, α2, αT, λ, c)
end
function model_name(model::M) where {M <: AbstractLRL} "LRL" end


function LRLSoftmax(env; α, αT, λ, c, β, α2=nothing)
    if isnothing(α2)
        α2 = α
    end
    LRL = LRLModel(env, α, α2, αT, λ, c)
    policy = PolicySoftmax(β)
    StateAgent(env, LRL, policy)
end

function LRLTwoStepSoftmax(env; α, αT, λ, c, β1, β2, α2=nothing)
    if isnothing(α2)
        α2 = α
    end
    LRL = LRLModel(env, α, α2, αT, λ, c)
    policy = PolicyTwoStepSoftmax(β1, β2)
    StateAgent(env, LRL, policy)
end

function LRLGreedy(env; α, αT, λ, c, α2=nothing)
    if isnothing(α2)
        α2 = α
    end
    LRL = LRLModel(env, α, α2, αT, λ, c)
    policy = PolicyGreedy()
    StateAgent(env, LRL, policy)
end

function LRL_ϵ_Greedy(env; α, αT, λ, c, ϵ, α2=nothing)
    if isnothing(α2)
        α2 = α
    end
    LRL = LRLModel(env, α, α2, αT, λ, c)
    policy = Policy_ϵ_Greedy(ϵ)
    StateAgent(env, LRL, policy)
end

function LRLOnPolicy(env; α, αT, λ, c, α2=nothing)
    if isnothing(α2)
        α2 = α
    end
    LRL = LRLModel(env, α, α2, αT, λ, c)
    policy = PolicyLRLOnPolicy()
    StateAgent(env, LRL, policy)
end

function LRLOnPolicy_ϵ_Greedy(env; α, αT, λ, c, ϵ, α2=nothing)
    if isnothing(α2)
        α2 = α
    end
    LRL = LRLModel(env, α, α2, αT, λ, c)
    policy = PolicyLRLOnPolicy_ϵ_Greedy(ϵ)
    StateAgent(env, LRL, policy)
end

function update_model_start!(agent::StateAgent{E, M, P}) where {E, M <: AbstractLRL, P} end

function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: AbstractLRL, P} end

function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: AbstractLRL, P} end

"Update reward, transitions, value and policy at the end of an episode"
function update_model_end!(agent::StateAgent{E, M, P}, ep::Episode) where {E, M <: AbstractLRL, P}
    # Update terminal reward
    if ep[1].S == 1
        α = agent.model.α
    else
        α = agent.model.α2
    end
    for (s, r) in ep
        if agent.env.terminal_states[s]
            agent.model.R[s] = (α * r) + ((1 - α) * agent.model.R[s])
        end
    end

    # Update transitions
    # s_prev = 0
    # for (s, r) in ep
    #     if (s_prev > 0)
    #         agent.model.T[s_prev,:] .*= (1 - agent.model.αT)
    #         agent.model.T[s_prev,s] += agent.model.αT
    #     end
    #     s_prev = s
    # end

    recompute_z!(agent)
    recompute_V!(agent)
    recompute_policy!(agent)
end

"""Recompute D and z_hat directly for an LRL model

Necessary if transition matrix changes, or terminal states are added or removed
"""
function recompute_z!(agent::StateAgent{E, M, Q}) where {E, M <: AbstractLRL, Q}
    recompute_z!(agent.env, agent.model)
end
function recompute_z!(env, LRL::AbstractLRL)
    terminals = findall(env.terminal_states)

    # computing M
    # M is D[nonterminals, terminals]
    # Instead, we're calculatimg z_hat = DP and then subsetting the nonterminal states
    L = diagm(exp.(-LRL.R./LRL.λ)) - LRL.T
    LRL.D .= L^-1
    P = LRL.T[:, terminals]

    LRL.z_hat .= LRL.D * P
end

"Recompute e_V and V for a Linear RL agent"
function recompute_V!(agent::StateAgent{E, M, Q}) where {E, M <: AbstractLRL, Q}
    recompute_V!(agent.env, agent.model)
end
function recompute_V!(env, LRL::AbstractLRL)
    terminals = findall(env.terminal_states)
    nonterminals = findall(.!env.terminal_states)

    e_R = exp.(LRL.R[terminals]./LRL.λ)

    LRL.e_V[nonterminals] .= LRL.z_hat[nonterminals, :] * e_R
    LRL.e_V[terminals] .= e_R

    LRL.V .= log.(LRL.e_V) .* LRL.λ
    LRL.Q .= LRL.V # v(s) already includes r(s), this is what we should use for deciding between actions
end

"Recompute policy (π) for a Linear RL agent"
function recompute_policy!(agent::StateAgent{E, M, Q}) where {E, M <: AbstractLRL, Q}
    recompute_policy!(agent.model)
end
function recompute_policy!(LRL::AbstractLRL)
    LRL.T_policy .= diagm(1 ./ (LRL.T * LRL.e_V)) * LRL.T * diagm(LRL.e_V)
end

"KL Distance between default and current policy at each state"
function kl_distance(model::AbstractLRL)
    kl = model.T_policy * model.V / model.λ - log.(model.T * model.e_V)
    max.(kl, 0) # KL is non-negative (eliminate rounding errors)
end
function kl_distance(agent::StateAgent{E, M, Q}) where {E, M <: AbstractLRL, Q}
    kl_distance(agent.model)
end

"Control cost (λ*KL) at each state"
function control_cost(model::AbstractLRL)
    model.λ * kl_distance(model)
end
function control_cost(agent::StateAgent{E, M, Q}) where {E, M <: AbstractLRL, Q}
    control_cost(agent.model)
end

##################
# Snapshot code
##################
struct LRLModelSnapshot <: AbstractModelSnapshot
    V::Vector{Float64}
    R::Vector{Float64}
    control_cost::Vector{Float64}
    T::Matrix{Float64}
    T_policy::Matrix{Float64}
end
function LRLModelSnapshot(model::LRLModel)
    LRLModelSnapshot(copy(model.V), copy(model.R), control_cost(model), copy(model.T), copy(model.T_policy))
end
mutable struct LRLModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    V::Matrix{Float64}
    R::Matrix{Float64}
    control_cost::Matrix{Float64}
    T::Array{Float64, 3}
    T_policy::Array{Float64, 3}
    n::Int
end
function LRLModelRecord(agent::StateAgent{E,M,P}, maxsize::Int)::LRLModelRecord where {E <: AbstractEnv, M <: LRLModel, P <: AbstractPolicy}
    LRLModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env), length(agent.env)),
        zeros(maxsize, length(agent.env), length(agent.env)),
        0)
end
Base.firstindex(record::LRLModelRecord) = 1
Base.lastindex(record::LRLModelRecord) = length(record)
Base.length(record::LRLModelRecord) = record.n
function Base.push!(record::LRLModelRecord, model::LRLModel)
    record.n += 1
    (sx, sy) = size(record.V)
    if record.n > sx
        new_V = zeros(sx * 2, sy)
        view(new_V, 1:sx, :) .= record.V
        record.V = new_V

        new_R = zeros(sx * 2, sy)
        view(new_R, 1:sx, :) .= record.R
        record.R = new_R

        new_control_cost = zeros(sx * 2, sy)
        new_control_cost[1:sx, :] .= record.control_cost
        record.control_cost = new_control_cost

        new_T = zeros(sx * 2, sy, sy)
        new_T[1:sx, :, :] .= record.T
        record.T = new_T

        new_T_policy = zeros(sx * 2, sy, sy)
        new_T_policy[1:sx, :, :] .= record.T_policy
        record.T_policy = new_T_policy
    end
    record.V[record.n, :] = model.V[:]
    record.R[record.n, :] = model.R[:]
    record.control_cost[record.n, :] = control_cost(model)
    record.T[record.n, :, :] = model.T[:, :]
    record.T_policy[record.n, :, :] = model.T_policy[:, :]
end
function Base.iterate(record::LRLModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (LRLModelSnapshot(
            record.V[state, :],
            record.R[state, :],
            record.control_cost[state, :],
            record.T[state, :, :],
            record.T_policy[state, :, :]),
            state+1)
    end
end
function Base.getindex(record::LRLModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    LRLModelSnapshot(
        record.V[i, :],
        record.R[i, :],
        record.control_cost[i, :],
        record.T[i, :, :],
        record.T_policy[i, :, :])
end
Base.getindex(record::LRLModelRecord, I) = LRLModelRecord(
    record.env,
    record.policy,
    record.V[I, :],
    record.R[I, :],
    record.control_cost[I, :],
    record.T[I, :, :],
    record.T_policy[I, :, :],
    length(I))

function Record(agent::StateAgent{E, M, P}, maxsize::Int)::LRLModelRecord where {E, M <: LRLModel, P}
    LRLModelRecord(agent, maxsize)
end