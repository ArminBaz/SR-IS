abstract type AbstractSR <: AbstractStateModel end
mutable struct SRModel <: AbstractSR
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
    αDR::Float64
    DR::Matrix{Float64}
    P::Matrix{Float64}
    e_V::Vector{Float64}
    e_R::Vector{Float64}
    z_hat::Matrix{Float64}
    terminals::Vector{Int64}
    nonterminals::Vector{Int64}
end
function SRModel(env, α, αM, γ, λ, c)
    n = length(env)
    # Define terminal states
    # terminals = falses(n)
    # terminals[end-3:end] .= true
    terminals = findall(env.terminal_states)
    nonterminals = findall(.!env.terminal_states)

    # Make R -1 for all the states
    # R = fill(-1, n)  # where n is the number of states
    # R[terminals] .= 0
    R = zeros(n)
    R[nonterminals] .= -c

    # exp(r) at terminal states
    e_R = exp.(R[terminals]./λ)

    # Initialize the SR
    M = inv(I(n) - γ * stochastic_matrix(env))

    # Get transition matrix
    T = stochastic_matrix(env)

    # Use T and terminal states to define P and initialize the DR
    # DR = zeros(n, n)
    P = T[:, terminals]
    # L = diagm(exp.(-R./λ)) - T
    # DR .= L^-1
    # DR = inv(I(n) - γ * stochastic_matrix(env))
    DR = fill(0.01, (n, n))
    DR[diagind(DR)] .= 1
    DR[terminals, terminals] .= 1/(1-γ)
    
    # Set M to zero for the dummy terminal states
    # Ensures that they won't affect our weight estimates or norming
    view(M, :, env.terminal_states) .= 0
    # Ident is used every step, save ourselves from reallocating
    ident = Matrix{Float64}(I, n, n)
    trace = zeros(n)

    # Define z_hat, exp(V), V, and Q
    e_V = zeros(n)
    V = zeros(n)
    Q = zeros(n)
    z_hat = zeros(n, length(terminals))
    z_hat .= DR * P
    e_V[nonterminals] .= z_hat[nonterminals, :] * e_R
    e_V[terminals] .= e_R

    V .= log.(e_V) .* λ
    Q .= V
    
    SRModel(M, R, zeros(n), zeros(n), α, αM, γ, λ, trace, ident, αM, DR, P, e_V, e_R, z_hat, terminals, nonterminals)
end
SRModel(env; α, αM, γ, λ, c)::SRModel = SRModel(env, α, αM, γ, λ, c)
function model_name(model::M) where {M <: AbstractSR} "SR" end

function SRSoftmax(env; α, αM, γ, λ, β)
    SR = SRModel(env, α, αM, γ, λ, c)
    policy = PolicySoftmax(β)
    StateAgent(env, SR, policy)
end

function SRTwoStepSoftmax(env; α, αM, γ, λ, β1, β2)
    SR = SRModel(env, α, αM, γ, λ, c)
    policy = PolicyTwoStepSoftmax(β1, β2)
    StateAgent(env, SR, policy)
end

function SR_ϵ_Greedy(env; α, αM, γ, λ, ϵ)
    SR = SRModel(env, α, αM, γ, λ, c)
    policy = Policy_ϵ_Greedy(ϵ)
    StateAgent(env, SR, policy)
end

function SRGreedy(env; α, αM, γ, λ)
    SR = SRModel(env, α, αM, γ, λ, c)
    policy = PolicyGreedy()
    StateAgent(env, SR, policy)
end

"""SR Non-Trace Update:

After a transition s → s′,

1. δ = R(s, a) + γV(s′) - V(s)
    - Error term
2. M[s, :] = (1 - αM) * M[s, :] + αM * (Iₛ + γ * M[s′, :])
    - M[s] should approach Iₛ plus the discounted M of the successor state
3. w(i) ← w(i) + αw δ M(s,i)
    - Each state that s could transition to (that s has as a feature) has its weight adjusted

- To maintain consistency, the M update should occur /before/ the w update
- Need to ensure that V is updated after both M and w updates


SR Trace Update:

After a transition s → s′,

1. trace = I_s + λ*γ*trace 
    - Discount the existing trace, and add 1 the current state
2. δM = (Iₛ + γ M_s') - (M_s)
    Observed state transition, versus prior prediction
3. For each state, M_sx = M_sx + α * trace[sx] * δM

NB should verify what trace M updates should look like?


"""
function update_model_start!(agent::StateAgent{E, M, P}) where {E, M <: AbstractSR, P}
    agent.model.trace .= 0
end

function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: AbstractSR, P}
    SR = agent.model

    # Update eligibility trace
    SR.trace[:] = SR.trace .* (SR.γ * SR.λ)
    SR.trace[s] = SR.trace[s] + 1

    # Update M first
    # δM is our state misprediction: Iₛ + γ M_s' vs. M_s
    # Move each state's prediction in the direction of δM based on the trace
    δM = SR.ident[s, :] + SR.γ .* SR.M[s′, :] - SR.M[s, :]
    # The Iₛ cancels out, leaving us with γM_s′.
    # M_s should already be an average of γM_s′1, γM_s′2, etc.
    for sx in 1:length(SR)
        SR.M[sx, :] = SR.M[sx, :] + (SR.αM * SR.trace[sx]) .* δM
    end

    # Update V
    SR.V[:] = SR.M * SR.R

    # Update Q to include the discount
    # This isn't technically correct, since we're ignoring reward from the current state,
    # but as long as all rewards are terminal, it should be fine
    SR.Q[:] = SR.γ * SR.V
end

function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: AbstractSR, P}
    SR = agent.model

    # Update eligibility trace
    SR.trace[:] = SR.trace .* (SR.γ * SR.λ)
    SR.trace[s] = SR.trace[s] + 1

    # Update M first
    # δM is our state misprediction: (Iₛ + γ M_s') vs. (M_s)
    # Move each state's prediction in the direction of δM based on the trace
    δM = SR.ident[s, :] + SR.γ .* SR.M[s′, :] - SR.M[s, :]
    # The Iₛ cancels out, leaving us with γM_s′.
    # M_s should already be an average of γM_s′1, γM_s′2, etc.
    for sx in 1:length(SR)
        SR.M[sx, :] = SR.M[sx, :] + (SR.αM * SR.trace[sx]) .* δM
    end

    # Update R - for proper discounting w/ the dummy terminal states,
    # we're updating the step before
    SR.R[s] = (SR.α * reward) + ((1 - SR.α) * SR.R[s])

    # Update V
    SR.V[:] = SR.M * SR.R

    # Update Q to include the discount
    # This isn't technically correct, since we're ignoring reward from the current state,
    # but as long as all rewards are terminal, it should be fine
    SR.Q[:] = SR.γ * SR.V
end

function update_model_end!(::StateAgent{E, M, P}, ::Episode) where {E, M <: AbstractSR, P} end

# Snapshot code
struct SRModelSnapshot <: AbstractModelSnapshot
    V::Vector{Float64}
    M::Matrix{Float64}
    P::Matrix{Float64}
    DR::Matrix{Float64}
    e_V::Vector{Float64}
end
function SRModelSnapshot(model::SRModel)
    SRModelSnapshot(copy(model.V), copy(model.M), copy(model.P), copy(model.DR), copy(model.e_V))
end
mutable struct SRModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    V::Matrix{Float64}
    M::Array{Float64, 3}
    n::Int
end
function SRModelRecord(agent::StateAgent{E,M,P}, maxsize::Int)::SRModelRecord where {E <: AbstractEnv, M <: SRModel, P <: AbstractPolicy}
    SRModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        zeros(maxsize, length(agent.env), length(agent.env)),
        0)
end
Base.firstindex(record::SRModelRecord) = 1
Base.lastindex(record::SRModelRecord) = length(record)
Base.length(record::SRModelRecord) = record.n
function Base.push!(record::SRModelRecord, model::SRModel)
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
function Base.iterate(record::SRModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (SRModelSnapshot(record.V[state, :], record.M[state, :, :]), state+1)
    end
end
function Base.getindex(record::SRModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    SRModelSnapshot(record.V[i, :], record.M[i, :, :])
end
Base.getindex(record::SRModelRecord, I) = SRModelRecord(record.env, record.policy, record.V[I, :], record.M[I, I, :], length(I))

function Record(agent::StateAgent{E, M, P}, maxsize::Int)::SRModelRecord where {E, M <: SRModel, P}
    SRModelRecord(agent, maxsize)
end