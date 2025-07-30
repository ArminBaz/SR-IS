abstract type AbstractSRTD <: AbstractStateModel end
struct SRTDModel <: AbstractSRTD
    M::Matrix{Float64}
    w::Vector{Float64}
    V::Vector{Float64}
    Q::Vector{Float64}
    α::Float64
    αM::Float64
    γ::Float64
    λ::Float64
    trace::Vector{Float64}
    ident::Matrix{Float64}
end
function SRTDModel(env, α, αM, γ, λ)
    n = length(env)
    # Initialize M by inverting the transition matrix
    M = inv(I(n) - γ * stochastic_matrix(env))
    # Set M to zero for the dummy terminal states
    # Ensures that they won't affect our weight estimates or norming
    M[:, env.terminal_states] .= 0
    # Ident is used every step, save ourselves from reallocating
    ident = Matrix{Float64}(I, n, n)
    trace = zeros(n)
    SRTDModel(M, zeros(n), zeros(n), zeros(n), α, αM, γ, λ, trace, ident)
end
SRTDModel(env; α, αM, γ, λ)::SRTDModel = SRTDModel(env, α, αM, γ, λ)
function model_name(model::M) where {M <: AbstractSRTD} "SR-TD" end

function SRTDSoftmax(env; α, αM, γ, λ, β)
    SRTD = SRTDModel(env, α, αM, γ, λ)
    policy = PolicySoftmax(β)
    StateAgent(env, SRTD, policy)
end

function SRTD_ϵ_Greedy(env; α, αM, γ, λ, ϵ)
    SRTD = SRTDModel(env, α, αM, γ, λ)
    policy = Policy_ϵ_Greedy(ϵ)
    StateAgent(env, SRTD, policy)
end

function SRTDGreedy(env; α, αM, γ, λ)
    SRTD = SRTDModel(env, α, αM, γ, λ)
    policy = PolicyGreedy()
    StateAgent(env, SRTD, policy)
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
function update_model_start!(agent::StateAgent{E, M, P}) where {E, M <: AbstractSRTD, P}
    agent.model.trace .= 0
end

function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: AbstractSRTD, P}
    SRTD = agent.model

    # Update eligibility trace
    SRTD.trace[:] = SRTD.trace .* (SRTD.γ * SRTD.λ)
    SRTD.trace[s] = SRTD.trace[s] + 1

    # Update M first
    # δM is our state misprediction: Iₛ + γ M_s' vs. M_s
    # Move each state's prediction in the direction of δM based on the trace
    δM = SRTD.ident[s, :] + SRTD.γ .* SRTD.M[s′, :] - SRTD.M[s, :]
    # The Iₛ cancels out, leaving us with γM_s′.
    # M_s should already be an average of γM_s′1, γM_s′2, etc.
    for sx in 1:length(SRTD)
        SRTD.M[sx, :] = SRTD.M[sx, :] + (SRTD.αM * SRTD.trace[sx]) .* δM
    end
    # Non-trace update:
    # SRTD.M[s, :] = (SRTD.αM * (SRTD.ident[s, :] + SRTD.γ * SRTD.M[s′, :])) + ((1 - SRTD.αM) * SRTD.M[s, :])

    # Keep terminal states from biasing V
    # Not necessary so long as M matrix is set to zero for terminal states
    # SRTD.w[agent.env.terminal_states] .= 0

    # Update V again
    SRTD.V[:] = SRTD.M * SRTD.w

    # Update Q to include the discount
    # This isn't technically correct, since we're ignoring reward from the current state,
    # but as long as all rewards are terminal, it should be fine
    SRTD.Q[:] = SRTD.γ * SRTD.V
end

function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: AbstractSRTD, P}
    SRTD = agent.model

    # Update eligibility trace
    SRTD.trace[:] = SRTD.trace .* (SRTD.γ * SRTD.λ)
    SRTD.trace[s] = SRTD.trace[s] + 1

    # Update M first
    # δM is our state misprediction: Iₛ + γ M_s' vs. M_s
    # Move each state's prediction in the direction of δM based on the trace
    δM = SRTD.ident[s, :] + SRTD.γ .* SRTD.M[s′, :] - SRTD.M[s, :]
    # The Iₛ cancels out, leaving us with γM_s′.
    # M_s should already be an average of γM_s′1, γM_s′2, etc.
    for sx in 1:length(SRTD)
        SRTD.M[sx, :] = SRTD.M[sx, :] + (SRTD.αM * SRTD.trace[sx]) .* δM
    end
    # Non-trace update:
    # SRTD.M[s, :] = (SRTD.αM * (SRTD.ident[s, :] + SRTD.γ * SRTD.M[s′, :])) + ((1 - SRTD.αM) * SRTD.M[s, :])

    # Update V with the new M before our w update
    SRTD.V[:] = SRTD.M * SRTD.w

    # Update w second
    # The norm allows a consistent interpretation of learning rate parameter
    # M[s,:] is normed to sum to 1, meaning alpha of 1 will fully account for
    # the prediction error, e.g. reward + SRTD.γ * SRTD.V[s′] - SRTD.V[s] should now sum to 0
    δ = reward + SRTD.γ * SRTD.V[s′] - SRTD.V[s]
    feature_rep = SRTD.M[s, :]
    feature_rep_normed = feature_rep ./ (feature_rep' * feature_rep)
    w_diff = (SRTD.α * δ) * feature_rep_normed
    SRTD.w[:] = SRTD.w + w_diff

    # Keep terminal states from biasing V
    # Not necessary so long as M matrix is set to zero for terminal states
    # SRTD.w[agent.env.terminal_states] .= 0

    # Update V again
    SRTD.V[:] = SRTD.M * SRTD.w

    # Update Q to include the discount
    # This isn't technically correct, since we're ignoring reward from the current state,
    # but as long as all rewards are terminal, it should be fine
    SRTD.Q[:] = SRTD.γ * SRTD.V
end

function update_model_end!(::StateAgent{E, M, P}, ::Episode) where {E, M <: AbstractSRTD, P} end