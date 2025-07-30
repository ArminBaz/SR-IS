abstract type AbstractSARSA <: AbstractActionModel end
struct SARSAModel <: AbstractSARSA
    Q::Vector{Float64}
    trace::Vector{Float64}
    α::Float64
    λ::Float64
    γ::Float64
end
function SARSAModel(env, α, λ, γ)
    n = ne(env.line_graph)
    SARSAModel(zeros(n), zeros(n), α, λ, γ)
end
SARSAModel(env; α, λ, γ)::SARSAModel = SARSAModel(env, α, λ, γ)
function model_name(model::M) where {M <: AbstractSARSA} "SARSA" end

function SARSA_ϵ_Greedy(env; α, λ, γ, ϵ)
    SARSA = SARSAModel(env, α, λ, γ)
    policy = Policy_ϵ_Greedy(ϵ)
    ActionAgent(env, SARSA, policy)
end

function SARSASoftmax(env; α, λ, γ, β)
    SARSA = SARSAModel(env, α, λ, γ)
    policy = PolicySoftmax(β)
    ActionAgent(env, SARSA, policy)
end

function SARSAGreedy(env; α, λ, γ)
    SARSA = SARSAModel(env, α, λ, γ)
    policy = PolicyGreedy()
    ActionAgent(env, SARSA, policy)
end

function update_model_start!(agent::ActionAgent{E, M, P}) where {E, M <: AbstractSARSA, P}
    agent.model.trace[:] .= 0
end

function update_model_step_blind!(agent::ActionAgent{E, M, P}, s::Int, s′::Int) where {E, M <: AbstractSARSA, P} end

function update_model_step!(agent::ActionAgent{E, M, P}, s_prev, s, reward, s_next) where {E, M <: AbstractSARSA, P}
    # Actions are (s_prev, s) and (s, s_next)
    SARSA = agent.model
    if !isnothing(s_prev)
        # Update eligibility trace
        e_prev = edge_to_ind(agent.env, (s_prev, s))
        Q_next = 0
        if !isnothing(s_next)
            e_next = edge_to_ind(agent.env, (s, s_next))
            Q_next = SARSA.Q[e_next]
        end

        SARSA.trace[:] = SARSA.trace * (SARSA.γ * SARSA.λ)
        SARSA.trace[e_prev] = SARSA.trace[e_prev] + 1

        δ = reward + SARSA.γ * Q_next - SARSA.Q[e_prev]
        SARSA.Q[:] = SARSA.Q + (SARSA.trace * SARSA.α * δ)
    end
end

function update_model_end!(::ActionAgent{E, M, P}, ::Episode) where {E, M <: AbstractSARSA, P}

end