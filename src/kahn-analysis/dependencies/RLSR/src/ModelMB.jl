"""
R is a vector of rewards when transitioning /to/ a state, if we think
of the state as the action.

That is, s → a → s′ induces a reward based on s′, which is estimated by R[s′]

The Q-value of an action should then be R[s′] + γ * V[s′]

"""
abstract type AbstractMB <: AbstractStateModel end
mutable struct MBModel <: AbstractMB
    V::Vector{Float64}
    Q::Vector{Float64}
    R::Vector{Float64}
    α::Float64
    γ::Float64
end
function MBModel(env, α, γ)
    n = length(env)
    MBModel(zeros(n), zeros(n), zeros(n), α, γ)
end
MBModel(env; α, γ)::MBModel = MBModel(env, α, γ)
function model_name(model::M) where {M <: AbstractMB} "MB" end

function MBSoftmax(env; α, γ, β)
    MB = MBModel(env, α, γ)
    policy = PolicySoftmax(β)
    StateAgent(env, MB, policy)
end

function MBTwoStepSoftmax(env; α, γ, β1, β2)
    MB = MBModel(env, α, γ)
    policy = PolicyTwoStepSoftmax(β1, β2)
    StateAgent(env, MB, policy)
end

function MBGreedy(env; α, γ)
    MB = MBModel(env, α, γ)
    policy = PolicyGreedy()
    StateAgent(env, MB, policy)
end

function MB_ϵ_Greedy(env; α, γ, ϵ)
    MB = MBModel(env, α, γ)
    policy = Policy_ϵ_Greedy(ϵ)
    StateAgent(env, MB, policy)
end

function update_model_start!(agent::StateAgent{E, M, P}) where {E, M <: AbstractMB, P} end

function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: AbstractMB, P} end

function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: AbstractMB, P}
    agent.model.R[s′] = (agent.model.α * reward) + ((1 - agent.model.α) * agent.model.R[s′])
    agent.model.Q[s′] = agent.model.V[s′] * agent.model.γ + agent.model.R[s′]
end

function update_model_end!(agent::StateAgent{E, M, P}, ::Episode) where {E, M <: AbstractMB, P}
    value_iteration!(agent)
end

"""
Off-policy evaluation (greedy)
"""
function value_iteration!(agent::StateAgent{E, M, P}) where {E, M <: AbstractMB, P}
    nonterminal = findall(.!agent.env.terminal_states)
    while true
        diff = 0.0
        for state in nonterminal
            neighbors = find_neighbors(agent.env, state)
            Q_ests = agent.model.Q[neighbors]
            V_est_new = maximum(Q_ests) 
            diff = max(diff, abs(V_est_new - agent.model.V[state]))
            agent.model.V[state] = V_est_new
            # The Q-value of an action that arrives in this state
            agent.model.Q[state] = V_est_new * agent.model.γ + agent.model.R[state]
        end
        if diff < 1e-6
            break
        end
    end
end

"""
The below is for on-policy evaluation
"""
# function value_iteration!(agent::StateAgent{E, M, P}) where {E, M <: AbstractMB, P <: Policy_ϵ_Greedy}
#     nonterminal = findall(.!agent.env.terminal_states)
#     while true
#         diff = 0.0
#         for state in nonterminal
#             neighbors = find_neighbors(agent.env, state)
#             Q_ests = agent.model.Q[neighbors]
#             V_est_new = agent.policy.ϵ * mean(Q_ests) + (1 - agent.policy.ϵ) * maximum(Q_ests) 
#             diff = max(diff, abs(V_est_new - agent.model.V[state]))
#             agent.model.V[state] = V_est_new
#             agent.model.Q[state] = V_est_new * agent.model.γ + agent.model.R[state]
#         end
#         if diff < 1e-6
#             break
#         end
#     end
# end

# function value_iteration!(agent::StateAgent{E, M, P}) where {E, M <: AbstractMB, P <: PolicyGreedy}
#     nonterminal = findall(.!agent.env.terminal_states)
#     while true
#         diff = 0.0
#         for state in nonterminal
#             neighbors = find_neighbors(agent.env, state)
#             Q_ests = agent.model.Q[neighbors]
#             V_est_new = maximum(Q_ests) 
#             diff = max(diff, abs(V_est_new - agent.model.V[state]))
#             agent.model.V[state] = V_est_new
#             agent.model.Q[state] = V_est_new * agent.model.γ + agent.model.R[state]
#         end
#         if diff < 1e-6
#             break
#         end
#     end
# end

# function value_iteration!(agent::StateAgent{E, M, P}) where {E, M <: AbstractMB, P <: PolicySoftmax}
#     nonterminal = findall(.!agent.env.terminal_states)
#     while true
#         diff = 0.0
#         for state in nonterminal
#             neighbors = find_neighbors(agent.env, state)
#             Q_ests = agent.model.Q[neighbors]
#             weights = exp.(agent.policy.β * Q_ests)
#             weights_normed = weights ./ sum(weights)
#             V_est_new = sum(Q_ests .* weights_normed)
#             diff = max(diff, abs(V_est_new - agent.model.V[state]))
#             agent.model.V[state] = V_est_new
#             agent.model.Q[state] = V_est_new * agent.model.γ + agent.model.R[state]
#         end
#         if diff < 1e-6
#             break
#         end
#     end
# end