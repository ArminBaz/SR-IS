struct StateAgent{E <: AbstractEnv, M <: AbstractStateModel, P <: AbstractPolicy} <: AbstractAgent
    env::E
    model::M
    policy::P
end

function StateAgent(model, policy)
    StateAgent(model.env, model, policy)
end

struct ActionAgent{E <: AbstractEnv, M <: AbstractModel, P <: AbstractPolicy} <: AbstractAgent
    env::E
    model::M
    policy::P
end

function ActionAgent(model, policy)
    ActionAgent(model.env, model, policy)
end

"""
    active_episode!(agent, s)

Simulate an episode starting from state `s` and until reaching a terminal state.

Return an `episode` of the sequence of states and rewards

For a StateAgent: update each step with s[t], r[t], s[t+1]
"""
### ARMIN EDIT : THIS IS WHAT IS BEING USED
function active_episode!(agent::StateAgent, s::Int)
    S = [s]
    R = zeros(1)
    update_model_start!(agent)
    while true
        # Choose an action
        s′ = sample_successor(agent, s)
        if isnothing(s′)
            break
        end
        append!(S, s′)

        # Reward
        r = sample_reward(agent.env, s′)
        append!(R, r)

        update_model_step!(agent, s, r, s′)
        s = s′
    end
    episode = Episode(S, R)
    update_model_end!(agent, episode)
    episode
end

"""
    active_episode!(agent, s)

Simulate an episode starting from state `s` and until reaching a terminal state.

Return an `episode` of the sequence of states and rewards

For an ActionAgent: update each step with s[t-1], s[t], r[t], s[t+1]
"""
function active_episode!(agent::ActionAgent, s::Int)
    S = [s]
    R = []
    s_prev = nothing
    update_model_start!(agent)
    while true
        # Reward
        r = sample_reward(agent.env, s)
        append!(R, r)

        # Choose an action
        s′ = sample_successor(agent, s)
        update_model_step!(agent, s_prev, s, r, s′)

        if isnothing(s′)
            break
        end
        append!(S, s′)

        s_prev = s
        s = s′
    end
    episode = Episode(S, R)
    update_model_end!(agent, episode)
    episode
end

"""
    passive_episode!(agent, episode)

Replay the provided episode, updating the agent's model appropriately.

For a StateAgent: update each step with s[t], r[t], s[t+1]
"""
function passive_episode!(agent::StateAgent, episode::Episode)
    s = episode.S[1]
    r = episode.R[1]
    update_model_start!(agent)
    for (s′, r) in episode[2:end]
        update_model_step!(agent, s, r, s′)
        s = s′
    end
    update_model_end!(agent, episode)
end

"""
    passive_episode!(agent, episode)

Replay the provided episode, updating the agent's model appropriately.

For an ActionAgent: update each step with s[t-1], s[t], r[t], s[t+1]
"""
function passive_episode!(agent::ActionAgent, episode::Episode)
    s = episode.S[1]
    r = 0
    s_prev = nothing
    S = vcat(episode.S[2:end], nothing)
    update_model_start!(agent)
    for (s′, r) in zip(S, episode.R)
        update_model_step!(agent, s_prev, s, r, s′)
        s_prev = s
        s = s′
    end
    update_model_end!(agent, episode)
end

"""
    blind_episode(agent, s)

Simulate an episode starting from state `s` and until reaching a terminal state.

However, the agent is not updated with reward information.

Return an `episode` of the sequence of states and rewards
"""
function blind_episode(agent::AbstractAgent, s::Int)
    S = [s]
    R = zeros(1)
    while true
        # Choose an action
        s′ = sample_successor(agent, s)
        if isnothing(s′)
            break
        end
        append!(S, s′)

        # Reward
        r = sample_reward(agent.env, s′)
        append!(R, r)

        update_model_step_blind!(agent, s, s′)
        s = s′
    end
    Episode(S, R)
end

function update_rewards!(agent::AbstractAgent, R::Vector{Float64})
    update_rewards!(agent.env, R)
end

function update_rewards!(agent::AbstractAgent, R_μ::Vector{Float64}, R_σ::Vector{Float64})
    update_rewards!(agent.env, R_μ, R_σ)
end

function model_name(agent::AbstractAgent; kwargs...) model_name(agent.model; kwargs...) end
function policy_name(agent::AbstractAgent; kwargs...) policy_name(agent.policy; kwargs...) end

"""
    sample_successor(agent, s)

Choose a successor state `s'` to state `s` according to the agent's model and policy.

Return either a state index, or `Nothing` if there are no valid successors."""
function sample_successor(agent::AbstractAgent, s::Int)::Union{Int, Nothing}
    sample_successor(agent.env, agent.model, agent.policy, s)
end