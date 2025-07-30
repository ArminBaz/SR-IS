struct WeightedModel <: AbstractWeightedStateModel
    V::Vector{Float64}
    Q::Vector{Float64}
    models
    weights::Vector{Float64}
    function WeightedModel(V, Q, models, weights)
        if any(weights .< 0) || !isapprox(sum(weights), 1)
            error("invalid model weights: $weights")
        end
        if length(models) ≠ length(weights)
            error("number of models ($length(models)) ≠ number of weights ($length(weights))")
        end
        for m ∈ models
            if length(m.V) ≠ length(V)
                error("Different number of states: $m")
            end
        end
        new(V, Q, models, weights)
    end
end
function WeightedModel(models, weights)
    WeightedModel(zeros(length(models[1].V)), zeros(length(models[1].V)), models, weights)
end
function model_name(model::M; show_weights=false, kwargs...) where {M <: AbstractWeightedStateModel}
    name = "WeightedModel: ["
    for (i, (m, w)) ∈ enumerate(zip(model.models, model.weights))
        if i > 1
            name *= " / "
        end
        if show_weights
            name *= string(w) * " " * model_name(m, kwargs...)
        else
            name *= model_name(m, kwargs...)
        end
    end
    name * "]"
end

function WeightedModel_ϵ_Greedy(env, models, weights; ϵ)
    model = WeightedModel(models, weights)
    policy = Policy_ϵ_Greedy(ϵ)
    StateAgent(env, model, policy)
end

function WeightedModelSoftmax(env, models, weights; β)
    model = WeightedModel(models, weights)
    policy = PolicySoftmax(β)
    StateAgent(env, model, policy)
end

function WeightedModelTwoStepSoftmax(env, models, weights; β1, β2)
    model = WeightedModel(models, weights)
    policy = PolicyTwoStepSoftmax(β1, β2)
    StateAgent(env, model, policy)
end

function WeightedModelGreedy(env, models, weights)
    model = WeightedModel(models, weights)
    policy = PolicyGreedy()
    StateAgent(env, model, policy)
end

function reaverage_models!(agent::StateAgent{E, M, P}) where {E, M <: WeightedModel, P}
    agent.model.V[:] = sum([m.V * w for (m, w) in zip(agent.model.models, agent.model.weights)])
    agent.model.Q[:] = sum([m.Q * w for (m, w) in zip(agent.model.models, agent.model.weights)])
end

function update_model_start!(agent::StateAgent{E, M, P}) where {E, M <: WeightedModel, P}
    for m ∈ agent.model.models
        update_model_start!(StateAgent(agent.env, m, agent.policy))
    end
    reaverage_models!(agent)
end

function update_model_step!(agent::StateAgent{E, M, P}, s::Int, reward::Real, s′::Int) where {E, M <: WeightedModel, P}
    for m ∈ agent.model.models
        update_model_step!(StateAgent(agent.env, m, agent.policy), s, reward, s′)
    end
    reaverage_models!(agent)
end

function update_model_step_blind!(agent::StateAgent{E, M, P}, s::Int, s′::Int) where {E, M <: WeightedModel, P}
    for m ∈ agent.model.models
        update_model_step_blind!(StateAgent(agent.env, m, agent.policy), s, s′)
    end
    reaverage_models!(agent)
end

function update_model_end!(agent::StateAgent{E, M, P}, episode::Episode) where {E, M <: WeightedModel, P}
    for m ∈ agent.model.models
        update_model_end!(StateAgent(agent.env, m, agent.policy), episode)
    end
    reaverage_models!(agent)
end