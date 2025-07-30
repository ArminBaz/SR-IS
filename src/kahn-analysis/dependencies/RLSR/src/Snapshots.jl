# State value based Model Snapshots
struct StateModelSnapshot <: AbstractModelSnapshot
    V::Vector{Float64}
end
function StateModelSnapshot(model::M) where M <: AbstractStateModel
    StateModelSnapshot(copy(model.V))
end

mutable struct StateModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    V::Matrix{Float64}
    n::Int
end
function StateModelRecord(agent, maxsize::Int)::StateModelRecord
    StateModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        0)
end
Base.firstindex(record::StateModelRecord) = 1
Base.lastindex(record::StateModelRecord) = length(record)
Base.length(record::StateModelRecord) = record.n
function Base.push!(record::StateModelRecord, model)
    record.n += 1
    (sx, sy) = size(record.V)
    if record.n > sx
        new_V = zeros(sx * 2, sy)
        new_V[1:sx, :] .= record.V
        record.V = new_V
    end
    record.V[record.n, :] = model.V[:]
end
function Base.iterate(record::StateModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (StateModelSnapshot(record.V[state, :]), state+1)
    end
end
function Base.getindex(record::StateModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    StateModelSnapshot(record.V[i, :])
end
Base.getindex(record::StateModelRecord, I) = StateModelRecord(record.env, record.policy, record.V[I, :], length(I))

# Q Value based Model Snapshots
struct ActionModelSnapshot <: AbstractModelSnapshot
    Q::Vector{Float64}
end
function ActionModelSnapshot(model::M) where M <: AbstractActionModel
    ActionModelSnapshot(copy(model.Q))
end

mutable struct ActionModelRecord{E <: AbstractEnv, P <: AbstractPolicy} <: AbstractRecord
    env::E
    policy::P
    Q::Matrix{Float64}
    n::Int
end
function ActionModelRecord(agent, maxsize::Int)::ActionModelRecord
    ActionModelRecord(
        agent.env,
        agent.policy,
        zeros(maxsize, length(agent.env)),
        0)
end
Base.firstindex(record::ActionModelRecord) = 1
Base.lastindex(record::ActionModelRecord) = length(record)
Base.length(record::ActionModelRecord) = record.n
function Base.push!(record::ActionModelRecord, model)
    record.n += 1
    (sx, sy) = size(record.Q)
    if record.n > sx
        new_Q = zeros(sx * 2, sy)
        new_Q[1:sx, :] .= record.Q
        record.Q = new_Q
    end
    record.Q[record.n, :] = model.Q[:]
end
function Base.iterate(record::ActionModelRecord, state=1)
    if state > length(record)
        nothing
    else
        (ActionModelSnapshot(record.Q[state, :]), state+1)
    end
end
function Base.getindex(record::ActionModelRecord, i::Int)
    1 <= i <= length(record) || throw(BoundsError(record, i))
    ActionModelSnapshot(record.Q[i, :])
end
Base.getindex(record::ActionModelRecord, I) = ActionModelRecord(record.env, record.policy, record.Q[I, :], length(I))

# Generic functions
function ModelSnapshot(model::M) where M <: AbstractActionModel
    ActionModelSnapshop(model)
end
function ModelSnapshot(model::M) where M <: AbstractStateModel
    StateModelSnapshop(model)
end
function ModelRecord(agent::A, maxsize::Int)::StateModelRecord where A <: StateAgent
    StateModelRecord(agent, maxsize)
end
function ModelRecord(agent::A, maxsize::Int)::ActionModelRecord where A <: ActionAgent
    ActionModelRecord(agent, maxsize)
end
function Record(agent, maxsize)
    ModelRecord(agent, maxsize)
end
