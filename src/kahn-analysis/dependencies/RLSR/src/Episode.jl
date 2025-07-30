struct Observation
    S::Int
    R::Float64
end
Base.length(::Observation) = 2
Base.firstindex(::Observation) = 1
Base.lastindex(::Observation) = 2
function Base.iterate(obs::Observation, state=1)
    if state > 2
        nothing
    else
        if state == 1
            (obs.S, 2)
        elseif state == 2
            (obs.R, 3)
        end
    end
end

struct Episode
    S::Vector{Int}
    R::Vector{Float64}
    Episode(S, R) = length(S) != length(R) ? error("states and rewards have different lengths") : new(S, R)
end
Base.length(episode::Episode) = length(episode.S)
Base.firstindex(::Episode) = 1
Base.lastindex(episode::Episode) = length(episode)
function Base.iterate(episode::Episode, state=1)
    if state > length(episode)
        nothing
    else
        (Observation(episode.S[state], episode.R[state]), state+1)
    end
end
function Base.getindex(episode::Episode, i::Int)
    1 <= i <= length(episode) || throw(BoundsError(episode, i))
    Observation(episode.S[i], episode.R[i])
end
Base.getindex(episode::Episode, I) = [episode[i] for i in I]

function get_total_reward(episodeRecord::Vector{Episode})
    sum([x.R[end] for x in episodeRecord])
end

function get_mean_reward(episodeRecord::Vector{Episode})
    mean([x.R[end] for x in episodeRecord])
end