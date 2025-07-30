function Base.length(env::T) where {T <: AbstractGraphEnv}
    nv(env.graph)
end

struct GraphEnv <: AbstractGraphEnv
    graph::SimpleDiGraph
    line_graph::SimpleDiGraph
    adjacency_matrix::Matrix{Float64}
    terminal_states::Vector{Bool}
    neighbors::Dict{Int, Vector{Int}}
    edge_inds::Dict{Tuple{Int, Int}, Int}
    x_coords::Vector{Float64}
    y_coords::Vector{Float64}
    R::Vector{Float64}
end

struct GraphEnvStochastic <: AbstractGraphEnv
    graph::SimpleDiGraph
    line_graph::SimpleDiGraph
    adjacency_matrix::Matrix{Float64}
    terminal_states::Vector{Bool}
    neighbors::Dict{Int, Vector{Int}}
    edge_inds::Dict{Tuple{Int, Int}, Int}
    x_coords::Vector{Float64}
    y_coords::Vector{Float64}
    R_μ::Vector{Float64}
    R_σ::Vector{Float64}
end

struct GraphEnvStochasticBinary <: AbstractGraphEnv
    graph::SimpleDiGraph
    line_graph::SimpleDiGraph
    adjacency_matrix::Matrix{Float64}
    terminal_states::Vector{Bool}
    neighbors::Dict{Int, Vector{Int}}
    edge_inds::Dict{Tuple{Int, Int}, Int}
    x_coords::Vector{Float64}
    y_coords::Vector{Float64}
    isrewarded::Vector{Bool}
    R::Vector{Float64}
end

"""
Return a graph whose nodes correspond to edges in G
"""
function LineGraph(G, edge_inds)
    T_edge = zeros(ne(G), ne(G))
    for node in vertices(G)
        in_n = inneighbors(G, node)
        out_n = outneighbors(G, node)
        w = 1.0 / length(out_n)
        for src in in_n
            for dest in out_n
                T_edge[edge_inds[(src, node)], edge_inds[(node, dest)]] = w
            end
        end
    end
    SimpleDiGraph(T_edge)
end

function edge_to_ind(env, edge)
    env.edge_inds[edge]
end

function GraphEnv(A::Matrix, x_coords::Vector{Float64}, y_coords::Vector{Float64}, R::Vector{Float64};
                  terminal)
    if isnothing(terminal)
        terminal = sum(A; dims=2)
    end
    neighbors = Dict{Int, Vector{Int}}()
    for state in 1:size(A)[1]
        neighbors[state] = findall(A[state, :] .> 0)
    end
    G = SimpleDiGraph(A)
    edge_inds = Dict((edge.src, edge.dst) => i for (i, edge) in enumerate(edges(G)))
    LG = LineGraph(G, edge_inds)
    GraphEnv(G, LG, A, terminal, neighbors, edge_inds, x_coords, y_coords, R)
end

function GraphEnvStochastic(A::Matrix,
                            x_coords::Vector{Float64}, y_coords::Vector{Float64},
                            R_μ::Vector{Float64}, R_σ::Vector{Float64};
                            terminal)
    if isnothing(terminal)
        terminal = sum(A, dims=2)[:, 1] .== 0
    end
    neighbors = Dict{Int, Vector{Int}}()
    for state in 1:size(A)[1]
        neighbors[state] = findall(A[state, :] .> 0)
    end
    G = SimpleDiGraph(A)
    edge_inds = Dict((edge.src, edge.dst) => i for (i, edge) in enumerate(edges(G)))
    LG = LineGraph(G, edge_inds)
    GraphEnvStochastic(G, LG, A, terminal, neighbors, edge_inds, x_coords, y_coords, R_μ, R_σ)
end

function GraphEnvStochasticBinary(A::Matrix,
                                  x_coords::Vector{Float64}, y_coords::Vector{Float64},
                                  isrewarded::Union{Vector{Bool}, Vector{Int}}, R::Vector{Float64};
                                  terminal)
    if isnothing(terminal)
        terminal = sum(A, dims=2)[:, 1] .== 0
    end
    neighbors = Dict{Int, Vector{Int}}()
    for state in 1:size(A)[1]
        neighbors[state] = findall(A[state, :] .> 0)
    end
    G = SimpleDiGraph(A)
    edge_inds = Dict((edge.src, edge.dst) => i for (i, edge) in enumerate(edges(G)))
    LG = LineGraph(G, edge_inds)
    GraphEnvStochasticBinary(G, LG, A, terminal, neighbors, edge_inds, x_coords, y_coords, isrewarded, R)
end

"""
Takes a graph environment and its stochastic transition matrix,
such that all valid transitions from a state sum to 1
"""
function stochastic_matrix(env::E)::Matrix{Float64} where E <: AbstractGraphEnv
    T = copy(env.adjacency_matrix)
    for r in axes(T, 1)
        s = sum(T[r, :])
        if s > 0
            T[r, :] /= s
        end
    end
    T
end

# For now, neighbors are the same as actions
function find_actions(env::E, state::Int) where E <: AbstractGraphEnv
    env.neighbors[state]
end

function find_neighbors(env::E, state::Int) where E <: AbstractGraphEnv
    env.neighbors[state]
end

"""
sample_reward:
Takes an env and a state, and samples a random reward
from the given state
"""
function sample_reward(env::GraphEnvStochastic, state::Int)::Float64
    # println("Sample reward 1")
    rand(Normal{Float64}(env.R_μ[state], env.R_σ[state]))
end

function sample_reward(env::GraphEnv, state::Int)::Float64
    # println("Sample reward 2")
    env.R[state]
end    

## ARMIN EDIT : THIS IS WHAT IS BEING USED
function sample_reward(env::GraphEnvStochasticBinary, state::Int)::Float64
    # println("Sample reward 3")
    if env.isrewarded[state]
        ((rand() < env.R[state]) * 2.0) - 1.0
    else
        0
        ## Armin addition
        # -1
    end
end

# The following are specific to environments where the last four states are rewarded
function get_reward_state(env::GraphEnvStochastic)
    env.R_μ[end-3:end]
end

function get_reward_state(env::GraphEnvStochasticBinary)
    env.R[end-3:end]
end

function update_rewards!(env::GraphEnvStochasticBinary, R::Vector{Float64})
    env.R[end-3:end] = R
    ## Armin addition
    # env.R .= -1  # Set all rewards to -1
    # env.R[end-3:end] = R  # Set last 4 to R
end

function update_rewards!(env::GraphEnvStochastic, R_μ::Vector{Float64}, R_σ::Vector{Float64})
    env.R_μ[end-3:end] = R_μ
    env.R_σ[end-3:end] = R_σ
end

function drift_rewards!(env::GraphEnvStochasticBinary, σ::Float64; lb::Float64, ub::Float64)
    rewards = env.R[end-3:end] + rand(Normal(0, σ), 4)
    rewards = rewards .- 2 * (rewards .> ub) .* (rewards .- ub)
    rewards = rewards .+ 2 * (rewards .< lb) .* (lb .- rewards)
    env.R[end-3:end] = rewards
end

function drift_rewards!(env::GraphEnvStochastic, σ::Float64; lb::Float64, ub::Float64)	
    rewards = env.R_μ[end-3:end] + rand(Normal(0, σ), 4)
    rewards = rewards .- 2 * (rewards .> ub) .* (rewards .- ub)
    rewards = rewards .+ 2 * (rewards .< lb) .* (lb .- rewards)
    env.R_μ[end-3:end] = rewards
end