using LinearAlgebra

"""
eᵛ = core_lrl(T, costs, λ)

Core LRL Function

- T is the transition probability
    - Terminal states are indicated by a self-transition probability of 1
- costs is the cost vector (i.e. negative reward) across all states
- λ scales the control cost

e^(v/λ) = MP e^(r/λ)

- M is the default representation (DR)
- P is T_NT (transitions from non-terminal to terminal states)

See Piray and Daw (2019), "A common model explaining flexible decision
making, grid fields and cognitive control", biorxiv.
"""
function core_lrl(T, costs, λ)
    # implementation of the linear RL model
    # 
    # [pii, expv, M] = core_lrl(T,c)
    # T is the transition probability under the default policy
    # c is the cost vector (i.e. negative reward) across all states
    # pii is the optimized decision policy
    # expv = exp(v), where v is the value function
    # M is the default representation (DR)
    # 
    # [pii, expv, M] = core_lrl(T,c,M)
    # This one relies on a given DR matrix (M) as the input.
    # 
    # See Piray and Daw (2019), "A common model explaining flexible decision
    # making, grid fields and cognitive control", biorxiv.
    # ------------------------

    # reward vector across all states
    r = -costs

    # terminal states
    terminals = diag(T) .== 1

    # computing M
    L = diagm(exp.(costs./λ)) - T
    # Drop terminal rows
    L = L[.!terminals, .!terminals]
    M = L^-1

    # P = T_NT
    P = T[.!terminals, terminals]
    exp_r = exp.(r[terminals]./λ)

    exp_V_N = M*P*exp_r
    exp_V = zeros(typeof(T[1]), length(r))
    exp_V[.!terminals] = exp_V_N
    exp_V[terminals] = exp.(r[terminals]./λ)

    # # A matrix formulation of equation 6 of manuscript
    # G = T*expv
    # zg = expv'./G
    # pii = T.*zg

    # return (pii, expv, M)
    return exp_V
end
"""
In-place implementation of LRL

Requires matrices for intermediate products as arguments

    - T: Transition matrix under the default policy
    - c: Cost vector (i.e. negative reward) across all states
    - λ: Control cost scaling
    - terminals: boolean array indicating whether each state is terminal
    - nonterminals: boolean array indicating whether each state is not terminal
    - L: (NN x NN) temp matrix
    - M: (NN x NN) temp matrix
    - exp_r: (NT) temp vector
    - P_exp_r: (NN) temp vector
    - exp_V: (NN + NT) vector: This is the primary output, represeting e^(v/λ)
    """
function core_lrl!(T, costs, λ, terminals, nonterminals, L, M, exp_r, P_exp_r, exp_V)
    # implementation of the linear RL model
    # 
    # [pii, expv, M] = core_lrl(T,c)
    # T is the transition probability under the default policy
    # c is the cost vector (i.e. negative reward) across all states
    # pii is the optimized decision policy
    # expv = exp(v), where v is the value function
    # M is the default representation (DR)
    # 
    # [pii, expv, M] = core_lrl(T,c,M)
    # This one relies on a given DR matrix (M) as the input.
    # 
    # See Piray and Daw (2019), "A common model explaining flexible decision
    # making, grid fields and cognitive control", biorxiv.
    # ------------------------

    # reward vector across all states
    r = -costs

    # terminal states
    # terminals = diag(T) .== 1

    # computing M
    # Drop terminal rows
    # L .= diagm(exp.(view(costs, nonterminals)./λ)) - view(T, nonterminals, nonterminals)
    L .= -view(T, nonterminals, nonterminals)
    view(L, [1,5,9]) .+= exp.(view(costs, nonterminals)./λ)
    M .= L^-1

    # P = T_NT
    P = view(T, nonterminals, terminals)

    #exp_V = M * P * exp_r
    exp_r .= exp.(view(r, terminals)./λ)
    mul!(P_exp_r, P, exp_r)
    mul!(view(exp_V, nonterminals), M, P_exp_r)
    exp_V[terminals] .= exp_r
end

function Base.length(x::DataFrame)
    nrow(x)
end

transitions = [
    (1, 2),
    (1, 3),
    (2, 4),
    (2, 5),
    (3, 6),
    (3, 7),
]