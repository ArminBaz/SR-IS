using Logging
using LinearAlgebra
using SpecialFunctions  # for erf
using DataFrames
using StatsFuns
using RLSR
using EM

function Base.length(x::DataFrame)
    nrow(x)
end

"""
Fits a β for TD, MB, and SR-IS (importance-sampled SR with soft log-sum-exp value function).

SR-IS differs from standard SR in two ways:
1. Value function: V[island] = λ * log(Σ_boat DR[island,boat] * exp(R[boat]/λ))
   instead of V = M * R
2. Update: DR[s,:] = (1-αM)*DR[s,:] + αM*w*(I_s + γ_SR*DR[s',:])
   where w = (1/2) / policy_prob is an importance sampling weight.

λ controls the softness of the value function (small λ → argmax, large λ → average).
γ_SR is the discount used in the DR update (default 0.9, matching simulation).
"""
function lik_td0_td1_mb_sris_blockwise_twotd(data, βTD0_1, βTD0_2, βTD1_1, βTD1_2, βMB1, βMB2, βSR_IS1, βSR_IS2, βBoat, island_stay_bias, boat_stay_bias, α1Home::T, α1Away, α2Home, α2Away, αM1, αM2, initial_V, decay_island, decay_boat, λ::Real, γ_SR::Real, rewscaled::Bool, decorrelateαβ::Bool, add_betas::Bool, record::Bool) where T
    TD0_V = zeros(T, 7)
    TD1_V = zeros(T, 7)
    MB_Q  = zeros(T, 7)
    SR_IS_Q = zeros(T, 7)
    βavg_Q  = zeros(T, 7)
    if (rewscaled)
        baseline_V = initial_V * 2.0 - 1.0
    else
        baseline_V = initial_V
    end
    TD0_V .= baseline_V
    TD1_V .= baseline_V
    MB_Q  .= baseline_V
    SR_IS_Q .= baseline_V
    βavg_Q  .= baseline_V

    ident = Matrix{T}(LinearAlgebra.I, 7, 7)

    # DR matrix (discounted representation, replaces M for SR-IS)
    # Initialized with task-consistent random-walk structure.
    # Terminal states (boats 4-7) get 1/(1-γ_SR) self-loops.
    DR = zeros(T, 7, 7)
    DR[1, 2] = T(0.5); DR[1, 3] = T(0.5)
    DR[1, 4] = T(0.25); DR[1, 5] = T(0.25); DR[1, 6] = T(0.25); DR[1, 7] = T(0.25)
    DR[2, 4] = T(0.5); DR[2, 5] = T(0.5)
    DR[3, 6] = T(0.5); DR[3, 7] = T(0.5)
    for t in 4:7
        DR[t, t] = T(1.0 / (1.0 - γ_SR))
    end

    # Scratch vector for DR update target
    dr_target = zeros(T, 7)

    decay_vec = zeros(T, 7)

    if rewscaled
        MB_SOFTMAX_PARAM = 10.0
    else
        MB_SOFTMAX_PARAM = 20.0
    end

    βTD0   = βTD0_1
    βTD1   = βTD1_1
    βMB    = βMB1
    βSR_IS = βSR_IS1
    αHome  = α1Home
    αAway  = α1Away
    αM     = αM1

    ntrials = length(data)
    if record
        TD0_V_record   = zeros(ntrials, 7)
        TD1_V_record   = zeros(ntrials, 7)
        MB_Q_record    = zeros(ntrials, 7)
        SR_IS_Q_record = zeros(ntrials, 7)
        βavg_Q_record  = zeros(ntrials, 7)
    end

    lik = 0.

    prev_state2       = 0
    prev_state_2_boat = 0
    prev_state_3_boat = 0
    prev_trial        = 0

    trial        = data.trial
    state1       = data.state1
    state2       = data.state2
    state3       = data.state3
    if rewscaled
        reward = data.rewscaled
    else
        reward = data.reward
    end
    rwd_swap_type = data.rwd_swap_type

    @inbounds for i in eachindex(state1)
        # Reset if new session
        if trial[i] < prev_trial
            TD0_V .= baseline_V
            TD1_V .= baseline_V
            MB_Q  .= baseline_V
            SR_IS_Q .= baseline_V
            βavg_Q  .= baseline_V
            DR[1, 2] = T(0.5); DR[1, 3] = T(0.5)
            DR[1, 4] = T(0.25); DR[1, 5] = T(0.25); DR[1, 6] = T(0.25); DR[1, 7] = T(0.25)
            DR[2, 4] = T(0.5); DR[2, 5] = T(0.5); DR[2, 6:7] .= T(0.0)
            DR[3, 4:5] .= T(0.0); DR[3, 6] = T(0.5); DR[3, 7] = T(0.5)
            DR[1, 1] = T(0.0); DR[2, 1:3] .= T(0.0); DR[3, 1:3] .= T(0.0)
            for t in 4:7
                DR[t, t] = T(1.0 / (1.0 - γ_SR))
            end
            prev_state2       = 0
            prev_state_2_boat = 0
            prev_state_3_boat = 0
            prev_trial        = 0
        end

        if (rwd_swap_type[i] == "good")
            βTD0   = βTD0_1
            βTD1   = βTD1_1
            βMB    = βMB1
            βSR_IS = βSR_IS1
            αHome  = α1Home
            αAway  = α1Away
            αM     = αM1
        else
            if add_betas
                βTD0   = βTD0_1 + βTD0_2
                βTD1   = βTD1_1 + βTD1_2
                βMB    = βMB1 + βMB2
                βSR_IS = βSR_IS1 + βSR_IS2
            else
                βTD0   = βTD0_2
                βTD1   = βTD1_2
                βMB    = βMB2
                βSR_IS = βSR_IS2
            end
            αHome = α2Home
            αAway = α2Away
            αM    = αM2
        end

        if record
            TD0_V_record[i, :]   .= TD0_V
            TD1_V_record[i, :]   .= TD1_V
        end

        # Full traversals
        if (state1[i] == 1)
            # MB Value Calculation
            MB_Q[2] = softmaximum(TD0_V[4], TD0_V[5], MB_SOFTMAX_PARAM)
            MB_Q[3] = softmaximum(TD0_V[6], TD0_V[7], MB_SOFTMAX_PARAM)

            # SR-IS Value Calculation (soft log-sum-exp using DR matrix)
            # V[island] = λ * log(Σ_boat DR[island,boat] * exp(TD0_V[boat]/λ))
            # Implemented via log-sum-exp trick: log(a*e^x + b*e^y) = logsumexp([log(a)+x, log(b)+y])
            let dr24 = max(DR[2, 4], T(1e-30)), dr25 = max(DR[2, 5], T(1e-30)),
                dr36 = max(DR[3, 6], T(1e-30)), dr37 = max(DR[3, 7], T(1e-30)),
                inv_λ = T(1.0) / T(λ)
                SR_IS_Q[2] = T(λ) * logsumexp((log(dr24) + TD0_V[4] * inv_λ,
                                                log(dr25) + TD0_V[5] * inv_λ))
                SR_IS_Q[3] = T(λ) * logsumexp((log(dr36) + TD0_V[6] * inv_λ,
                                                log(dr37) + TD0_V[7] * inv_λ))
            end

            # Q-value averaging (first-level: island choice)
            βavg_Q[2:3] .= (βTD0 .* view(TD0_V, 2:3)) .+ (βTD1 .* view(TD1_V, 2:3)) .+
                            (βMB .* view(MB_Q, 2:3)) .+ (βSR_IS .* view(SR_IS_Q, 2:3))
            # Island stay bias
            if (prev_state2 > 0)
                βavg_Q[prev_state2] += island_stay_bias
            end

            # Second-level Q values (boat choice uses TD0 values, same as standard)
            βavg_Q[4:7] .= (βBoat .* view(TD0_V, 4:7))

            # First-level likelihood
            lik += βavg_Q[state2[i]] - logsumexp(view(βavg_Q, 2:3))

            if state3[i] != -1  # Non-truncated trial
                # Second-level choices; capture lik_2nd for IS weight
                lik_2nd = zero(T)
                if state2[i] == 2
                    if (prev_state_2_boat > 0)
                        βavg_Q[prev_state_2_boat] += boat_stay_bias
                    end
                    lik_2nd = βavg_Q[state3[i]] - logsumexp(view(βavg_Q, 4:5))
                    lik += lik_2nd
                else
                    if (prev_state_3_boat > 0)
                        βavg_Q[prev_state_3_boat] += boat_stay_bias
                    end
                    lik_2nd = βavg_Q[state3[i]] - logsumexp(view(βavg_Q, 6:7))
                    lik += lik_2nd
                end

                # TD Updates
                if decorrelateαβ
                    TD0_V[state2[i]] = (1 - αAway) * TD0_V[state2[i]] + TD0_V[state3[i]]
                    TD0_V[state3[i]] = (1 - αAway) * TD0_V[state3[i]] + reward[i]
                else
                    TD0_V[state2[i]] = (1 - αAway) * TD0_V[state2[i]] + αAway * TD0_V[state3[i]]
                    TD0_V[state3[i]] = (1 - αAway) * TD0_V[state3[i]] + αAway * reward[i]
                end

                if decorrelateαβ
                    TD1_V[state2[i]] = (1 - αAway) * TD1_V[state2[i]] + TD1_V[state3[i]]
                    δTD1 = reward[i] / αAway - TD1_V[state3[i]]
                    TD1_V[state3[i]] = TD1_V[state3[i]] + αAway * δTD1
                    TD1_V[state2[i]] = TD1_V[state2[i]] + δTD1
                else
                    TD1_V[state2[i]] = (1 - αAway) * TD1_V[state2[i]] + αAway * TD1_V[state3[i]]
                    δTD1 = reward[i] - TD1_V[state3[i]]
                    TD1_V[state3[i]] = TD1_V[state3[i]] + αAway * δTD1
                    TD1_V[state2[i]] = TD1_V[state2[i]] + αAway * δTD1
                end

                # DR update with IS weight
                # IS weight: w = (1/2) / max(policy_prob, ε), clipped at 10
                # policy_prob = exp(lik_2nd) = second-level softmax probability
                w = min(T(0.5) / max(exp(lik_2nd), T(1e-10)), T(10.0))
                # target = I[state2,:] + γ_SR * DR[state3,:]
                # DR[state2,:] = (1-αM)*DR[state2,:] + αM*w*target
                dr_target .= view(ident, state2[i], :) .+ T(γ_SR) .* view(DR, state3[i], :)
                view(DR, state2[i], :) .= (T(1.0) - αM) .* view(DR, state2[i], :) .+ αM .* w .* dr_target

                prev_state2 = state2[i]
                prev_trial  = trial[i]
                if state2[i] == 2
                    prev_state_2_boat = state3[i]
                else
                    prev_state_3_boat = state3[i]
                end

                decay_vec[1]   = 1
                decay_vec[2:3] .= decay_island
                decay_vec[4:7] .= decay_boat
                decay_vec[state2[i]] = 1
                decay_vec[state3[i]] = 1
                TD0_V .= decay_vec .* TD0_V .+ (1 .- decay_vec) .* baseline_V
                TD1_V .= decay_vec .* TD1_V .+ (1 .- decay_vec) .* baseline_V
            end
        else  # Passive samples
            if decorrelateαβ
                TD0_V[state1[i]] = (1 - αHome) * TD0_V[state1[i]] + reward[i]
                TD1_V[state1[i]] = (1 - αHome) * TD1_V[state1[i]] + reward[i]
            else
                TD0_V[state1[i]] = (1 - αHome) * TD0_V[state1[i]] + αHome * reward[i]
                TD1_V[state1[i]] = (1 - αHome) * TD1_V[state1[i]] + αHome * reward[i]
            end

            decay_vec[1:3] .= 1
            decay_vec[4:7] .= decay_boat
            decay_vec[state1[i]] = 1
            TD0_V .= decay_vec .* TD0_V .+ (1 .- decay_vec) .* baseline_V
            TD1_V .= decay_vec .* TD1_V .+ (1 .- decay_vec) .* baseline_V
        end

        if record
            βavg_Q_record[i, :]  .= βavg_Q
            MB_Q_record[i, :]    .= MB_Q
            SR_IS_Q_record[i, :] .= SR_IS_Q
        end
    end

    if record
        record_df = DataFrame()
        for i in 1:7
            record_df[!, Symbol("TD0_V$i")]   = TD0_V_record[:, i]
            record_df[!, Symbol("TD1_V$i")]   = TD1_V_record[:, i]
            record_df[!, Symbol("βavg_Q$i")]  = βavg_Q_record[:, i]
            record_df[!, Symbol("MB_Q$i")]    = MB_Q_record[:, i]
            record_df[!, Symbol("SR_IS_Q$i")] = SR_IS_Q_record[:, i]
        end
        return -lik, record_df
    else
        return -lik
    end
end

function lik_td0_td1_mb_sris_blockwise_twotd(data;
    βTD0_1=0.0, βTD0_2=0.0, βTD1_1=0.0, βTD1_2=0.0, βMB1=0.0, βMB2=0.0, βSR_IS1=0.0, βSR_IS2=0.0, βBoat=0.0,
    island_stay_bias=0.0, boat_stay_bias=0.0,
    α1Home=0.0, α1Away=0.0, α2Home=0.0, α2Away=0.0, αM1=0.0, αM2=0.0,
    initial_V=0.5, decay_island=1.0, decay_boat=1.0, λ=0.1, γ_SR=0.9,
    rewscaled=true, decorrelateαβ=false, add_betas=false, record=false)
    lik_td0_td1_mb_sris_blockwise_twotd(data, βTD0_1, βTD0_2, βTD1_1, βTD1_2, βMB1, βMB2, βSR_IS1, βSR_IS2, βBoat, island_stay_bias, boat_stay_bias, α1Home, α1Away, α2Home, α2Away, αM1, αM2, initial_V, decay_island, decay_boat, λ, γ_SR, rewscaled, decorrelateαβ, add_betas, record)
end


"""
Fit β for TD, MB, and SR-IS to behavioral data via EM.

Parameters matching sailing_base_mbsr_blockwise_betas_twotd.jl are identical.
New SR-IS-specific parameters:
  add_λ   – fit λ (soft-value temperature) as a free subject-level parameter
  λ       – fixed value for λ when add_λ=false (default 0.1)
  γ_SR    – discount for DR update; fixed (default 0.9)
"""
function run_tdλ_mb_sris_blockwise_twotd(data; maxiter=200, emtol=1e-3, full=true, extended=false, quiet=false, threads=true, initx=false, nstarts=1,
    add_TDλ1=false,
    add_TDλ2=false,
    add_βTD_1=false,
    add_βTD_2=false,
    add_βMB1=false,
    add_βMB2=false,
    add_βSR_IS1=false,
    add_βSR_IS2=false,
    add_βBoat=false,
    add_island_stay_bias=false,
    add_boat_stay_bias=false,
    add_α1=false,
    add_α2=false,
    separate_home_away=false,
    add_αM1=false,
    add_αM2=false,
    add_initial_V=false,
    add_decay=false,
    separate_decay=false,
    add_λ=false,

    TDλ1=0.0,
    TDλ2=nothing,
    βTD_1=nothing,
    βTD_2=nothing,
    βMB1=nothing,
    βMB2=nothing,
    βSR_IS1=nothing,
    βSR_IS2=nothing,
    βBoat=nothing,
    island_stay_bias=0.0,
    boat_stay_bias=0.0,
    α1Home=nothing,
    α1Away=nothing,
    α2Home=nothing,
    α2Away=nothing,
    αM1=nothing,
    αM2=nothing,
    initial_V=0.5,
    decay_island=1.0,
    decay_boat=1.0,
    λ=0.1,
    γ_SR=0.9,
    rewscaled=true,
    groups=nothing,
    decorrelateαβ=false,
    add_betas=false,

    loocv_data=nothing,
    loocv_subject=nothing,
    )

    subs = unique(data[:,:sub])
    NS   = length(subs)
    X    = ones(NS)

    initbetas = Vector{Float64}()
    initsigma = Vector{Float64}()
    varnames  = Vector{String}()

    if add_TDλ1
        push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "TDλ1")
    end
    if add_TDλ2
        push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "TDλ2")
    end
    if add_βTD_1
        push!(initbetas, 1); push!(initsigma, 5); push!(varnames, "βTD_1")
    end
    if add_βTD_2
        push!(initbetas, 1); push!(initsigma, 5); push!(varnames, "βTD_2")
    end
    if add_βMB1
        push!(initbetas, 1); push!(initsigma, 5); push!(varnames, "βMB1")
    end
    if add_βMB2
        push!(initbetas, 1); push!(initsigma, 5); push!(varnames, "βMB2")
    end
    if add_βSR_IS1
        push!(initbetas, 1); push!(initsigma, 5); push!(varnames, "βSR_IS1")
    end
    if add_βSR_IS2
        push!(initbetas, 1); push!(initsigma, 5); push!(varnames, "βSR_IS2")
    end
    if add_βBoat
        push!(initbetas, 1); push!(initsigma, 5); push!(varnames, "βBoat")
    end
    if add_island_stay_bias
        push!(initbetas, 0); push!(initsigma, 5); push!(varnames, "island_stay_bias")
    end
    if add_boat_stay_bias
        push!(initbetas, 0); push!(initsigma, 5); push!(varnames, "boat_stay_bias")
    end
    if add_α1
        if separate_home_away
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "α1Home")
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "α1Away")
        else
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "α1")
        end
    end
    if add_α2
        if separate_home_away
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "α2Home")
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "α2Away")
        else
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "α2")
        end
    end
    if add_αM1
        push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "αM1")
    end
    if add_αM2
        push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "αM2")
    end
    if add_initial_V
        push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "initial_V")
    end
    if add_decay
        if separate_decay
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "decay_island")
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "decay_boat")
        else
            push!(initbetas, 0); push!(initsigma, 1); push!(varnames, "decay")
        end
    end
    if add_λ
        push!(initbetas, -2); push!(initsigma, 1); push!(varnames, "λ")
    end

    if !isnothing(groups)
        initbetas = hcat(initbetas, zeros(length(initbetas)))
        X = hcat(X, groups)
    end
    initbetasT = Array(initbetas')

    if !quiet
        @show varnames
        @show initbetas
        @show initsigma
        @show X
        flush(stdout)
    end

    function fn(params, data)
        i = 1

        if add_TDλ1
            f_TDλ1 = unitnorm(params[i]); i += 1
        else
            f_TDλ1 = TDλ1
        end
        if add_TDλ2
            f_TDλ2 = unitnorm(params[i]); i += 1
        elseif !isnothing(TDλ2)
            f_TDλ2 = TDλ2
        else
            f_TDλ2 = f_TDλ1
        end

        if add_βTD_1
            f_βTD_1  = params[i]
            f_βTD0_1 = f_βTD_1 * (1 - f_TDλ1)
            f_βTD1_1 = f_βTD_1 * f_TDλ1
            i += 1
        elseif !isnothing(βTD_1)
            f_βTD_1  = βTD_1
            f_βTD0_1 = f_βTD_1 * (1 - f_TDλ1)
            f_βTD1_1 = f_βTD_1 * f_TDλ1
        else
            f_βTD0_1 = 0.0
            f_βTD1_1 = 0.0
        end

        if add_βTD_2
            f_βTD0_2 = params[i] * (1 - f_TDλ2)
            f_βTD1_2 = params[i] * f_TDλ2
            i += 1
        elseif !isnothing(βTD_2)
            f_βTD0_2 = βTD_2 * (1 - f_TDλ2)
            f_βTD1_2 = βTD_2 * f_TDλ2
        elseif add_βTD_1 | !isnothing(βTD_1)
            f_βTD0_2 = f_βTD_1 * (1 - f_TDλ2)
            f_βTD1_2 = f_βTD_1 * f_TDλ2
        else
            f_βTD0_2 = f_βTD0_1
            f_βTD1_2 = f_βTD1_1
        end

        if add_βMB1
            f_βMB1 = params[i]; i += 1
        elseif !isnothing(βMB1)
            f_βMB1 = βMB1
        else
            f_βMB1 = 0.0
        end
        if add_βMB2
            f_βMB2 = params[i]; i += 1
        elseif !isnothing(βMB2)
            f_βMB2 = βMB2
        else
            f_βMB2 = f_βMB1
        end

        if add_βSR_IS1
            f_βSR_IS1 = params[i]; i += 1
        elseif !isnothing(βSR_IS1)
            f_βSR_IS1 = βSR_IS1
        else
            f_βSR_IS1 = 0.0
        end
        if add_βSR_IS2
            f_βSR_IS2 = params[i]; i += 1
        elseif !isnothing(βSR_IS2)
            f_βSR_IS2 = βSR_IS2
        else
            f_βSR_IS2 = f_βSR_IS1
        end

        if add_βBoat
            f_βBoat = params[i]; i += 1
        elseif !isnothing(βBoat)
            f_βBoat = βBoat
        else
            f_βBoat = 0.0
        end

        if add_island_stay_bias
            f_island_stay_bias = params[i]; i += 1
        else
            f_island_stay_bias = island_stay_bias
        end
        if add_boat_stay_bias
            f_boat_stay_bias = params[i]; i += 1
        else
            f_boat_stay_bias = boat_stay_bias
        end

        if add_α1
            if separate_home_away
                f_α1Home = unitnorm(params[i]); i += 1
                f_α1Away = unitnorm(params[i]); i += 1
            else
                f_α1Home = unitnorm(params[i])
                f_α1Away = f_α1Home
                i += 1
            end
        elseif !isnothing(α1Home)
            f_α1Home = α1Home
            f_α1Away = α1Away
        else
            f_α1Home = 0.0
            f_α1Away = 0.0
        end
        if add_α2
            if separate_home_away
                f_α2Home = unitnorm(params[i]); i += 1
                f_α2Away = unitnorm(params[i]); i += 1
            else
                f_α2Home = unitnorm(params[i])
                f_α2Away = f_α2Home
                i += 1
            end
        elseif !isnothing(α2Home)
            f_α2Home = α2Home
            f_α2Away = α2Away
        else
            f_α2Home = f_α1Home
            f_α2Away = f_α1Away
        end

        if add_αM1
            f_αM1 = unitnorm(params[i]); i += 1
        elseif !isnothing(αM1)
            f_αM1 = αM1
        else
            f_αM1 = 0.0
        end
        if add_αM2
            f_αM2 = unitnorm(params[i]); i += 1
        elseif !isnothing(αM2)
            f_αM2 = αM2
        else
            f_αM2 = f_αM1
        end

        if add_initial_V
            f_initial_V = unitnorm(params[i]); i += 1
        else
            f_initial_V = initial_V
        end

        if add_decay
            if separate_decay
                f_decay_island = unitnorm(params[i]); i += 1
                f_decay_boat   = unitnorm(params[i]); i += 1
            else
                f_decay_island = unitnorm(params[i])
                f_decay_boat   = unitnorm(params[i])
                i += 1
            end
        else
            f_decay_island = decay_island
            f_decay_boat   = decay_boat
        end

        # λ: if fit, use softplus to keep positive; if fixed, use the kwarg value
        if add_λ
            f_λ = log1p(exp(params[i])); i += 1  # softplus, ensures λ > 0
        else
            f_λ = λ
        end

        return lik_td0_td1_mb_sris_blockwise_twotd(data, f_βTD0_1, f_βTD0_2, f_βTD1_1, f_βTD1_2, f_βMB1, f_βMB2, f_βSR_IS1, f_βSR_IS2, f_βBoat, f_island_stay_bias, f_boat_stay_bias, f_α1Home, f_α1Away, f_α2Home, f_α2Away, f_αM1, f_αM2, f_initial_V, f_decay_island, f_decay_boat, f_λ, γ_SR, rewscaled, decorrelateαβ, add_betas, false)
    end

    if !isnothing(loocv_data)
        if !isnothing(loocv_subject)
            return loocv_singlesubject(data,subs,loocv_subject,loocv_data.x,X,loocv_data.betas,loocv_data.sigma,fn; emtol, full, maxiter)
        else
            return loocv(data,subs,loocv_data.x,X,loocv_data.betas,loocv_data.sigma,fn; emtol, full, maxiter)
        end
    else
        startx = []
        if initx
            startx = eminits(data,subs,X,initbetasT,initsigma,fn;nstarts=nstarts,threads=threads)
        end

        (betas,sigma,x,l,h,opt_rec) = em(data,subs,X,initbetasT,initsigma,fn; emtol=emtol, full=full, maxiter=maxiter, quiet=quiet, threads=threads, startx=startx);
        if extended
            try
                @info "Running emerrors"
                (standarderrors,pvalues,covmtx) = emerrors(data,subs,x,X,h,betas,sigma,fn)
                return EMResultsExtended(varnames,betas,sigma,x,l,h,opt_rec,standarderrors,pvalues,covmtx)
            catch err
                if isa(err, SingularException) || isa(err, DomainError) || isa(err, ArgumentError) || isa(err, LoadError)
                    @warn err
                    @warn "emerrors failed to run. Re-check fitting. Returning EMResults"
                    return EMResults(varnames,betas,sigma,x,l,h,opt_rec)
                else
                    rethrow()
                end
            end
        else
            return EMResults(varnames,betas,sigma,x,l,h,opt_rec)
        end
    end
end
