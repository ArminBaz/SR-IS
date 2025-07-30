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
Fits a β for each of MFTD, MB, and SR, with a separate α used for M and R. Full function.

We're assuming initial_V varies between 0 and 1, and we want to transform it to -1 to 1, so a safe default would be 0.5
"""
function lik_td0_td1_mb_sr_blockwise_twotd(data, βTD0_1, βTD0_2, βTD1_1, βTD1_2, βMB1, βMB2, βSR1, βSR2, βBoat, island_stay_bias, boat_stay_bias, α1Home::T, α1Away, α2Home, α2Away, αM1, αM2, initial_V, decay_island, decay_boat, rewscaled::Bool, decorrelateαβ::Bool, add_betas::Bool, record::Bool) where T
    TD0_V = zeros(T, 7)
    TD1_V = zeros(T, 7)
    SR_V = zeros(T, 7)
    MB_Q = zeros(T, 7)
    SR_Q = zeros(T, 7)
    βavg_Q = zeros(T, 7)
    if (rewscaled)
        baseline_V = initial_V * 2.0 - 1.0 
    else
        baseline_V = initial_V
    end
    TD0_V .= baseline_V
    TD1_V .= baseline_V
    SR_V .= baseline_V
    MB_Q .= baseline_V
    SR_Q .= baseline_V
    βavg_Q .= baseline_V
    
    ident = Matrix{T}(LinearAlgebra.I, 7, 7)
    # Initialize SR matrix to random walk
    # Diag is 1, other transitions are p
    M = Matrix{T}(LinearAlgebra.I, 7, 7)
    M[1, 2:3] .= 0.5
    M[1, 4:7] .= 0.25
    M[2, 4:5] .= 0.5
    M[3, 6:7] .= 0.5
    δM = zeros(T, 7)

    decay_vec = zeros(T, 7)

    if rewscaled
        MB_SOFTMAX_PARAM = 10.0
    else
        MB_SOFTMAX_PARAM = 20.0
    end

    βTD0 = βTD0_1
    βTD1 = βTD1_1
    βMB = βMB1
    βSR = βSR1
    αHome = α1Home
    αAway = α1Away
    αM = αM1

    ntrials = length(data)
    if record
        TD0_V_record = zeros(ntrials, 7)
        TD1_V_record = zeros(ntrials, 7)
        MB_Q_record = zeros(ntrials, 7)
        SR_Q_record = zeros(ntrials, 7)
        βavg_Q_record = zeros(ntrials, 7)
    end 

    lik = 0.
    
    prev_state2 = 0
    prev_state_2_boat = 0
    prev_state_3_boat = 0
    prev_trial = 0

    trial = data.trial
    state1 = data.state1
    state2 = data.state2
    state3 = data.state3
    if rewscaled
        reward = data.rewscaled
    else
        reward = data.reward
    end
    rwd_swap_type = data.rwd_swap_type
    
    @inbounds for i in eachindex(state1)
        # Reset if new session (trial number has decreased)
        if trial[i] < prev_trial
            TD0_V .= baseline_V
            TD1_V .= baseline_V
            SR_V .= baseline_V
            MB_Q .= baseline_V
            SR_Q .= baseline_V
            βavg_Q .= baseline_V
            M[1, 2:3] .= 0.5
            M[1, 4:7] .= 0.25
            M[2, 4:5] .= 0.5
            M[3, 6:7] .= 0.5
            prev_state2 = 0
            prev_state_2_boat = 0
            prev_state_3_boat = 0
            prev_trial = 0
        end

        # Swap options are "good", "bad", "within", "start"
        # "Good" is the only one where second-level policy remains consistent
        if (rwd_swap_type[i] == "good")
            βTD0 = βTD0_1
            βTD1 = βTD1_1
            βMB = βMB1
            βSR = βSR1
            αHome = α1Home
            αAway = α1Away
            αM = αM1
        else
            if add_betas
                βTD0 = βTD0_1 + βTD0_2
                βTD1 = βTD1_1 + βTD1_2
                βMB = βMB1 + βMB2
                βSR = βSR1 + βSR2
            else
                βTD0 = βTD0_2
                βTD1 = βTD1_2
                βMB = βMB2
                βSR = βSR2
            end
            αHome = α2Home
            αAway = α2Away
            αM = αM2
        end

        if record
            TD0_V_record[i, :] .= TD0_V
            TD1_V_record[i, :] .= TD1_V
        end
        
        # Full traversals
        if (state1[i] == 1)
            # MFTD Value Calculations are already done

            # MB Value Calculation
            MB_Q[2] = softmaximum(TD0_V[4], TD0_V[5], MB_SOFTMAX_PARAM)
            MB_Q[3] = softmaximum(TD0_V[6], TD0_V[7], MB_SOFTMAX_PARAM)

            # SR Value Calculation
            SR_V[4:7] .= view(TD0_V, 4:7)
            mul!(SR_Q, M, SR_V)

            # Q-value averaging
            βavg_Q[2:3] .= (βTD0 .* view(TD0_V, 2:3)) .+ (βTD1 .* view(TD1_V, 2:3)) .+ (βMB .* view(MB_Q, 2:3)) .+ (βSR .* view(SR_Q, 2:3))
            # Island stay bias
            if (prev_state2 > 0)
                βavg_Q[prev_state2] += island_stay_bias
            end
            
            βavg_Q[4:7] .= (βBoat .* view(TD0_V, 4:7))

            # Likelihood for top choice
            lik += βavg_Q[state2[i]] - logsumexp(view(βavg_Q, 2:3))
            if state3[i] != -1  # If this wasn't a truncated trial
                # Second-level choices
                if state2[i] == 2
                    if (prev_state_2_boat > 0)
                        # Boat stay bias
                        βavg_Q[prev_state_2_boat] += boat_stay_bias
                    end
                    # Left island
                    lik += βavg_Q[state3[i]] - logsumexp(view(βavg_Q, 4:5))
                else
                    # Right island
                    if (prev_state_3_boat > 0)
                        # Boat stay bias
                        βavg_Q[prev_state_3_boat] += boat_stay_bias
                    end
                    lik += βavg_Q[state3[i]] - logsumexp(view(βavg_Q, 6:7))
                end

                # Temporal Difference Updates
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

                # Update SR M matrix
                δM .= view(ident, state2[i], :) .+ view(M, state3[i], :) .- view(M, state2[i], :)
                view(M, state2[i], :) .+= αM .* δM

                prev_state2 = state2[i]
                prev_trial = trial[i]
                if state2[i] == 2
                    prev_state_2_boat = state3[i]
                else
                    prev_state_3_boat = state3[i]
                end

                decay_vec[1] = 1
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
            βavg_Q_record[i, :] .= βavg_Q
            MB_Q_record[i, :] .= MB_Q
            SR_Q_record[i, :] .= SR_Q
        end
    end
    if record
        record_df = DataFrame()
        for i in 1:7
            record_df[!, Symbol("TD0_V$i")] = TD0_V_record[:, i]
            record_df[!, Symbol("TD1_V$i")] = TD1_V_record[:, i]
            record_df[!, Symbol("βavg_Q$i")] = βavg_Q_record[:, i]
            record_df[!, Symbol("MB_Q$i")] = MB_Q_record[:, i]
            record_df[!, Symbol("SR_Q$i")] = SR_Q_record[:, i]
        end
        return -lik, record_df
    else
        return -lik
    end
end

function lik_td0_td1_mb_sr_blockwise_twotd(data;
    βTD0_1=0.0, βTD0_2=0.0, βTD1_1=0.0, βTD1_2=0.0, βMB1=0.0, βMB2=0.0, βSR1=0.0, βSR2=0.0, βBoat=0.0,
    island_stay_bias=0.0, boat_stay_bias=0.0,
    α1Home=0.0, α1Away=0.0, α2Home=0.0, α2Away=0.0, αM1=0.0, αM2=0.0,
    initial_V=0.5, decay_island=1.0, decay_boat=1.0, rewscaled=true, decorrelateαβ=false, add_betas=false, record=false, verbose=false)
    if verbose
        println("βTD0_1:$(βTD0_1)")
        println("βTD0_2:$(βTD0_2)")
        println("βTD1_1:$(βTD1_1)")
        println("βTD1_2:$(βTD1_2)")
        println("βMB1:$(βMB1)")
        println("βMB2:$(βMB2)")
        println("βSR1:$(βSR1)")
        println("βSR2:$(βSR2)")
        println("βBoat:$(βBoat)")
        println("island_stay_bias:$(island_stay_bias)")
        println("boat_stay_bias:$(boat_stay_bias)")
        println("α1Home:$(α1Home)")
        println("α1Away:$(α1Away)")
        println("α2Home:$(α2Home)")
        println("α2Away:$(α2Away)")
        println("α1M:$(α1M)")
        println("α2M:$(α2M)")
        println("initial_V:$(initial_V)")
        println("decay_island:$(decay_island)")
        println("decay_boat:$(decay_boat)")
        println("$(minimum(data.reward))")
        println("$(maximum(data.reward))")
        println("rewscaled: $(rewscaled)")
        println("decorrelateαβ: $(decorrelateαβ)")
        println("add_betas: $(add_betas)")
        println("record: $(record)")
    end
    lik_td0_td1_mb_sr_blockwise_twotd(data, βTD0_1, βTD0_2, βTD1_1, βTD1_2, βMB1, βMB2, βSR1, βSR2, βBoat, island_stay_bias, boat_stay_bias, α1Home, α1Away, α2Home, α2Away, αM1, αM2, initial_V, decay_island, decay_boat, rewscaled, decorrelateαβ, add_betas, record)
end


"""
Constructs and runs the model.

add_* are flags for whether to include the corresponding parameter in the model.
Otherwise, parameter names represent default values if not fit as a model parameter.
"""
function run_tdλ_mb_sr_blockwise_twotd(data; maxiter=200, emtol=1e-3, full=true, extended=false, quiet=false, threads=true, initx=false, nstarts=1,
    add_TDλ1=false,
    add_TDλ2=false,
    add_βTD_1=false,    
    add_βTD_2=false,    
    add_βMB1=false,
    add_βMB2=false,
    add_βSR1=false,
    add_βSR2=false,
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

    TDλ1=0.0,
    TDλ2=nothing,
    βTD_1=nothing,
    βTD_2=nothing,
    βMB1=nothing,
    βMB2=nothing,
    βSR1=nothing,
    βSR2=nothing,
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
    rewscaled=true,
    groups=nothing,
    decorrelateαβ=false,
    add_betas=false,

    loocv_data=nothing,
    loocv_subject=nothing,
    )

    # data[:, :sub] = data[:, :daynum]
    subs = unique(data[:,:sub]) #in this case subs is just differentiating days rather than rats/subjects
    NS = length(subs) #number of subjects/days
    X = ones(NS) # (group level design matrix); #x group level design matrix...

    # initbetas = [1.]
    # initsigma = [5]
    # varnames = ["βBoat"]
    initbetas = Vector{Float64}()
    initsigma = Vector{Float64}()
    varnames = Vector{String}()

    if add_TDλ1
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "TDλ1")
    end
    if add_TDλ2
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "TDλ2")
    end
    if add_βTD_1
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βTD_1")
    end
    if add_βTD_2
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βTD_2")
    end
    if add_βMB1
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βMB1")
    end
    if add_βMB2
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βMB2")
    end
    if add_βSR1
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βSR1") 
    end
    if add_βSR2
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βSR2") 
    end
    if add_βBoat
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βBoat") 
    end
    if add_island_stay_bias
        push!(initbetas, 0)
        push!(initsigma, 5)
        push!(varnames, "island_stay_bias")
    end
    if add_boat_stay_bias
        push!(initbetas, 0)
        push!(initsigma, 5)
        push!(varnames, "boat_stay_bias")
    end
    if add_α1
        if (separate_home_away)
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "α1Home")
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "α1Away")
        else
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "α1")
        end
    end
    if add_α2
        if (separate_home_away)
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "α2Home")
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "α2Away")
        else
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "α2")
        end
    end
    if add_αM1
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "αM1")
    end
    if add_αM2
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "αM2")
    end
    if add_initial_V
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "initial_V")
    end
    if add_decay
        if separate_decay
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "decay_island")
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "decay_boat")
        else
            push!(initbetas, 0)
            push!(initsigma, 1)
            push!(varnames, "decay")
        end
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
            f_TDλ1 = unitnorm(params[i])
            i += 1
        else
            f_TDλ1 = TDλ1
        end

        if add_TDλ2
            f_TDλ2 = unitnorm(params[i])
            i += 1
        elseif !isnothing(TDλ2)
            f_TDλ2 = TDλ2
        else
            f_TDλ2 = f_TDλ1
        end

        if add_βTD_1
            f_βTD_1 = params[i]
            f_βTD0_1 = f_βTD_1 * (1 - f_TDλ1)
            f_βTD1_1 = f_βTD_1 * f_TDλ1
            i += 1
        elseif !isnothing(βTD_1)
            f_βTD_1 = βTD_1
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
            f_βMB1 = params[i]
            i += 1
        elseif !isnothing(βMB1)
            f_βMB1 = βMB1
        else
            f_βMB1 = 0.0
        end

        if add_βMB2
            f_βMB2 = params[i]
            i += 1
        elseif !isnothing(βMB2)
            f_βMB2 = βMB2
        else
            f_βMB2 = f_βMB1
        end

        if add_βSR1
            f_βSR1 = params[i]
            i += 1
        elseif !isnothing(βSR1)
            f_βSR1 = βSR1
        else
            f_βSR1 = 0.0
        end

        if add_βSR2
            f_βSR2 = params[i]
            i += 1
        elseif !isnothing(βSR2)
            f_βSR2 = βSR2
        else
            f_βSR2 = f_βSR1
        end

        if add_βBoat
            f_βBoat = params[i]
            i += 1
        elseif !isnothing(βBoat)
            f_βBoat = βBoat
        else
            f_βBoat = 0.0
        end 

        if add_island_stay_bias
            f_island_stay_bias = params[i]
            i += 1
        else
            f_island_stay_bias = island_stay_bias
        end

        if add_boat_stay_bias
            f_boat_stay_bias = params[i]
            i += 1
        else
            f_boat_stay_bias = boat_stay_bias
        end

        if add_α1
            if separate_home_away
                f_α1Home = unitnorm(params[i])
                i += 1
                f_α1Away = unitnorm(params[i])
                i += 1
            else
                f_α1Home = unitnorm(params[i])
                f_α1Away = unitnorm(params[i])
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
                f_α2Home = unitnorm(params[i])
                i += 1
                f_α2Away = unitnorm(params[i])
                i += 1
            else
                f_α2Home = unitnorm(params[i])
                f_α2Away = unitnorm(params[i])
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
            f_αM1 = unitnorm(params[i])
            i += 1
        elseif !isnothing(αM1)
            f_αM1 = αM1
        else 
            f_αM1 = 0.0
        end

        if add_αM2
            f_αM2 = unitnorm(params[i])
            i += 1
        elseif !isnothing(αM2)
            f_αM2 = αM2
        else
            f_αM2 = f_αM1
        end

        if add_initial_V
            f_initial_V = unitnorm(params[i])
            i += 1
        else
            f_initial_V = initial_V
        end

        if add_decay
            if separate_decay
                f_decay_island = unitnorm(params[i])
                i += 1
                f_decay_boat = unitnorm(params[i])
                i += 1
            else
                f_decay_island = unitnorm(params[i])
                f_decay_boat = unitnorm(params[i])
                i += 1
            end
        else
            f_decay_island = decay_island
            f_decay_boat = decay_boat
        end

        return lik_td0_td1_mb_sr_blockwise_twotd(data, f_βTD0_1, f_βTD0_2, f_βTD1_1, f_βTD1_2, f_βMB1, f_βMB2, f_βSR1, f_βSR2, f_βBoat, f_island_stay_bias, f_boat_stay_bias, f_α1Home, f_α1Away, f_α2Home, f_α2Away, f_αM1, f_αM2, f_initial_V, f_decay_island, f_decay_boat, rewscaled, decorrelateαβ, add_betas, false)
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

function run_models_tdλ_mb_sr_blockwise_twotd(data, outdir, file_prefix, groups,
                                        add_initial_V, add_TDλ1, add_TDλ2, add_βTD_1, add_βTD_2, add_βMB1, add_βMB2, add_βSR1, add_βSR2, add_βBoat, add_α1, add_α2, add_αM1, add_αM2,
                                        add_decay, separate_decay, add_island_stay_bias, add_boat_stay_bias, separate_home_away, rewscaled, decorrelateαβ, add_betas, initx;
                                        TDλ1=nothing, TDλ2=nothing, βTD_1=nothing, βTD_2=nothing, βMB1=nothing, βMB2=nothing, βSR1=nothing, βSR2=nothing, βBoat=nothing, α1Home=nothing, α1Away=nothing, α2Home=nothing, α2Away=nothing, αM1=nothing, αM2=nothing,
                                        threads=false, run_loocv=false, loocv_subject=nothing, full=false, nstarts=10)
    suffix = "blockwise_betas_twotd"
    if full
        suffix *= "_full"
    end
    if add_initial_V
        suffix *= "_initialV"
    end

    if add_TDλ1
        suffix *= "_TDlam1"
    elseif !isnothing(TDλ1)
        suffix *= "_TDlam1-$(TDλ1)"
    end
    if add_TDλ2
        suffix *= "_TDlam2"
    elseif !isnothing(TDλ2)
        suffix *= "_TDlam2-$(TDλ2)"
    end

    if add_βTD_1
        suffix *= "_BTD_1"
    elseif !isnothing(βTD_1)
        suffix *= "_BTD_1-$(βTD_1)"
    end
    if add_βTD_2
        suffix *= "_BTD_2"
    elseif !isnothing(βTD_2)
        suffix *= "_BTD_2-$(βTD_2)"
    end

    if add_βMB1
        suffix *= "_BMB1"
    elseif !isnothing(βMB1)
        suffix *= "_BMB1-$(βMB1)"
    end
    if add_βMB2
        suffix *= "_BMB2"
    elseif !isnothing(βMB2)
        suffix *= "_BMB2-$(βMB2)"
    end
    if add_βSR1
        suffix *= "_BSR1"
    elseif !isnothing(βSR1)
        suffix *= "_BSR1-$(βSR1)"
    end
    if add_βSR2
        suffix *= "_BSR2"
    elseif !isnothing(βSR2)
        suffix *= "_BSR2-$(βSR2)"
    end

    if add_βBoat
        suffix *= "_BBoat"
    elseif !isnothing(βBoat)
        suffix *= "_BBoat-$(βBoat)"
    end

    if add_α1
        suffix *= "_a1"
    elseif !isnothing(α1Home)
        suffix *= "_a1Home-$(α1Home)"
        suffix *= "_a1Away-$(α1Away)"
    end
    if add_α2
        suffix *= "_a2"
    elseif !isnothing(α2Home)
        suffix *= "_a2Home-$(α2Home)"
        suffix *= "_a2Away-$(α2Away)"
    end

    if add_αM1
        suffix *= "_aM1"
    elseif !isnothing(αM1)
        suffix *= "_aM1-$(αM1)"
    end
    if add_αM2
        suffix *= "_aM2"
    elseif !isnothing(αM2)
        suffix *= "_aM2-$(αM2)"
    end

    if add_decay
        suffix *= "_decay"
        if separate_decay
            suffix *= "-separate"
        end
    end
    if add_island_stay_bias
        suffix *= "_islandbias"
    end
    if add_boat_stay_bias
        suffix *= "_boatbias"
    end
    if separate_home_away
        suffix *= "_homeaway"
    end
    if rewscaled
        suffix *= "_rewscaled"
    end
    if decorrelateαβ
        suffix *= "_decorrelateAB"
    end
    if add_betas
        suffix *= "_addbetas"
    end
    if initx
        suffix *= "_initx-$(nstarts)"
    end

    fn = run_tdλ_mb_sr_blockwise_twotd

    if run_loocv
        @info "TD / MB /SR LOO-CV: $(suffix)"
        results = load("$(outdir)/$(file_prefix)/$(file_prefix)_$(suffix).jld2")["results"]
        if isnothing(loocv_subject)
            loocv_results = fn(data; extended=true, threads, rewscaled, groups, initx, decorrelateαβ, full, add_betas,
                add_TDλ1, add_TDλ2, add_βTD_1, add_βTD_2, add_βMB1, add_βMB2, add_βSR1, add_βSR2, add_βBoat, add_α1, add_α2, add_αM1, add_αM2, add_initial_V,
                add_decay, separate_decay, add_island_stay_bias, add_boat_stay_bias, separate_home_away,
                TDλ1, TDλ2, βTD_1, βTD_2, βMB1, βMB2, βSR1, βSR2, βBoat, α1Home, α1Away, α2Home, α2Away, αM1, αM2,
                nstarts,
                loocv_data=results,
                );
            save("$(outdir)/$(file_prefix)/$(file_prefix)_loocv_$(suffix).jld2", "loocv_results", loocv_results; compress=true)
        else
            @info "Subject: $(loocv_subject)"
            loocv_results = fn(data; extended=true, threads, rewscaled, groups, initx, decorrelateαβ, full, add_betas,
                add_TDλ1, add_TDλ2, add_βTD_1, add_βTD_2, add_βMB1, add_βMB2, add_βSR1, add_βSR2, add_βBoat, add_α1, add_α2, add_αM1, add_αM2, add_initial_V,
                add_decay, separate_decay, add_island_stay_bias, add_boat_stay_bias, separate_home_away,
                TDλ1, TDλ2, βTD_1, βTD_2, βMB1, βMB2, βSR1, βSR2, βBoat, α1Home, α1Away, α2Home, α2Away, αM1, αM2,
                nstarts,
                loocv_data=results,
                loocv_subject,
                );
            save("$(outdir)/$(file_prefix)/$(file_prefix)_loocv_s$(loocv_subject)_$(suffix).jld2", "loocv_results", loocv_results; compress=true)
        end
    else
        @info "TD / MB /SR: $(suffix)"
        results = fn(data; extended=true, threads, rewscaled, groups, initx, decorrelateαβ, full, add_betas,
            add_TDλ1, add_TDλ2, add_βTD_1, add_βTD_2, add_βMB1, add_βMB2, add_βSR1, add_βSR2, add_βBoat, add_α1, add_α2, add_αM1, add_αM2, add_initial_V,
            add_decay, separate_decay, add_island_stay_bias, add_boat_stay_bias, separate_home_away,
            TDλ1, TDλ2, βTD_1, βTD_2, βMB1, βMB2, βSR1, βSR2, βBoat, α1Home, α1Away, α2Home, α2Away, αM1, αM2,
            nstarts,
            );
        save("$(outdir)/$(file_prefix)/$(file_prefix)_$(suffix).jld2", "results", results; compress=true)
    end
end

transforms = Dict(
    "TDλ1" => x -> unitnorm(x),
    "TDλ2" => x -> unitnorm(x),
    "α1" => x -> unitnorm(x),
    "α2" => x -> unitnorm(x),
    "αM1" => x -> unitnorm(x),
    "αM2" => x -> unitnorm(x),
    "initial_V" => x -> unitnorm(x),
    "α1Home" => x -> unitnorm(x),
    "α1Away" => x -> unitnorm(x),
    "α2Home" => x -> unitnorm(x),
    "α2Away" => x -> unitnorm(x),
    "decay" => x -> unitnorm(x),
    "decay_island" => x -> unitnorm(x),
    "decay_boat" => x -> unitnorm(x),
)
