
using LinearAlgebra
using SpecialFunctions  # for erf
using StatsFuns
using DataFrames
using RLSR
using EM

include("sailing_base_lrl_shared.jl")

function lik_lrl_blockwise_twoλ(data, βLRL::U, βBoat, island_stay_bias, boat_stay_bias, α, αT, c, log_λ1, log_λ2, initial_V, rewscaled::Bool, record::Bool) where U
    ntrials = length(data)

    e_V_div_λ = zeros(U, 7)  # e^v/lambda
    # Below are all for temporary computation
    LRL_exp_r = zeros(U, 4)
    LRL_P_exp_r = zeros(U, 3)
    LRL_L = zeros(U, 3, 3)
    LRL_M = zeros(U, 3, 3)

    V_div_λ = zeros(U, 7)

    policy = zeros(U, 7, 7)  # Optimal Policy

    R = zeros(U, 7)

    # Default Policy
    T = zeros(U, 7, 7)
    T[1, 2:3] .= 0.5
    T[2, 4:5] .= 0.5
    T[3, 6:7] .= 0.5
    T[4, 4] = 1
    T[5, 5] = 1
    T[6, 6] = 1
    T[7, 7] = 1
    terminals = [false, false, false, true, true, true, true]
    nonterminals = [true, true, true, false, false, false, false]

    Q = zeros(U, 7)
    if (rewscaled)
        baseline_V = initial_V * 2.0 - 1.0 
    else
        baseline_V = initial_V
    end
    R[1:3] .= -c
    R[4:7] .= baseline_V

    Q .= baseline_V
    
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

    if record
        R_record = zeros(ntrials, 7)
        P_record = zeros(ntrials, 7, 7)
        e_V_record = zeros(ntrials, 7)
        V_record = zeros(ntrials, 7)
        Q_record = zeros(ntrials, 7)
        policy_record = zeros(ntrials, 7, 7)
        kl_record = zeros(ntrials, 7)
        cc_record = zeros(ntrials, 7)
    end 

    for i in eachindex(state1)
        # Reset if new session (trial number has decreased)
        if trial[i] < prev_trial
            T[1, 2:3] .= 0.5
            T[2, 4:5] .= 0.5
            T[3, 6:7] .= 0.5
            T[4, 4] = 1
            T[5, 5] = 1
            T[6, 6] = 1
            T[7, 7] = 1

            R[1:3] .= -c
            R[4:7] .= baseline_V

            Q .= baseline_V

            prev_state2 = 0
            prev_state_2_boat = 0
            prev_state_3_boat = 0
            prev_trial = 0
        end
      
        # Update LRL estimates
        # Swap options are "good", "bad", "within", "start"
        # "Good" is the only one where second-level policy remains consistent
        if (rwd_swap_type[i] == "good")
            λ = exp(log_λ1)
        else
            λ = exp(log_λ2)
        end
        core_lrl!(T, -R, λ, terminals, nonterminals, LRL_L, LRL_M, LRL_exp_r, LRL_P_exp_r, e_V_div_λ)
        V_div_λ .= log.(e_V_div_λ)
        policy .= diagm(1 ./ (T * e_V_div_λ)) * T * diagm(e_V_div_λ)

        if record
            R_record[i, :] .= R
            P_record[i, :, :] .= T
            e_V_record[i, :] .= e_V_div_λ
            V_record[i, :] .= V_div_λ .* λ

            policy_record[i, :, :] .= policy
            for s in 1:7
                successors = findall(T[s, :] .> 0)
                for sx in successors
                    kl_record[i, s] += policy[s, sx] * log(policy[s, sx] / T[s, sx])
                    cc_record[i, s] += λ * policy[s, sx] * log(policy[s, sx] / T[s, sx])
                end
            end
        end

        # Full traversals
        if (state1[i] == 1)

            # Q-value averaging
            Q[2:3] .= (βLRL * λ) .* view(V_div_λ, 2:3)
            # Island stay bias
            if (prev_state2 > 0)
                Q[prev_state2] += island_stay_bias
            end
            
            # Q[4:7] .= (βBoat .* view(policy, 2, 4:7))
            # Q[4:7] .= (βBoat * λ) .* view(V_div_λ, 4:7)
            Q[4:7] .= βBoat .* view(R, 4:7)  # Just query this directly

            # Likelihood for top choice
            lik += Q[state2[i]] - logsumexp(view(Q, 2:3))
            if state3[i] != -1  # If this wasn't a truncated trial
                # Second-level choices
                if state2[i] == 2
                    if (prev_state_2_boat > 0)
                        # Boat stay bias
                        Q[prev_state_2_boat] += boat_stay_bias
                    end
                    # Left island
                    lik += Q[state3[i]] - logsumexp(view(Q, 4:5))
                else
                    # Right island
                    if (prev_state_3_boat > 0)
                        # Boat stay bias
                        Q[prev_state_3_boat] += boat_stay_bias
                    end
                    lik += Q[state3[i]] - logsumexp(view(Q, 6:7))
                end

                # Temporal Difference Updates
                R[state3[i]] = (1 - α) * R[state3[i]] + α * reward[i]

                # Update P matrix, and re-normalize
                # 5 - 3 => 2
                # 5 - 2 => 3
                T[1, 2] *= (1 - αT)
                T[1, 3] *= (1 - αT)
                T[1, state2[i]] += αT

                # 4 +5 9: 4->5 and 5-> 4, 13: 6->7 and 7->6
                # 2*(2 or 3) + mod(x+1,2)
                # 5 - 2 => 3
                if state2[i] == 2
                    T[2, 4] *= (1 - αT)
                    T[2, 5] *= (1 - αT)
                    T[2, state3[i]] += αT
                else
                    T[3, 6] *= (1 - αT)
                    T[3, 7] *= (1 - αT)
                    T[3, state3[i]] += αT
                end

                prev_state2 = state2[i]
                prev_trial = trial[i]
                if state2[i] == 2
                    prev_state_2_boat = state3[i]
                else
                    prev_state_3_boat = state3[i]
                end
            end
        else  # Passive samples
            R[state1[i]] = (1 - α) * R[state1[i]] + α * reward[i]
        end
        if record
            Q_record[i, :] .= Q  # Record Q value that was actually used for this trial
        end
    end
    if record
        record_df = DataFrame()
        for i in 1:7
            record_df[!, Symbol("Q$i")] = Q_record[:, i]
            record_df[!, Symbol("R$i")] = R_record[:, i]
            record_df[!, Symbol("e_V$i")] = e_V_record[:, i]
            record_df[!, Symbol("V$i")] = V_record[:, i]
            record_df[!, Symbol("kl$i")] = kl_record[:, i]
            record_df[!, Symbol("cc$i")] = cc_record[:, i]
        end
        for (i, j) in transitions
            record_df[!, Symbol("policy_$(i)_$(j)")] = policy_record[:, i, j]
            record_df[!, Symbol("P_$(i)_$(j)")] = P_record[:, i, j]
        end
        return -lik, record_df
    else
        return -lik
    end
end

function lik_lrl_blockwise_twoλ(data;
    βLRL=0.0, βBoat=0.0,
    island_stay_bias=0.0, boat_stay_bias=0.0,
    α=0.0, αT=0.2,
    c=1.0, log_λ1=0.0, log_λ2=0.0,
    initial_V=0.5, rewscaled=true, record=false, verbose=true)
    if verbose
        println("βLRL:$(βLRL)")
        println("βBoat:$(βBoat)")
        println("island_stay_bias:$(island_stay_bias)")
        println("boat_stay_bias:$(boat_stay_bias)")
        println("α:$(α)")
        println("αT:$(αT)")
        println("c:$(c)")
        println("log_λ1:$(log_λ1)")
        println("log_λ2:$(log_λ2)")
        println("initial_V:$(initial_V)")
        println("$(minimum(data.reward))")
        println("$(maximum(data.reward))")
    end
    lik_lrl_blockwise_twoλ(data, βLRL, βBoat, island_stay_bias, boat_stay_bias, α, αT, c, log_λ1, log_λ2, initial_V, rewscaled, record)
end


function lik_lrl_blockwise_twoλ(data, results::T; subject=0, params=nothing, rewscaled=true, record=false, verbose=false) where T <: EMResultsAbstract
    d = Dict{Symbol, Any}()
    # The trick here is that we can pass in a dictionary of (symbol => value) as kwargs
    # Then everything not present the EMResults struct is left at its default value
    if subject == 0
        for i in eachindex(results.varnames)
            d[Symbol(results.varnames[i])] = results.betas[i]
        end
    else
        for i in eachindex(results.varnames)
            d[Symbol(results.varnames[i])] = results.x[subject, i]
        end
    end

    if !haskey(d, :βLRL)
        d[:βLRL] = 1.0
    end
    if !haskey(d, :βBoat)
        d[:βBoat] += d[:βLRL] 
    end
    if haskey(d, :initial_V)
        d[:initial_V] = unitnorm(d[:initial_V])
    end
    # Unpack learning rate
    if haskey(d, :α)
        d[:α] = unitnorm(d[:α])
    end
    # Transition learning rate
    if haskey(d, :αT)
        d[:αT] = unitnorm(d[:αT])
    end
    if haskey(d, :log_λ1) && !haskey(d, :log_λ2)
        d[:log_λ2] = d[:log_λ1]
    end
    if haskey(d, :c)
        d[:c] = unitnorm(d[:c])
    end
    # Incorporate any extra parameters
    if !isnothing(params)
        for (k, v) in params
            d[k] = v
        end
    end
    lik_lrl_blockwise_twoλ(data; verbose=verbose, rewscaled=rewscaled, record=record, d...)
end
function run_lrl_blockwise_twoλ(data; maxiter=200, emtol=1e-3, full=true, extended=false, quiet=false, threads=true, initx=false, noprior=false, nstarts=1,
    add_βLRL=false,    
    add_βBoat=false,
    add_island_stay_bias=false,
    add_boat_stay_bias=false,
    add_α=false,
    add_αT=false,
    add_λ1=false,
    add_λ2=false,
    add_c=false,
    add_initial_V=false,

    βLRL=1.0,
    βBoat=1.0,
    island_stay_bias=0.0,
    boat_stay_bias=0.0,
    α=0.0,
    αT=0.2,
    log_λ1=0.0,
    c=0.1,
    initial_V=0.5,
    use_βBoat=false,
    rewscaled=true,
    groups=nothing,

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

    if add_βLRL
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βLRL")
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
    if add_α
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "α")
    end
    if add_αT
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "αT")
    end
    if add_c
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "c")
    end
    if add_λ1
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "log_λ1")
    end
    if add_λ2
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "log_λ2")
    end
    if add_initial_V
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "initial_V")
    end

    if !isnothing(groups)
        initbetas = hcat(initbetas, zeros(length(initbetas)))
        X = hcat(X, groups)
    end
    initbetasT = Array(initbetas')

    function fn(params, data)

        i = 1

        if add_βLRL
            f_βLRL = params[i]
            i += 1
        else
            f_βLRL = βLRL
        end

        if add_βBoat
            f_βBoat = params[i]
            i += 1
        elseif use_βBoat
            f_βBoat = βBoat
        else
            f_βBoat = f_βLRL
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

        if add_α
            f_α = unitnorm(params[i])
            i += 1
        else
            f_α = α
        end

        if add_αT
            f_αT = unitnorm(params[i])
            i += 1
        else
            f_αT = αT
        end

        if add_c
            f_c = unitnorm(params[i])
            i += 1
        else
            f_c = c
        end

        if add_λ1
            f_log_λ1 = params[i]
            i += 1
        else
            f_log_λ1 = log_λ1
        end

        if add_λ2
            f_log_λ2 = params[i]
            i += 1
        else
            f_log_λ2 = f_log_λ1
        end

        if add_initial_V
            f_initial_V = unitnorm(params[i])
            i += 1
        else
            f_initial_V = initial_V
        end

        return lik_lrl_blockwise_twoλ(data, f_βLRL, f_βBoat, f_island_stay_bias, f_boat_stay_bias, f_α, f_αT, f_c, f_log_λ1, f_log_λ2, f_initial_V, rewscaled, false)
    end

    if !isnothing(loocv_data)
        if !isnothing(loocv_subject)
            return loocv_singlesubject(data,subs,loocv_subject,loocv_data.x,X,loocv_data.betas,loocv_data.sigma,fn; emtol, full, maxiter)
        else
            return loocv(data,subs,loocv_data.x,X,loocv_data.betas,loocv_data.sigma,fn; emtol, full, maxiter)
        end
    elseif noprior
        return emnoprior(data,subs,X,initbetasT,initsigma,fn;nstarts=nstarts)
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

function run_models_lrl_blockwise_twoλ(data, outdir, file_prefix, groups, add_initial_V, add_βLRL, add_βBoat, add_c, add_λ1, add_λ2, add_αT, add_island_stay_bias, add_boat_stay_bias, rewscaled, initx; log_λ1=0.0, αT=0.2, threads=false, full=true, noprior=false, run_loocv=false, loocv_subject=nothing)
    suffix = "blockwise_twoλ"
    if full
        suffix *= "_full"
    end
    if noprior
        suffix *= "_noprior"
    end
    if add_initial_V
        suffix *= "_initialV"
    end
    if add_βLRL
        suffix *= "_BLRL"
    end
    if add_βBoat
        suffix *= "_BBoat"
        use_βBoat = true
    else
        use_βBoat = false
    end
    if add_c
        suffix *= "_c"
    end
    if add_λ1
        suffix *= "_λ1"
    else
        suffix *= "_logλ1-$(log_λ1)"
    end
    if add_λ2
        suffix *= "_λ2"
    end
    if add_αT
        suffix *= "_aT"
    else
        suffix *= "_aT$(αT)"
    end
    if add_island_stay_bias
        suffix *= "_islandbias"
    end
    if add_boat_stay_bias
        suffix *= "_boatbias"
    end
    if rewscaled
        suffix *= "_rewscaled"
    end
    if initx
        suffix *= "_initx"
    end

    fn = run_lrl_blockwise_twoλ

    if run_loocv
        @info "LRL LOO-CV: $(suffix)"
        results = load("$(outdir)/$(file_prefix)/$(file_prefix)_lrl$(suffix).jld2")["$(file_prefix)_lrl$(suffix)"]
        if isnothing(loocv_subject)
            loocv_results = fn(data; extended=true, full=full, noprior=noprior, threads=threads, rewscaled=rewscaled, groups=groups, initx=initx,
                add_βLRL=add_βLRL, use_βBoat=use_βBoat, add_α=true, add_αT=add_αT, αT=αT, add_initial_V=add_initial_V, add_c=add_c, log_λ1=log_λ1, add_λ1=add_λ1, add_λ2=add_λ2,
                add_βBoat=add_βBoat, add_island_stay_bias=add_island_stay_bias, add_boat_stay_bias=add_boat_stay_bias,
                loocv_data=results,
                );
            save("$(outdir)/$(file_prefix)/$(file_prefix)_lrl_loocv$(suffix).jld2", "$(file_prefix)_lrl_loocv$(suffix)", loocv_results; compress=true)
        else
            @info "LRL LOO-CV Subject: $(loocv_subject)"
            loocv_results = fn(data; extended=true, full=full, noprior=noprior, threads=threads, rewscaled=rewscaled, groups=groups, initx=initx,
                add_βLRL=add_βLRL, use_βBoat=use_βBoat, add_α=true, add_αT=add_αT, αT=αT, add_initial_V=add_initial_V, add_c=add_c, log_λ1=log_λ1, add_λ1=add_λ1, add_λ2=add_λ2,
                add_βBoat=add_βBoat, add_island_stay_bias=add_island_stay_bias, add_boat_stay_bias=add_boat_stay_bias,
                loocv_data=results, loocv_subject=loocv_subject,
                );
            save("$(outdir)/$(file_prefix)/$(file_prefix)_lrl_loocv_s$(loocv_subject)$(suffix).jld2", "$(file_prefix)_lrl_loocv_s$(loocv_subject)$(suffix)", loocv_results; compress=true)
        end
    else
        @info "LRL: $(suffix)"
        results = fn(data; extended=true, full=full, noprior=noprior, threads=threads, rewscaled=rewscaled, groups=groups, initx=initx,
            add_βLRL=add_βLRL, use_βBoat=use_βBoat, add_α=true, add_αT=add_αT, αT=αT, add_initial_V=add_initial_V, add_c=add_c, log_λ1=log_λ1, add_λ1=add_λ1, add_λ2=add_λ2,
            add_βBoat=add_βBoat, add_island_stay_bias=add_island_stay_bias, add_boat_stay_bias=add_boat_stay_bias,
            );
        save("$(outdir)/$(file_prefix)/$(file_prefix)_lrl$(suffix).jld2", "$(file_prefix)_lrl$(suffix)", results; compress=true)
    end
end

transforms_lrl = Dict(
    "α" => x -> unitnorm(x),
    "αT" => x -> unitnorm(x),
    "initial_V" => x -> unitnorm(x),
    "λ1" => x -> exp(x),
    "λ2" => x -> exp(x),
    "log_λ1" => x -> exp(x),
    "log_λ2" => x -> exp(x),
    "c" => x -> unitnorm(x),
)
