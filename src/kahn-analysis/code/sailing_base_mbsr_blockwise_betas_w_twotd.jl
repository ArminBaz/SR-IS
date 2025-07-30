include("sailing_base_mbsr_blockwise_betas_twotd.jl")

"""
Rewrite of the TD / MBSR model to fit MBSR as a single β and a weight
"""
function run_tdλ_mb_sr_blockwise_w_twotd(data; maxiter=200, emtol=1e-3, full=true, extended=false, quiet=false, threads=true, initx=false, nstarts=1,
    add_TDλ1=false,
    add_TDλ2=false,
    add_βTD_1=false,    
    add_βTD_2=false,    
    add_wSR1=false,
    add_wSR2=false,
    add_βMBSR_1=false,
    add_βMBSR_2=false,
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
    wSR1=nothing,
    wSR2=nothing,
    βMBSR_1=nothing,
    βMBSR_2=nothing,
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
    if add_wSR1
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "wSR1")
    end
    if add_wSR2
        push!(initbetas, 0)
        push!(initsigma, 1)
        push!(varnames, "wSR2")
    end
    if add_βMBSR_1
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βMBSR_1")
    end
    if add_βMBSR_2
        push!(initbetas, 1)
        push!(initsigma, 5)
        push!(varnames, "βMBSR_2")
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
    
        if add_wSR1
            f_wSR1 = unitnorm(params[i])
            i += 1
        else
            f_wSR1 = wSR1
        end

        if add_wSR2
            f_wSR2 = unitnorm(params[i])
            i += 1
        elseif !isnothing(wSR2)
            f_wSR2 = wSR2
        else
            f_wSR2 = f_wSR1
        end

        if add_βMBSR_1
            f_βMBSR_1 = params[i]
            f_βMB1 = f_βMBSR_1 * (1.0 - f_wSR1)
            f_βSR1 = f_βMBSR_1 * f_wSR1
            i += 1
        elseif !isnothing(βMBSR_1)
            f_βMBSR_1 = βMBSR_1
            f_βMB1 = f_βMBSR_1 * (1.0 - f_wSR1)
            f_βSR1 = f_βMBSR_1 * f_wSR1
        else
            f_βMB1 = 0.0
            f_βSR1 = 0.0
        end

        if add_βMBSR_2
            f_βMB2 = params[i] * (1.0 - f_wSR2)
            f_βSR2 = params[i] * f_wSR2
            i += 1
        elseif !isnothing(βMBSR_2)
            f_βMB2 = βMBSR_2 * (1.0 - f_wSR2)
            f_βSR2 = βMBSR_2 * f_wSR2
        elseif add_βMBSR_1 | !isnothing(βMBSR_1)
            f_βMB2 = f_βMBSR_1 * (1.0 - f_wSR2)
            f_βSR2 = f_βMBSR_1 * f_wSR2
        else
            f_βMB2 = f_βMB1
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

function run_models_tdλ_mb_sr_blockwise_w_twotd(data, outdir, file_prefix, groups,
                                        add_initial_V, add_TDλ1, add_TDλ2, add_βTD_1, add_βTD_2, add_wSR1, add_wSR2, add_βMBSR_1, add_βMBSR_2, add_βBoat, add_α1, add_α2, add_αM1, add_αM2,
                                        add_decay, separate_decay, add_island_stay_bias, add_boat_stay_bias, separate_home_away, rewscaled, decorrelateαβ, add_betas, initx;
                                        TDλ1=nothing, TDλ2=nothing, βTD_1=nothing, βTD_2=nothing, wSR1=nothing, wSR2=nothing, βMBSR_1=nothing, βMBSR_2=nothing, βBoat=nothing, α1Home=nothing, α1Away=nothing, α2Home=nothing, α2Away=nothing, αM1=nothing, αM2=nothing,
                                        threads=false, run_loocv=false, loocv_subject=nothing, full=false, nstarts=10)
    suffix = "blockwise_betas_w_twotd"
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

    if add_wSR1
        suffix *= "_wSR1"
    elseif !isnothing(wSR1)
        suffix *= "_wSR1-$(wSR1)"
    end
    if add_wSR2
        suffix *= "_wSR2"
    elseif !isnothing(wSR2)
        suffix *= "_wSR2-$(wSR2)"
    end

    if add_βMBSR_1
        suffix *= "_BMBSR_1"
    elseif !isnothing(βMBSR_1)
        suffix *= "_BMBSR_1-$(βMBSR_1)"
    end
    if add_βMBSR_2
        suffix *= "_BMBSR_2"
    elseif !isnothing(βMBSR_2)
        suffix *= "_BMBSR_2-$(βMBSR_2)"
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

    fn = run_tdλ_mb_sr_blockwise_w_twotd

    if run_loocv
        @info "TD / MB /SR LOO-CV: $(suffix)"
        results = load("$(outdir)/$(file_prefix)/$(file_prefix)_$(suffix).jld2")["results"]
        if isnothing(loocv_subject)
            loocv_results = fn(data; extended=true, threads, rewscaled, groups, initx, decorrelateαβ, full, add_betas,
                add_TDλ1, add_TDλ2, add_βTD_1, add_βTD_2, add_wSR1, add_wSR2, add_βMBSR_1, add_βMBSR_2, add_βBoat, add_α1, add_α2, add_αM1, add_αM2, add_initial_V,
                add_decay, separate_decay, add_island_stay_bias, add_boat_stay_bias, separate_home_away,
                TDλ1, TDλ2, βTD_1, βTD_2, wSR1, wSR2, βMBSR_1, βMBSR_2, βBoat, α1Home, α1Away, α2Home, α2Away, αM1, αM2,
                nstarts,
                loocv_data=results,
                );
            save("$(outdir)/$(file_prefix)/$(file_prefix)_loocv_$(suffix).jld2", "loocv_results", loocv_results; compress=true)
        else
            @info "Subject: $(loocv_subject)"
            loocv_results = fn(data; extended=true, threads, rewscaled, groups, initx, decorrelateαβ, full, add_betas,
                add_TDλ1, add_TDλ2, add_βTD_1, add_βTD_2, add_wSR1, add_wSR2, add_βMBSR_1, add_βMBSR_2, add_βBoat, add_α1, add_α2, add_αM1, add_αM2, add_initial_V,
                add_decay, separate_decay, add_island_stay_bias, add_boat_stay_bias, separate_home_away,
                TDλ1, TDλ2, βTD_1, βTD_2, wSR1, wSR2, βMBSR_1, βMBSR_2, βBoat, α1Home, α1Away, α2Home, α2Away, αM1, αM2,
                nstarts,
                loocv_data=results,
                loocv_subject,
                );
            save("$(outdir)/$(file_prefix)/$(file_prefix)_loocv_s$(loocv_subject)_$(suffix).jld2", "loocv_results", loocv_results; compress=true)
        end
    else
        @info "TD / MB /SR: $(suffix)"
        results = fn(data; extended=true, threads, rewscaled, groups, initx, decorrelateαβ, full, add_betas,
            add_TDλ1, add_TDλ2, add_βTD_1, add_βTD_2, add_wSR1, add_wSR2, add_βMBSR_1, add_βMBSR_2, add_βBoat, add_α1, add_α2, add_αM1, add_αM2, add_initial_V,
            add_decay, separate_decay, add_island_stay_bias, add_boat_stay_bias, separate_home_away,
            TDλ1, TDλ2, βTD_1, βTD_2, wSR1, wSR2, βMBSR_1, βMBSR_2, βBoat, α1Home, α1Away, α2Home, α2Away, αM1, αM2,
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
    "wSR1" => x -> unitnorm(x),
    "wSR2" => x -> unitnorm(x),
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
