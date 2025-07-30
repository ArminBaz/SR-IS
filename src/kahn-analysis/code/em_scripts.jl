"""
Pretty-print group-level means and SDs.

results: EMResults
mu: bool, whether to show means
sigma: bool, whether to show SDs
transforms: optionally, a dictionary of columns names to functions
    Intended for variables which are transformed prior to use in the likelihood
    For column x, x* is added applying func
"""
function show_results(results; mu=true, sigma=true, transforms=nothing)
    io = IOBuffer()
    
    varnames = results.varnames
    betas = results.betas
    for i in axes(betas, 2)
        for j in 2:length(betas[:,i])
            betas[j,i] = betas[1,i] + betas[j,i]
        end
    end
    betas_t = copy(results.betas)
    if !isnothing(transforms)
        for i in axes(betas, 2)
            v = varnames[i]
            if haskey(transforms, v)
                fn = transforms[v]
                betas_t[:, i] .= fn.(betas[:, i])
                # new_beta = zeros(size(betas, 1))
                # new_beta[1] = fn(betas[1,i])
                # for j in 2:length(new_beta)
                #     # new_beta[j] = fn(betas[1,i] + betas[j,i]) - new_beta[1]
                #     new_beta[j] = fn(betas[1,i] + betas[j,i])
                # end
                # betas_t[:, i] .= new_beta
            end
        end
    end

    # Want abbreviated variable names. Since the the leading minus sign will effectively
    # pad negative numbers, we include on more character on those, so everything lines up
    varnames_short = copy(varnames)
    for i in eachindex(varnames_short)
        if any(betas[:, i] .< 0)
            if length(eachindex(varnames_short[i])) > 6
                varnames_short[i] = varnames_short[i][1:nextind(varnames_short[i],0,6)]
            end
            varnames_short[i] *= " "^(7 - length(eachindex(varnames_short[i])))
        else 
            if length(eachindex(varnames_short[i])) > 5
                varnames_short[i] = varnames_short[i][1:nextind(varnames_short[i],0,5)]
            end
            varnames_short[i] *= " "^(6 - length(eachindex(varnames_short[i])))
        end
    end
    varnames_short_t = copy(varnames)
    for i in eachindex(varnames_short_t)
        if any(betas_t[:, i] .< 0)
            if length(eachindex(varnames_short_t[i])) > 6
                varnames_short_t[i] = varnames_short_t[i][1:nextind(varnames_short[i],0,6)]
            end
            varnames_short_t[i] *= " "^(7 - length(eachindex(varnames_short_t[i])))
        else 
            if length(eachindex(varnames_short_t[i])) > 5
                varnames_short_t[i] = varnames_short_t[i][1:nextind(varnames_short[i],0,5)]
            end
            varnames_short_t[i] *= " "^(6 - length(eachindex(varnames_short_t[i])))
        end
    end
    
    print(io, "<pre>")
    print(io, "$(varnames)\n")
    print(io, "</pre>")

    if mu
        print(io, "β:<br/>")
        print(io, "<pre>")
        print(io, "$(" " * join(varnames_short))\n")
        show(io, MIME"text/plain"(), round.(betas; digits=2))        
        if !isnothing(transforms)
            print(io, "<br/>β*:<br/>")
            print(io, "$(" " * join(varnames_short_t))\n")
            show(io, MIME"text/plain"(), round.(betas_t; digits=2))        
        end
        if results isa EMResultsExtended
            pvalues = reshape(results.pvalues,size(results.betas'))'
            print(io, "<br/>p:<br/>")
            show(io, MIME"text/plain"(), round.(pvalues, RoundUp; digits=2))
        end
        print(io, "</pre>")
    end
    
    if sigma
        print(io, "σ²:<br/>")
        print(io, "<pre>")
        show(io, MIME"text/plain"(), round.(results.sigma; digits=2))
        print(io, "</pre>")
    end
    
    my_string = String(take!(io))
    my_string = replace(my_string, r"[0-9]+×[0-9]+ Matrix{Float64}:\n" => "")
    my_string = replace(my_string, r"(-[0-9\.]+)" => s"<span style='color: red'>\1</span>")
    my_string = replace(my_string, "\n" => "<br/>")
    
    my_string
end

function group_cov(results)
    io = IOBuffer()
    
    x = repeat(sqrt.(diag(results.sigma)), outer=[1, size(results.sigma)[1]])
    cov_scaled = results.sigma ./ x ./ x'

    varnames_short = copy(results.varnames)
    for i in eachindex(varnames_short)
        if any(cov_scaled[:, i] .< 0)
            if length(eachindex(varnames_short[i])) > 6
                varnames_short[i] = varnames_short[i][1:nextind(varnames_short[i],0,6)]
            end
            varnames_short[i] *= " "^(7 - length(eachindex(varnames_short[i])))
        else 
            if length(eachindex(varnames_short[i])) > 5
                varnames_short[i] = varnames_short[i][1:nextind(varnames_short[i],0,5)]
            end
            varnames_short[i] *= " "^(6 - length(eachindex(varnames_short[i])))
        end
    end
    
    print(io, "Scaled Group-Level Covariances:<br/>")
    print(io, "<pre>")
    print(io, "$(" " * join(varnames_short))\n")
    show(io, MIME"text/plain"(), round.(cov_scaled; digits=2))
    print(io, "</pre>")
    
    my_string = String(take!(io))
    my_string = replace(my_string, r"[0-9]+×[0-9]+ Matrix{Float64}:\n" => "")
    my_string = replace(my_string, r"(-[0-9\.]+)" => s"<span style='color: red'>\1</span>")
    my_string = replace(my_string, "\n" => "<br/>")
    
    my_string
end

function scale_cov(results)
    x = repeat(sqrt.(diag(results.sigma)), outer=[1, size(results.sigma)[1]])
    results.sigma ./ x ./ x'
end

function scale_subject_cov(results)
    avg_h = dropdims(mean(results.h,dims=3),dims=3)
    x = repeat(sqrt.(diag(avg_h)), outer=[1, size(avg_h)[1]])
    avg_h ./ x ./ x'
end

function subject_cov(results; scale=true)
    io = IOBuffer()
    
    if scale
        avg_h = dropdims(mean(results.h,dims=3),dims=3)
        x = repeat(sqrt.(diag(avg_h)), outer=[1, size(avg_h)[1]])
        cov_scaled = avg_h ./ x ./ x'

        varnames_short = copy(results.varnames)
        for i in eachindex(varnames_short)
            if any(cov_scaled[:, i] .< 0)
                if length(eachindex(varnames_short[i])) > 6
                    varnames_short[i] = varnames_short[i][1:nextind(varnames_short[i],0,6)]
                end
                varnames_short[i] *= " "^(7 - length(eachindex(varnames_short[i])))
            else 
                if length(eachindex(varnames_short[i])) > 5
                    varnames_short[i] = varnames_short[i][1:nextind(varnames_short[i],0,5)]
                end
                varnames_short[i] *= " "^(6 - length(eachindex(varnames_short[i])))
            end
        end

        print(io, "Scaled Average Subject Covariances:<br/>")
        print(io, "<pre>")
        print(io, "$(" " * join(varnames_short))\n")
        show(io, MIME"text/plain"(), round.(cov_scaled; digits=2))
        print(io, "</pre>")
    else
        avg_h = dropdims(mean(results.h,dims=3),dims=3)

        varnames_short = copy(results.varnames)
        for i in eachindex(varnames_short)
            if any(avg_h[:, i] .< 0)
                if length(eachindex(varnames_short[i])) > 6
                    varnames_short[i] = varnames_short[i][1:nextind(varnames_short[i],0,6)]
                end
                varnames_short[i] *= " "^(7 - length(eachindex(varnames_short[i])))
            else 
                if length(eachindex(varnames_short[i])) > 5
                    varnames_short[i] = varnames_short[i][1:nextind(varnames_short[i],0,5)]
                end
                varnames_short[i] *= " "^(6 - length(eachindex(varnames_short[i])))
            end
        end

        print(io, "Average Subject Covariances:<br/>")
        print(io, "<pre>")
        print(io, "$(" " * join(varnames_short))\n")
        show(io, MIME"text/plain"(), round.(avg_h; digits=2))
        print(io, "</pre>")
    end
    
    my_string = String(take!(io))
    my_string = replace(my_string, r"[0-9]+×[0-9]+ Matrix{Float64}:\n" => "")
    my_string = replace(my_string, r"(-[0-9\.]+)" => s"<span style='color: red'>\1</span>")
    my_string = replace(my_string, "\n" => "<br/>")
    
    my_string
end