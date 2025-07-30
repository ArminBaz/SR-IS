module EM

using DataFrames
using ForwardDiff
using Optim
using LinearAlgebra       # for tr, diagonal
using StatsFuns           # for logsumexp
using SpecialFunctions    # for erf
using Statistics          # for mean
using Distributions		  # for tDist

export em,emerrors,eminits,lml,ibic,iaic,loocv,loocv_singlesubject,qlik,jianlik,seqlik,simq,simseq,simjian
export EMResultsAbstract,EMResults,EMResultsExtended
export emnoprior


include("emcore.jl")
include("emutils.jl")
include("emlikfuns.jl")

# uncomment this (and code in emutils.jl) to use python optimizer
#using PyCall
#const so = PyNULL()
#function __init__()
#	copy!(so,pyimport("scipy.optimize"))
#end

end
