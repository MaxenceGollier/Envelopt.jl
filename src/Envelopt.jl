module Envelopt

using LinearAlgebra
using Logging
using Printf

using ProximalCore: prox!

using LinearOperators
using NCL
using NLPModels
using NLPModelsModifiers
using SolverCore

include("moreau_envelope.jl")
include("F_model.jl")
include("envelopt_model.jl")
include("envelopt_solver.jl")

include("eq_feasibility.jl")
include("remove_eqs.jl")

# Extensions
export KnitroEnveloptSubSolver

function KnitroEnveloptSubSolver(args...; kwargs...)
  error(
    "Knitro support is not loaded. Install and load KNITRO.jl and NLPModelsKnitro.jl to use KnitroEnveloptSubSolver.",
  )
end

export IPOPTEnveloptSubSolver

function IPOPTEnveloptSubSolver(args...; kwargs...)
  error(
    "IPOPT support is not loaded. Install and load NLPModelsIpopt.jl to use IPOPTEnveloptSubSolver.",
  )
end

export exact_penalty_solver

function exact_penalty_solver(args...; kwargs...)
  error(
    "Exact penalty support is not loaded. Install and load ProximalOperators.jl to use exact_penalty_solver.",
  )
end

export TronEnveloptSubSolver, TrunkEnveloptSubSolver

function TronEnveloptSubSolver(args...; kwargs...)
  error("TRON support is not loaded. Install and load JSOSolvers.jl to use TronEnveloptSubSolver.")
end

function TrunkEnveloptSubSolver(args...; kwargs...)
  error(
    "TRUNK support is not loaded. Install and load JSOSolvers.jl to use TrunkEnveloptSubSolver.",
  )
end

end
