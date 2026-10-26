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
include("envelopt_stats.jl")
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

export R2NEnveloptSubSolver, R2EnveloptSubSolver

"""
    R2NEnveloptSubSolver(env_model::EnveloptNLPModel, regularizer; kwargs...)

An Envelopt subproblem solver based on R2N from RegularizedOptimization.jl.
Together with [`envelopt`](@ref), it solves

    minimize f(x) + h(F(x)) + regularizer(x),

where `regularizer` is a proper, lower semi-continuous and prox-bounded function,
possibly nonconvex, supported by ShiftedProximalOperators.jl,
and `h` is the convex function of `env_model`.
Each subproblem

    minimize f(x) + h_μ(F(x) + μy) + regularizer(x)

is solved by R2N, which treats `regularizer` through its proximal operator.
`env_model` must not have constraints c(x) ∈ C; all constraints must be part of F.

This solver is defined in an extension: load RegularizedOptimization.jl and
RegularizedProblems.jl to use it.

## Keyword arguments

* `hessian_approximation::Symbol = :lbfgs`: the quasi-Newton approximation of the Hessian
  of the whole smooth objective used by R2N, either `:lbfgs` or `:lsr1`;
* `mem::Int = 5`: memory of the quasi-Newton approximation;
* `subsolver = R2Solver`, `m_monotone::Int = 1`: passed to `R2NSolver`;
* `selected = 1:nvar`: the variables to which `regularizer` applies.

Any other keyword argument is passed to R2N at each subproblem solve,
e.g., `max_iter`, `verbose` or `σmin`.
"""
function R2NEnveloptSubSolver(args...; kwargs...)
  error(
    "R2N support is not loaded. Install and load RegularizedOptimization.jl and RegularizedProblems.jl to use R2NEnveloptSubSolver.",
  )
end

"""
    R2EnveloptSubSolver(env_model::EnveloptNLPModel, regularizer; kwargs...)

An Envelopt subproblem solver based on R2 from RegularizedOptimization.jl,
a first-order quadratic regularization method that only uses the objective and gradient
of `env_model`.
It solves the same problems and subproblems as [`R2NEnveloptSubSolver`](@ref),
under the same assumptions.

This solver is defined in an extension: load RegularizedOptimization.jl and
RegularizedProblems.jl to use it.

## Keyword arguments

* `selected = 1:nvar`: the variables to which `regularizer` applies.

Any other keyword argument is passed to R2 at each subproblem solve,
e.g., `max_iter`, `verbose` or `σmin`.
"""
function R2EnveloptSubSolver(args...; kwargs...)
  error(
    "R2 support is not loaded. Install and load RegularizedOptimization.jl and RegularizedProblems.jl to use R2EnveloptSubSolver.",
  )
end

end
