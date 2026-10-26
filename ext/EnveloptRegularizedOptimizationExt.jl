module EnveloptRegularizedOptimizationExt

using Envelopt, NLPModels, NLPModelsModifiers, SolverCore
using RegularizedOptimization, RegularizedProblems

include("RegularizedOptimization/r2.jl")
include("RegularizedOptimization/r2n.jl")

# The problem minimize f(x) + h(F(x)) + regularizer(x),
# where reg_nlp.model is the EnveloptNLPModel of f(x) + h(F(x)) and reg_nlp.h is the regularizer.
const EnveloptRegularizedNLPModel{T, V} = RegularizedNLPModel{T, V, <:EnveloptNLPModel{T, V}}

const RegOptEnveloptSubSolver = Union{R2EnveloptSubSolver, R2NEnveloptSubSolver}

# ... subproblem solvers
Envelopt.R2EnveloptSubSolver(reg_nlp::EnveloptRegularizedNLPModel) =
  Envelopt.R2EnveloptSubSolver(reg_nlp.model, reg_nlp.h; selected = reg_nlp.selected)

Envelopt.R2NEnveloptSubSolver(reg_nlp::EnveloptRegularizedNLPModel; kwargs...) =
  Envelopt.R2NEnveloptSubSolver(reg_nlp.model, reg_nlp.h; selected = reg_nlp.selected, kwargs...)

# ... workspace and stats
function Envelopt.EnveloptSolver(
  reg_nlp::EnveloptRegularizedNLPModel;
  subsolver::RegOptEnveloptSubSolver = Envelopt.R2EnveloptSubSolver(reg_nlp),
)
  return EnveloptSolver(reg_nlp.model; subsolver = subsolver)
end

Envelopt.EnveloptExecutionStats(reg_nlp::EnveloptRegularizedNLPModel) =
  EnveloptExecutionStats(reg_nlp.model)

# ... solve
function Envelopt.envelopt(
  reg_nlp::EnveloptRegularizedNLPModel;
  subsolver::RegOptEnveloptSubSolver = Envelopt.R2EnveloptSubSolver(reg_nlp),
  kwargs...,
)
  solver = EnveloptSolver(reg_nlp; subsolver = subsolver)
  return solve!(solver, reg_nlp; kwargs...)
end

function SolverCore.solve!(solver::EnveloptSolver, reg_nlp::EnveloptRegularizedNLPModel; kwargs...)
  stats = EnveloptExecutionStats(reg_nlp)
  return solve!(solver, reg_nlp, stats; kwargs...)
end

function SolverCore.solve!(
  solver::EnveloptSolver,
  reg_nlp::EnveloptRegularizedNLPModel,
  stats::EnveloptExecutionStats;
  kwargs...,
)
  # a smooth subproblem solver would silently ignore the regularizer
  solver.subsolver isa RegOptEnveloptSubSolver || throw(
    ArgumentError(
      "the subproblem solver must be R2NEnveloptSubSolver or R2EnveloptSubSolver to handle the regularizer",
    ),
  )
  return solve!(solver, reg_nlp.model, stats; kwargs...)
end

end
