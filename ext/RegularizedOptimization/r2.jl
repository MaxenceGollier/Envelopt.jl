# R2 subproblem solver
mutable struct R2EnveloptSubSolver{
  SOLVER <: R2Solver,
  STATS <: GenericExecutionStats,
  REG_NLP <: AbstractRegularizedNLPModel,
} <: Envelopt.AbstractEnveloptSubSolver
  solver::SOLVER
  stats::STATS
  name::String
  reg_nlp::REG_NLP  # env_model + regularizer
end

# ... constructor
function Envelopt.R2EnveloptSubSolver(
  env_model::EnveloptNLPModel,
  regularizer;
  selected = 1:get_nvar(env_model),
)
  @debug "initializing R2 subproblem solver"
  get_ncon(env_model) == 0 || error("R2 only supports unconstrained and bound-constrained problems")
  reg_nlp = RegularizedNLPModel(env_model, regularizer, selected)
  solver = R2Solver(reg_nlp)
  stats = RegularizedExecutionStats(reg_nlp)
  return R2EnveloptSubSolver(solver, stats, "R2", reg_nlp)
end

const r2_fixed_options = Dict(:max_iter => 10000, :max_time => 600.0)

# ... solve
function (M::R2EnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  args...;
  tol::Float64 = 1.0e-6,
  kwargs...,
)
  SolverCore.solve!(
    M.solver,
    M.reg_nlp,
    M.stats;
    atol = tol,
    rtol = 0.0,
    x = x0,
    r2_fixed_options...,
    kwargs...,
  )
  return M.stats
end
