# R2N subproblem solver
mutable struct R2NEnveloptSubSolver{
  SOLVER <: R2NSolver,
  STATS <: GenericExecutionStats,
  REG_NLP <: AbstractRegularizedNLPModel,
} <: Envelopt.AbstractEnveloptSubSolver
  solver::SOLVER
  stats::STATS
  name::String
  reg_nlp::REG_NLP  # quasi-Newton model of env_model + regularizer
end

# ... constructor
function Envelopt.R2NEnveloptSubSolver(
  env_model::EnveloptNLPModel,
  regularizer;
  hessian_approximation::Symbol = :lbfgs,
  mem::Int = 5,
  subsolver = R2Solver,
  m_monotone::Int = 1,
  selected = 1:get_nvar(env_model),
)
  @debug "initializing R2N subproblem solver"
  get_ncon(env_model) == 0 ||
    error("R2N only supports unconstrained and bound-constrained problems")
  # quasi-Newton approximation of the Hessian of f(x) + h_μ(F(x) + μy)
  if hessian_approximation == :lbfgs
    qn_model = LBFGSModel(env_model; mem = mem)
  elseif hessian_approximation == :lsr1
    qn_model = LSR1Model(env_model; mem = mem)
  else
    error("unknown Hessian approximation :$hessian_approximation; choose :lbfgs or :lsr1")
  end
  reg_nlp = RegularizedNLPModel(qn_model, regularizer, selected)
  # The solver is tied to reg_nlp because it stores its Hessian operator.
  solver = R2NSolver(reg_nlp; subsolver = subsolver, m_monotone = m_monotone)
  stats = RegularizedExecutionStats(reg_nlp)
  return R2NEnveloptSubSolver(solver, stats, "R2N", reg_nlp)
end

const r2n_fixed_options = Dict(:max_iter => 10000, :max_time => 600.0)

# ... solve
function (M::R2NEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  args...;
  tol::Float64 = 1.0e-6,
  kwargs...,
)
  # μ and y have changed since the last subproblem, so previous quasi-Newton pairs are outdated.
  SolverCore.reset!(M.solver)
  SolverCore.solve!(
    M.solver,
    M.reg_nlp,
    M.stats;
    atol = tol,
    rtol = 0.0,
    x = x0,
    r2n_fixed_options...,
    kwargs...,
  )
  return M.stats
end
