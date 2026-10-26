# Subproblem solver based on a solver from RegularizedOptimization.jl for
#
#   minimize f(x) + h_μ(F(x) + μy) + regularizer(x),
#
# where the smooth part is the objective of an EnveloptNLPModel and
# the regularizer is handled through its proximal operator.
mutable struct RegOptEnveloptSubSolver{
  E <: EnveloptNLPModel,
  R <: AbstractRegularizedNLPModel,
  SOLVER <: AbstractOptimizationSolver,
  STATS <: GenericExecutionStats,
  O <: NamedTuple,
} <: Envelopt.AbstractEnveloptSubSolver
  env_model::E    # the model passed to envelopt
  reg_nlp::R      # smooth part (a quasi-Newton model of env_model or env_model itself) + regularizer
  solver::SOLVER  # e.g., R2NSolver or R2Solver
  stats::STATS
  options::O      # options passed to the solver at each solve
  name::String
end

const regopt_fixed_options = (max_iter = 10000, max_time = 600.0)

function check_unconstrained(env_model::EnveloptNLPModel, name::String)
  get_ncon(env_model) == 0 ||
    error("$name only supports problems without constraints c(x) ∈ C; include them in F")
end

# Allocate once: envelopt updates μ and y in env_model in place,
# so reg_nlp and the solver workspace remain valid across outer iterations.
function RegOptEnveloptSubSolver(
  env_model::EnveloptNLPModel,
  smooth_model::AbstractNLPModel,
  regularizer,
  selected,
  make_solver,
  name::String,
  options::NamedTuple,
)
  reg_nlp = RegularizedNLPModel(smooth_model, regularizer, selected)
  solver = make_solver(reg_nlp)
  stats = RegularizedExecutionStats(reg_nlp)
  return RegOptEnveloptSubSolver(
    env_model,
    reg_nlp,
    solver,
    stats,
    merge(regopt_fixed_options, options),
    name,
  )
end

# ... solve
function (sub::RegOptEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  args...;
  tol::Real = 1.0e-6,
  kwargs...,
)
  env_model === sub.env_model ||
    error("$(sub.name)EnveloptSubSolver was constructed for a different EnveloptNLPModel")
  T = eltype(x0)
  SolverCore.solve!(
    sub.solver,
    sub.reg_nlp,
    sub.stats;
    x = x0,
    atol = T(tol),
    rtol = zero(T),
    sub.options...,
    kwargs...,
  )
  return sub.stats
end

Envelopt.regularizer_value(sub::RegOptEnveloptSubSolver, x::AbstractVector) =
  @views sub.reg_nlp.h(x[sub.reg_nlp.selected])
