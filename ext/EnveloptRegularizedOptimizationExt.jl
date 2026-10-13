module EnveloptRegularizedOptimizationExt

using Envelopt
using NLPModels
using NLPModelsModifiers
using RegularizedOptimization
using RegularizedProblems
using SolverCore

# R2N subproblem solver for
#
#   minimize f(x) + h_μ(F(x) + μy) + regularizer(x),
#
# where the smooth part is the objective of an EnveloptNLPModel and
# the regularizer is handled by R2N through its proximal operator.
mutable struct R2NEnveloptSubSolver{
  E <: EnveloptNLPModel,
  R <: AbstractRegularizedNLPModel,
  S <: R2NSolver,
  ST <: GenericExecutionStats,
  O <: NamedTuple,
} <: Envelopt.AbstractEnveloptSubSolver
  env_model::E  # the model passed to envelopt
  reg_nlp::R    # smooth part (env_model or a quasi-Newton model of it) + regularizer
  solver::S
  stats::ST
  options::O    # options passed to R2N at each solve
  name::String
end

const r2n_fixed_options = (max_iter = 10000, max_time = 600.0)

# ... constructor
function Envelopt.R2NEnveloptSubSolver(
  env_model::EnveloptNLPModel,
  regularizer;
  hessian_approximation::Symbol = :exact_f,
  mem::Int = 5,
  subsolver = R2Solver,
  m_monotone::Int = 1,
  selected = 1:get_nvar(env_model),
  kwargs...,
)
  @debug "initializing R2N subproblem solver"
  get_ncon(env_model) == 0 || error(
    "R2NEnveloptSubSolver only supports problems without constraints c(x) ∈ C; include them in F",
  )

  # The Hessian of env_model is ∇²f + 0 because its hprod! is that of the inner model.
  smooth_model = if hessian_approximation == :exact_f
    env_model
  elseif hessian_approximation == :lbfgs
    LBFGSModel(env_model; mem = mem)
  elseif hessian_approximation == :lsr1
    LSR1Model(env_model; mem = mem)
  else
    throw(
      ArgumentError(
        "unknown Hessian approximation :$hessian_approximation; choose :exact_f, :lbfgs or :lsr1",
      ),
    )
  end

  reg_nlp = RegularizedNLPModel(smooth_model, regularizer, selected)

  # Allocate once: envelopt updates μ and y in env_model in place,
  # so reg_nlp and the solver workspace remain valid across outer iterations.
  solver = R2NSolver(reg_nlp; subsolver = subsolver, m_monotone = m_monotone)
  stats = RegularizedExecutionStats(reg_nlp)
  options = merge(r2n_fixed_options, values(kwargs))
  return R2NEnveloptSubSolver(env_model, reg_nlp, solver, stats, options, "R2N")
end

# ... solve
function (sub::R2NEnveloptSubSolver)(
  env_model::EnveloptNLPModel,
  x0::AbstractVector,
  args...;
  tol::Real = 1.0e-6,
  kwargs...,
)
  env_model === sub.env_model ||
    error("R2NEnveloptSubSolver was constructed for a different EnveloptNLPModel")
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

Envelopt.regularizer_value(sub::R2NEnveloptSubSolver, x::AbstractVector) =
  @views sub.reg_nlp.h(x[sub.reg_nlp.selected])

end
