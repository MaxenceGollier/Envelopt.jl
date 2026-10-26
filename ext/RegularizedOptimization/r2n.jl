# R2N subproblem solver
function Envelopt.R2NEnveloptSubSolver(
  env_model::EnveloptNLPModel,
  regularizer;
  hessian_approximation::Symbol = :lbfgs,
  mem::Int = 5,
  subsolver = R2Solver,
  m_monotone::Int = 1,
  selected = 1:get_nvar(env_model),
  kwargs...,
)
  @debug "initializing R2N subproblem solver"
  check_unconstrained(env_model, "R2NEnveloptSubSolver")

  # quasi-Newton approximation of the Hessian of the whole smooth objective f(x) + h_μ(F(x) + μy)
  smooth_model = if hessian_approximation == :lbfgs
    LBFGSModel(env_model; mem = mem)
  elseif hessian_approximation == :lsr1
    LSR1Model(env_model; mem = mem)
  else
    throw(
      ArgumentError(
        "unknown Hessian approximation :$hessian_approximation; choose :lbfgs or :lsr1",
      ),
    )
  end

  make_solver = reg_nlp -> R2NSolver(reg_nlp; subsolver = subsolver, m_monotone = m_monotone)
  return RegOptEnveloptSubSolver(
    env_model,
    smooth_model,
    regularizer,
    selected,
    make_solver,
    "R2N",
    values(kwargs),
  )
end
