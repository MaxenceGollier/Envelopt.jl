# R2 subproblem solver
function Envelopt.R2EnveloptSubSolver(
  env_model::EnveloptNLPModel,
  regularizer;
  selected = 1:get_nvar(env_model),
  kwargs...,
)
  @debug "initializing R2 subproblem solver"
  check_unconstrained(env_model, "R2EnveloptSubSolver")
  # R2 only uses first-order information of env_model.
  make_solver = reg_nlp -> R2Solver(reg_nlp)
  return RegOptEnveloptSubSolver(
    env_model,
    env_model,
    regularizer,
    selected,
    make_solver,
    "R2",
    values(kwargs),
  )
end
