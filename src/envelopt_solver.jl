export envelopt, EnveloptSolver, solve!

import SolverCore.solve!

# abstract type for Envelopt subproblem solvers
abstract type AbstractEnveloptSubSolver end
name(sub::AbstractEnveloptSubSolver) = sub.name
failed(stats::GenericExecutionStats) = stats.status != :first_order
first_order(stats::GenericExecutionStats) = stats.status == :first_order
include("madnlp_sub.jl")

"""
    solver = EnveloptSolver(env_model; subsolver = MadNLPEnveloptSubSolver(env_model))

Preallocate the workspace used by [`envelopt`](@ref) to solve `env_model`, so that
`solve!(solver, env_model, stats)` can be called repeatedly, e.g., in a loop.

The subproblem solver `subsolver` is stored in the workspace and reused across calls.
"""
mutable struct EnveloptSolver{T <: Real, V <: AbstractVector{T}, Sub <: AbstractEnveloptSubSolver} <:
               AbstractOptimizationSolver
  xk::V    # current iterate
  x0::V    # copy of get_x0(env_model), restored at the end of solve!
  y0::V    # copy of get_y0(env_model), restored at the end of solve!
  Fk::V    # F(xk)
  Fμyk::V  # F(xk) + μ y
  uk::V    # prox_{μh}(F(xk) + μ y)
  dk::V    # F(xk) - uk
  yk::V    # updated multiplier estimate for F(x) - u = 0 (same size as env_model.y)
  subsolver::Sub
end

function EnveloptSolver(
  env_model::EnveloptNLPModel{T, V};
  subsolver::AbstractEnveloptSubSolver = MadNLPEnveloptSubSolver(env_model),
) where {T, V}
  x0 = get_x0(env_model)
  xk = similar(x0)
  x0_copy = similar(x0)
  y0_copy = similar(get_y0(env_model))
  Fk = similar(env_model.y)
  Fμyk = similar(env_model.y)
  uk = similar(env_model.y)
  dk = similar(env_model.y)
  yk = similar(env_model.y)
  return EnveloptSolver{T, V, typeof(subsolver)}(
    xk,
    x0_copy,
    y0_copy,
    Fk,
    Fμyk,
    uk,
    dk,
    yk,
    subsolver,
  )
end

"""
    stats = envelopt(model::AbstractNLPModel, h; kwargs...)
    stats = envelopt(model::AbstractNLPModel, F::AbstractNLPModel, h; kwargs...)
    stats = envelopt(env_model::EnveloptNLPModel; kwargs...)

A solver for the nonsmooth regularized problem

    minimize f(x) + h(F(x))  subject to  c(x) ∈ C,

where h is a proper, closed, convex function, and f, F and c are smooth.
In the first calling form, the `AbstractNLPModel` represents the problem

    minimize f(x) subject to  c(x) ∈ C,

and F is implicity set to the identity mapping (F(x) = x).

In the last calling form, the `EnveloptNLPModel` will have been constructed from a user-defined F.

The Envelopt solver solves a sequence of Moreau envelope approximations of the problem via an augmented Lagrangian method.
Each subproblem has the form

    minimize f(x) + h_μ (F(x) + μ y)  subject to  c(x) ∈ C,

where h_μ is the Moreau envelope of h with parameter μ > 0, and y is a dual variable estimate.
The subproblems are solved with a solver chosen by the user using a quasi-Newton approximation of the Hessian.

For advanced usage, first preallocate the workspace and then call `solve!`:

    solver = EnveloptSolver(env_model; subsolver = MadNLPEnveloptSubSolver(env_model))
    stats = GenericExecutionStats(env_model)
    solve!(solver, env_model, stats; kwargs...)

# Keyword arguments

- `subsolver::AbstractEnveloptSubSolver = MadNLPEnveloptSubSolver(env_model)`: the subproblem solver (only for `envelopt`; with `solve!`, it is part of the `EnveloptSolver`);
- `x::V = get_x0(env_model)`: the initial guess;
- `y::V = get_y0(env_model)`: the initial estimate of the multipliers of the constraints c(x) ∈ C;
- `verbose::Bool = true`: whether to display a log of the outer iterations;
- `max_outer::Int = 20`: maximum number of outer iterations;
- `dtol_min::Float64 = 1.0e-6`: final dual (subproblem KKT) tolerance;
- `ptol_min::Float64 = 1.0e-6`: final primal tolerance on ‖F(x) - u‖;
- `indent::String = ""`: prefix of each log line;
- `callback = (args...) -> nothing`: see below.

# Output

The value returned is a `GenericExecutionStats`, see `SolverCore.jl`, where

- `stats.iter` is the number of outer iterations, i.e., of subproblems solved;
- `stats.objective` is f(x) + h(u);
- `stats.dual_feas`, `stats.primal_feas`, `stats.multipliers`, `stats.multipliers_L` and `stats.multipliers_U` are those of the last subproblem;
- `stats.solver_specific[:u]` is u = prox_{μh}(F(x) + μ y);
- `stats.solver_specific[:inner_iter]` is the total number of iterations of the subproblem solver;
- `stats.solver_specific[:smooth_obj]` and `stats.solver_specific[:nonsmooth_obj]` are f(x) and h(u).

On return, `get_x0(env_model)` and `get_y0(env_model)` are restored to their values on entry.

# Callback

The callback is called at the beginning and after each outer iteration as `callback(env_model, solver, stats)`.
It can be used to update the model, e.g., if it is an `NCLModel`, or to stop the solver
by setting `stats.status = :user` with `set_status!(stats, :user)`.
"""
function envelopt(model::AbstractNLPModel, args...; kwargs...)
  env_model = EnveloptNLPModel(model, args...)
  return envelopt(env_model; kwargs...)
end

function envelopt(
  env_model::EnveloptNLPModel;
  subsolver::AbstractEnveloptSubSolver = MadNLPEnveloptSubSolver(env_model),
  kwargs...,
)
  solver = EnveloptSolver(env_model; subsolver = subsolver)
  return solve!(solver, env_model; kwargs...)
end

function SolverCore.solve!(solver::EnveloptSolver, env_model::EnveloptNLPModel; kwargs...)
  stats = GenericExecutionStats(env_model)
  return solve!(solver, env_model, stats; kwargs...)
end

function SolverCore.solve!(
  solver::EnveloptSolver{T, V},
  env_model::EnveloptNLPModel{T, V},
  stats::GenericExecutionStats{T, V};
  callback = (args...) -> nothing,
  x::V = get_x0(env_model),
  y::V = get_y0(env_model),
  verbose::Bool = true,
  max_outer::Int = 20,
  dtol_min::Float64 = 1.0e-6,
  ptol_min::Float64 = 1.0e-6,
  indent::String = "",
) where {T, V}
  SolverCore.reset!(stats)
  NLPModels.reset!(env_model)
  NLPModels.reset!(env_model.model) # TODO NLPModels.reset! should reset the model and its submodels recursively.
  NLPModels.reset!(env_model.F)

  f, h = env_model.model, env_model.h
  subsolver = solver.subsolver
  substats = subsolver.stats

  # Retrieve workspace.
  # Some subsolvers (e.g., MadNLP) overwrite get_x0(env_model) and get_y0(env_model) to warm start,
  # so we save them here and restore them on return.
  x0 = solver.x0 .= get_x0(env_model)
  y0 = solver.y0 .= get_y0(env_model)
  xk = solver.xk .= x
  Fk, Fμyk, uk, dk, yk = solver.Fk, solver.Fμyk, solver.uk, solver.dk, solver.yk

  # set_constraint_multipliers!(substats, y)
  substats.multipliers .= y
  substats.multipliers_L .= 0
  substats.multipliers_U .= 0

  set_multiplier!(env_model, zero(T))  # FIXME: smarter initialization
  set_penalty!(env_model, one(T))      # initial penalty

  dtol = 1.0e-1
  ptol = 1.0e-1
  tot_inner_iter = 0

  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)

  # We don't have u at this point.
  # To save a prox evaluation, we use h(F(x)) instead of h(u).
  fk = obj(f, xk)
  eval_F!(env_model, xk, Fk)
  hk = h(Fk)
  uk .= Fk
  @. Fμyk = Fk + env_model.μ * env_model.y
  env_val = obj(env_model, fk, Fμyk)
  yNorm = norm(env_model.y)

  log_line = ""
  if verbose
    @info @sprintf "%s%4s  %8s  %8s  %8s  %7s  %7s  %7s  %7s  %7s  %7s  %7s  %5s  %1s\n" indent "iter" "f" "h" "f + h_μ" "‖y‖" "μ" "ϵd" "ϵp" "dfeas" "pfeas" "‖F - u‖" "inner" "type"
    log_line =
      @sprintf "%s%4d  %8.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %7.1e  " indent stats.iter fk hk env_val yNorm env_model.μ dtol ptol
  end

  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)
  set_solver_specific!(stats, :inner_iter, tot_inner_iter)
  set_status!(stats, stats.iter ≥ max_outer ? :max_iter : :unknown)

  callback(env_model, solver, stats)

  done = stats.status != :unknown

  # FIXME: smarter stopping condition
  while !done
    # solve subproblem with xk as initial guess
    subsolver(env_model, xk, stats.iter; tol = dtol)
    set_iter!(stats, stats.iter + 1)
    tot_inner_iter += substats.iter
    set_solver_specific!(stats, :inner_iter, tot_inner_iter)

    if failed(substats)
      verbose && @error "subproblem solver fails with" substats.status
      # FIXME: try to recover
      set_status!(stats, :exception)
      break
    end

    xk .= substats.solution
    eval_F!(env_model, xk, Fk)
    @. Fμyk = Fk + env_model.μ * env_model.y
    prox!(uk, h, Fμyk, env_model.μ)
    @. dk = Fk - uk
    lift_feasibility = norm(dk)
    kkt = max(substats.dual_feas, substats.primal_feas)

    feasibility = lift_feasibility
    if isa(f, NCLModel)
      rNorm = norm(@view xk[(f.nx + 1):end])
      feasibility = norm((lift_feasibility, rNorm))
    end

    if verbose
      log_line *=
        @sprintf "%7.1e  %7.1e  %7.1e  %5d  " substats.dual_feas substats.primal_feas feasibility substats.iter
    end

    if feasibility ≤ ptol
      @. yk = env_model.y + dk / env_model.μ
      set_multiplier!(env_model, yk)
      yNorm = norm(env_model.y)
      if isa(f, NCLModel)
        rk = @view xk[(f.nx + 1):end]
        @. f.y += f.ρ * rk
      end
      dtol = max(dtol_min / 2, min(dtol, kkt) / 5)
      ptol = max(ptol_min / 2, feasibility / 5)
      verbose && (log_line *= @sprintf "y\n")
    else
      set_penalty!(env_model, env_model.μ / 5)
      if isa(f, NCLModel)
        # NCLModel uses penalty parameter ρ whereas Envelopt uses μ⁻¹.
        f.ρ *= 5
      end
      dtol = max(dtol_min / 2, min(dtol, kkt) / 2)
      ptol = max(ptol_min / 2, feasibility / 2)
      verbose && (log_line *= @sprintf "μ\n")
    end
    verbose && @info log_line

    fk = obj(f, xk)
    hk = h(uk)
    env_val = obj(env_model, fk, Fμyk)

    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_time!(stats, time() - start_time)

    if verbose
      log_line =
        @sprintf "%s%4d  %8.1e  %8.1e  %8.1e  %7.1e  %7.1e  %7.1e  %7.1e  " indent stats.iter fk hk env_val yNorm env_model.μ dtol ptol
    end

    dual_feasible = kkt ≤ dtol_min
    primal_feasible = feasibility ≤ ptol_min
    stationary = dual_feasible && primal_feasible
    set_status!(
      stats,
      stationary ? :first_order : (stats.iter ≥ max_outer ? :max_iter : :unknown),
    )

    callback(env_model, solver, stats)

    done = stats.status != :unknown
  end
  verbose && @info log_line

  if stats.status == :max_iter
    verbose && @info "$(indent)maximum number of outer iterations reached"
  elseif stats.status == :first_order
    verbose && @info "$(indent)found an approximate stationary point"
  end

  copyto!(get_x0(env_model), x0)  # restore env_model.meta.x0
  copyto!(get_y0(env_model), y0)  # restore env_model.meta.y0

  if stats.status == :exception
    # Report the last iterate of the subproblem solver, as before the workspace was preallocated.
    set_solution!(stats, substats.solution)
    set_objective!(stats, obj(f, stats.solution) + hk)
  else
    set_solution!(stats, xk)
  end
  set_residuals!(stats, substats.primal_feas, substats.dual_feas)
  set_constraint_multipliers!(stats, substats.multipliers)
  if length(stats.multipliers_L) == length(substats.multipliers_L)
    # stats has no storage for bound multipliers if env_model has no bounds
    set_bounds_multipliers!(stats, substats.multipliers_L, substats.multipliers_U)
  end
  u = get!(() -> similar(uk), stats.solver_specific, :u)  # only allocate on first use of stats
  u .= uk
  set_solver_specific!(stats, :u, u)
  set_time!(stats, time() - start_time)
  return stats
end
