export EnveloptExecutionStats

"""
    stats = EnveloptExecutionStats(env_model)

Execution statistics returned by [`envelopt`](@ref) and `solve!(::EnveloptSolver, ...)`.

It wraps a `GenericExecutionStats` (see `SolverCore.jl`): all of its fields
(`stats.status`, `stats.solution`, `stats.objective`, `stats.iter`, ...) and setters
(`set_status!`, `set_solution!`, ...) are available on an `EnveloptExecutionStats`.
In addition, it has the following fields, all concretely typed, so that reading or updating
them neither allocates nor is type-unstable (as opposed to `solver_specific`):

- `u::V`: u = prox_{μh}(F(x) + μ y) at the last outer iteration;
- `inner_iter::Int`: the total number of iterations of the subproblem solver;
- `smooth_obj::T`: f(x);
- `nonsmooth_obj::T`: h(u).

The wrapped `GenericExecutionStats` is available as `stats.stats`.
"""
mutable struct EnveloptExecutionStats{
  T <: Real,
  V <: AbstractVector{T},
  GS <: GenericExecutionStats{T, V},
} <: AbstractExecutionStats
  stats::GS
  u::V
  inner_iter::Int
  smooth_obj::T
  nonsmooth_obj::T
end

function EnveloptExecutionStats(env_model::EnveloptNLPModel{T, V}) where {T, V}
  stats = GenericExecutionStats(env_model)
  return EnveloptExecutionStats(stats, similar(env_model.y), 0, T(Inf), T(Inf))
end

# Field access is forwarded to the wrapped GenericExecutionStats.
# `name` is a literal at call sites, so `hasfield` is constant-folded and access is type stable.
@inline function Base.getproperty(s::EnveloptExecutionStats, name::Symbol)
  return hasfield(typeof(s), name) ? getfield(s, name) : getproperty(getfield(s, :stats), name)
end

@inline function Base.setproperty!(s::EnveloptExecutionStats, name::Symbol, x)
  if hasfield(typeof(s), name)
    return setfield!(s, name, convert(fieldtype(typeof(s), name), x))
  else
    return setproperty!(getfield(s, :stats), name, x)
  end
end

Base.propertynames(s::EnveloptExecutionStats, private::Bool = false) =
  (fieldnames(typeof(s))..., propertynames(getfield(s, :stats), private)...)

for f in (
  :set_status!,
  :set_solution!,
  :set_objective!,
  :set_residuals!,
  :set_primal_residual!,
  :set_dual_residual!,
  :set_multipliers!,
  :set_constraint_multipliers!,
  :set_bounds_multipliers!,
  :set_iter!,
  :set_step_status!,
  :set_time!,
  :set_solver_specific!,
  :broadcast_solver_specific!,
)
  @eval function SolverCore.$f(s::EnveloptExecutionStats, args...)
    SolverCore.$f(getfield(s, :stats), args...)
    return s
  end
end

function SolverCore.reset!(s::EnveloptExecutionStats{T}) where {T}
  SolverCore.reset!(getfield(s, :stats))
  s.inner_iter = 0
  s.smooth_obj = T(Inf)
  s.nonsmooth_obj = T(Inf)
  return s
end

# Same as SolverCore.reset!(stats, nlp): adjust the preallocated vectors to the problem size.
function SolverCore.reset!(
  s::EnveloptExecutionStats{T, V},
  env_model::EnveloptNLPModel{T, V},
) where {T, V}
  SolverCore.reset!(getfield(s, :stats), env_model)
  s.u = similar(env_model.y)
  return SolverCore.reset!(s)
end

function Base.print(io::IO, s::EnveloptExecutionStats; showvec::Function = SolverCore.disp_vector)
  print(io, getfield(s, :stats); showvec = showvec)
  print(io, "  u: ")
  showvec(io, s.u)
  println(io, "")
  println(io, "  inner iterations: ", s.inner_iter)
  println(io, "  smooth objective: ", s.smooth_obj)
  println(io, "  nonsmooth objective: ", s.nonsmooth_obj)
end
Base.print(s::EnveloptExecutionStats; showvec::Function = SolverCore.disp_vector) =
  print(Base.stdout, s; showvec = showvec)
Base.println(io::IO, s::EnveloptExecutionStats; showvec::Function = SolverCore.disp_vector) =
  print(io, s; showvec = showvec)
Base.println(s::EnveloptExecutionStats; showvec::Function = SolverCore.disp_vector) =
  print(Base.stdout, s; showvec = showvec)
