export AbstractFModel, IdentityFModel

"""
    AbstractFModel{T, S} <: AbstractNLPModel{T, S}

Supertype of lightweight models representing a mapping F : ℝⁿ → ℝᵐ
for use in `EnveloptNLPModel`.

Conceptually, an `AbstractFModel` is the feasibility problem

    minimize_x 0  subject to  F(x) = 0,
"""
abstract type AbstractFModel{T, S} <: AbstractNLPModel{T, S} end

# Build the meta of an F model with `length(x0)` variables and `ncon` constraints F(x) = 0.
function _fmodel_meta(
  x0::S,
  ncon::Integer,
  nnzj::Integer,
  name::String;
  jac_available::Bool = true,
) where {T, S <: AbstractVector{T}}
  ncon > 0 || error("F must have at least one component")
  z = fill!(similar(x0, ncon), zero(T))
  return NLPModelMeta{T, S}(
    length(x0);
    x0 = x0,
    ncon = ncon,
    lcon = z,
    ucon = copy(z),
    y0 = copy(z),
    nnzj = nnzj,
    nnzh = 0,
    jac_available = jac_available,
    hess_available = false,
    hprod_available = false,
    name = name,
  )
end

# The objective of an F model is identically zero.

function NLPModels.obj(F::AbstractFModel{T, S}, x::AbstractVector) where {T, S}
  @lencheck F.meta.nvar x
  NLPModels.increment!(F, :neval_obj)
  return zero(T)
end

function NLPModels.grad!(F::AbstractFModel, x::AbstractVector, g::AbstractVector)
  @lencheck F.meta.nvar x g
  NLPModels.increment!(F, :neval_grad)
  fill!(g, zero(eltype(g)))
  return g
end

################################################################################
# Identity / selection mapping: F(x) = x[1:nx]
################################################################################

"""
    IdentityFModel(x0::S, nx::Integer = length(x0))

Represents the mapping F(x) = x[1:nx], whose Jacobian is [I 0].

* With `nx = length(x0)` (the default), this is the identity mapping F(x) = x.
* With `nx < length(x0)`, this is the mapping F(x, r) = x used for `NCLModel`s,
  where `x0` has `nx + nr` components.

`x0` only serves to determine the number of variables and the storage type `S`.
"""
struct IdentityFModel{T, S} <: AbstractFModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  nx::Int
end

function IdentityFModel(x0::S, nx::Integer = length(x0)) where {T, S <: AbstractVector{T}}
  0 < nx ≤ length(x0) || error("nx must satisfy 0 < nx ≤ length(x0)")
  meta = _fmodel_meta(fill!(similar(x0), zero(T)), nx, nx, "identity-F")
  return IdentityFModel{T, S}(meta, Counters(), nx)
end

function NLPModels.cons!(F::IdentityFModel, x::AbstractVector, c::AbstractVector)
  @lencheck F.meta.nvar x
  @lencheck F.nx c
  NLPModels.increment!(F, :neval_cons)
  @views c .= x[1:(F.nx)]
  return c
end

function NLPModels.jprod!(
  F::IdentityFModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck F.meta.nvar x v
  @lencheck F.nx Jv
  NLPModels.increment!(F, :neval_jprod)
  @views Jv .= v[1:(F.nx)]
  return Jv
end

function NLPModels.jtprod!(
  F::IdentityFModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck F.meta.nvar x Jtv
  @lencheck F.nx v
  NLPModels.increment!(F, :neval_jtprod)
  @views Jtv[1:(F.nx)] .= v
  @views Jtv[(F.nx + 1):end] .= zero(eltype(Jtv))
  return Jtv
end

function NLPModels.jac_structure!(
  F::IdentityFModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck F.meta.nnzj rows cols
  rows .= 1:(F.nx)
  cols .= 1:(F.nx)
  return rows, cols
end

function NLPModels.jac_coord!(F::IdentityFModel, x::AbstractVector, vals::AbstractVector)
  @lencheck F.meta.nvar x
  @lencheck F.meta.nnzj vals
  NLPModels.increment!(F, :neval_jac)
  fill!(vals, one(eltype(vals)))
  return vals
end
