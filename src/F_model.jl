using LinearAlgebra: mul!
using SparseArrays: SparseMatrixCSC, findnz, nonzeros, nnz

export AbstractFModel, IdentityFModel, LinearFModel, FunctionFModel

"""
    AbstractFModel{T, S} <: AbstractNLPModel{T, S}

Supertype of lightweight models representing a mapping F : ℝⁿ → ℝᵐ for use in `EnveloptNLPModel`.

Conceptually, an `AbstractFModel` is the feasibility problem

    minimize_x 0  subject to  F(x) = 0,

but only the parts of the NLPModels API that `EnveloptNLPModel` relies on are meaningful:
`cons!` evaluates F(x), and `jprod!` / `jtprod!` compute products with the Jacobian of F.
The objective is identically zero and no Hessian is available.

Subtyping `AbstractNLPModel` keeps the type constraint `FMODEL <: AbstractNLPModel{T, S}`
in `EnveloptNLPModel` valid, so any other `AbstractNLPModel` can still be passed as `F`.

Concrete subtypes must have fields `meta::NLPModelMeta{T, S}` and `counters::Counters`.

Only the "full" API (`cons!`, `jprod!`, `jtprod!`, `jac_structure!`, `jac_coord!`) is implemented;
the linear/nonlinear split (`cons_lin!`, `cons_nln!`, ...) is deliberately not used, so `meta.lin` is empty.
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
mutable struct IdentityFModel{T, S} <: AbstractFModel{T, S}
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

function NLPModels.jprod!(F::IdentityFModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck F.meta.nvar x v
  @lencheck F.nx Jv
  NLPModels.increment!(F, :neval_jprod)
  @views Jv .= v[1:(F.nx)]
  return Jv
end

function NLPModels.jtprod!(F::IdentityFModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
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

################################################################################
# Affine mapping: F(x) = A x + b
################################################################################

"""
    LinearFModel(A::AbstractMatrix{T}, b::S = zeros(T, size(A, 1)))

Represents the affine mapping F(x) = A x + b, whose Jacobian is A.
`A` may be dense or a `SparseMatrixCSC`; in the latter case, the Jacobian structure
is that of the stored entries of `A`.
"""
mutable struct LinearFModel{T, S, M <: AbstractMatrix{T}} <: AbstractFModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  A::M
  b::S
end

function LinearFModel(A::M, b::S) where {T, S <: AbstractVector{T}, M <: AbstractMatrix{T}}
  m, n = size(A)
  @lencheck m b
  nnzj = A isa SparseMatrixCSC ? nnz(A) : m * n
  meta = _fmodel_meta(fill!(similar(b, n), zero(T)), m, nnzj, "linear-F")
  return LinearFModel{T, S, M}(meta, Counters(), A, b)
end

LinearFModel(A::AbstractMatrix{T}) where {T} = LinearFModel(A, zeros(T, size(A, 1)))

function NLPModels.cons!(F::LinearFModel{T}, x::AbstractVector, c::AbstractVector) where {T}
  @lencheck F.meta.nvar x
  @lencheck F.meta.ncon c
  NLPModels.increment!(F, :neval_cons)
  c .= F.b
  mul!(c, F.A, x, one(T), one(T))  # c <- A x + b
  return c
end

function NLPModels.jprod!(F::LinearFModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck F.meta.nvar x v
  @lencheck F.meta.ncon Jv
  NLPModels.increment!(F, :neval_jprod)
  mul!(Jv, F.A, v)
  return Jv
end

function NLPModels.jtprod!(F::LinearFModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck F.meta.nvar x Jtv
  @lencheck F.meta.ncon v
  NLPModels.increment!(F, :neval_jtprod)
  mul!(Jtv, F.A', v)
  return Jtv
end

# Sparse A: structure = stored entries of A, in storage (column-major) order.
function NLPModels.jac_structure!(
  F::LinearFModel{T, S, <:SparseMatrixCSC},
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
) where {T, S}
  @lencheck F.meta.nnzj rows cols
  I, J, _ = findnz(F.A)
  rows .= I
  cols .= J
  return rows, cols
end

function NLPModels.jac_coord!(
  F::LinearFModel{T, S, <:SparseMatrixCSC},
  x::AbstractVector,
  vals::AbstractVector,
) where {T, S}
  @lencheck F.meta.nvar x
  @lencheck F.meta.nnzj vals
  NLPModels.increment!(F, :neval_jac)
  vals .= nonzeros(F.A)
  return vals
end

# Dense A: structure = all entries of A, in column-major order.
function NLPModels.jac_structure!(
  F::LinearFModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck F.meta.nnzj rows cols
  m, n = size(F.A)
  k = 1
  for j = 1:n, i = 1:m
    rows[k] = i
    cols[k] = j
    k += 1
  end
  return rows, cols
end

function NLPModels.jac_coord!(F::LinearFModel, x::AbstractVector, vals::AbstractVector)
  @lencheck F.meta.nvar x
  @lencheck F.meta.nnzj vals
  NLPModels.increment!(F, :neval_jac)
  vals .= vec(F.A)
  return vals
end

################################################################################
# General mapping defined by user callbacks
################################################################################

"""
    FunctionFModel(Fx!, Jv!, Jtv!, x0::S, ncon::Integer; name = "function-F")

Represents a general mapping F : ℝⁿ → ℝᵐ, with n = length(x0) and m = ncon, defined by
three in-place callbacks:

* `Fx!(Fx, x)`: store F(x) in `Fx`;
* `Jv!(Jv, x, v)`: store ∇F(x) v in `Jv`;
* `Jtv!(Jtv, x, v)`: store ∇F(x)ᵀ v in `Jtv`.

The sparse Jacobian (`jac_structure!` / `jac_coord!`) is not available for this model.
`x0` only serves to determine the number of variables and the storage type `S`.
"""
mutable struct FunctionFModel{T, S, FF, FJ, FJT} <: AbstractFModel{T, S}
  meta::NLPModelMeta{T, S}
  counters::Counters
  Fx!::FF
  Jv!::FJ
  Jtv!::FJT
end

function FunctionFModel(
  Fx!::FF,
  Jv!::FJ,
  Jtv!::FJT,
  x0::S,
  ncon::Integer;
  name::String = "function-F",
) where {T, S <: AbstractVector{T}, FF, FJ, FJT}
  meta = _fmodel_meta(copy(x0), ncon, 0, name; jac_available = false)
  return FunctionFModel{T, S, FF, FJ, FJT}(meta, Counters(), Fx!, Jv!, Jtv!)
end

function NLPModels.cons!(F::FunctionFModel, x::AbstractVector, c::AbstractVector)
  @lencheck F.meta.nvar x
  @lencheck F.meta.ncon c
  NLPModels.increment!(F, :neval_cons)
  F.Fx!(c, x)
  return c
end

function NLPModels.jprod!(F::FunctionFModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck F.meta.nvar x v
  @lencheck F.meta.ncon Jv
  NLPModels.increment!(F, :neval_jprod)
  F.Jv!(Jv, x, v)
  return Jv
end

function NLPModels.jtprod!(F::FunctionFModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck F.meta.nvar x Jtv
  @lencheck F.meta.ncon v
  NLPModels.increment!(F, :neval_jtprod)
  F.Jtv!(Jtv, x, v)
  return Jtv
end