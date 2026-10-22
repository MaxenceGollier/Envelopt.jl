"""
    MoreauEnvelope(h, μ, z)

Non-allocating Moreau envelope of `h` with parameter `μ > 0`:

    h_μ(z) = min_u  h(u) + ‖u - z‖² / (2μ)  =  h(p) + ‖z - p‖² / (2μ),   p = prox_{μh}(z).

`h` can be any function implementing `ProximalCore.prox!(y, h, x, γ)`, which must return `h(y)`.
This includes all functions from ProximalOperators.jl.
`z` is only used to allocate internal buffers with the right size and storage type.

After each evaluation, `e.p` contains `prox_{μh}(z)` at the last point `z` evaluated.
"""
mutable struct MoreauEnvelope{T <: Real, H, S <: AbstractVector{T}}
  h::H
  μ::T
  p::S  # prox_{μh}(z) at the last evaluated z
  r::S  # z - p at the last evaluated z
end

function MoreauEnvelope(h::H, μ::T, z::S) where {T <: Real, H, S <: AbstractVector{T}}
  μ > 0 || error("Moreau envelope parameter must be > 0")
  return MoreauEnvelope{T, H, S}(h, μ, similar(z), similar(z))
end

# Value of h_μ at z. Stores prox_{μh}(z) in e.p and z - prox_{μh}(z) in e.r.
function (e::MoreauEnvelope)(z::AbstractVector)
  hp = prox!(e.p, e.h, z, e.μ)  # returns h(p)
  e.r .= z .- e.p
  return hp + dot(e.r, e.r) / (2 * e.μ)
end

"""
    val = moreau_value_grad!(g, e::MoreauEnvelope, z)

Compute the value of the Moreau envelope `e` at `z` and its gradient

    ∇h_μ(z) = (z - prox_{μh}(z)) / μ

with a single prox evaluation. The gradient is stored in `g`, which may alias `z`.
"""
function moreau_value_grad!(g::AbstractVector, e::MoreauEnvelope, z::AbstractVector)
  val = e(z)
  g .= e.r ./ e.μ
  return val
end
