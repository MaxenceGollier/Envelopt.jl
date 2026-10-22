@testitem "Moreau envelope matches ProximalOperators" tags=[:envelope] begin
  using LinearAlgebra, ProximalOperators
  for h in (NormL1(1.0), NormL2(2.0), NormLinf(0.5), IndBox(-1.0, 1.0)), μ in (1.0, 0.1)
    z = [3.0, -0.2, 0.7, -4.0, 0.05]
    e = Envelopt.MoreauEnvelope(h, μ, z)
    ref = ProximalOperators.MoreauEnvelope(h, μ)
    g_ref, v_ref = ProximalOperators.gradient(ref, z)
    @test e(z) ≈ v_ref
    g = similar(z)
    @test Envelopt.moreau_value_grad!(g, e, z) ≈ v_ref
    @test g ≈ g_ref
    w = copy(z)  # gradient in place of the input
    Envelopt.moreau_value_grad!(w, e, w)
    @test w ≈ g_ref
  end
end

@testitem "Moreau envelope does not allocate" tags=[:envelope, :allocs] begin
  using ProximalOperators
  z = randn(10)
  g = similar(z)
  e = Envelopt.MoreauEnvelope(NormL1(1.0), 0.5, z)
  value_allocs(e, z) = @allocated e(z)
  grad_allocs(g, e, z) = @allocated Envelopt.moreau_value_grad!(g, e, z)
  value_allocs(e, z); grad_allocs(g, e, z)  # compile
  @test value_allocs(e, z) == 0
  @test grad_allocs(g, e, z) == 0
end

@testitem "EnveloptNLPModel obj/grad do not allocate with IdentityFModel" tags=[:model, :allocs] begin
  using NLPModels, ProximalOperators

  # Minimal non-allocating smooth model: f(x) = ½‖x‖².
  struct QuadModel <: AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModelMeta{Float64, Vector{Float64}}
    counters::Counters
  end
  QuadModel(n) = QuadModel(NLPModelMeta(n, x0 = ones(n)), Counters())
  NLPModels.obj(::QuadModel, x::AbstractVector) = sum(abs2, x) / 2
  NLPModels.grad!(::QuadModel, x::AbstractVector, g::AbstractVector) = (g .= x)

  model = QuadModel(5)
  env_model = EnveloptNLPModel(model, NormL1(1.0))
  x = randn(5)
  g = similar(x)
  obj_allocs(m, x) = @allocated obj(m, x)
  grad_allocs(m, x, g) = @allocated grad!(m, x, g)
  objgrad_allocs(m, x, g) = @allocated objgrad!(m, x, g)
  obj_allocs(env_model, x); grad_allocs(env_model, x, g); objgrad_allocs(env_model, x, g)
  @test obj_allocs(env_model, x) == 0
  @test grad_allocs(env_model, x, g) == 0
  @test objgrad_allocs(env_model, x, g) == 0
  set_penalty!(env_model, 0.1)
  @test env_model.envelope.μ == 0.1
  @test objgrad!(env_model, x, g)[1] ≈ obj(env_model, x)
end
