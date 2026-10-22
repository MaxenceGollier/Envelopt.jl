@testitem "IdentityFModel: identity mapping" tags=[:fmodel] begin
  using LinearAlgebra, NLPModels, SparseArrays
  for T in (Float32, Float64)
    n = 4
    F = IdentityFModel(zeros(T, n))
    @test get_nvar(F) == n
    @test get_ncon(F) == n
    @test get_nnzj(F) == n
    x = T[1, -2, 3, -4]
    v = T[5, 6, 7, 8]
    J = Matrix{T}(I, n, n)
    @test cons(F, x) == x
    @test jprod(F, x, v) == J * v
    @test jtprod(F, x, v) == J' * v
    rows, cols = jac_structure(F)
    vals = jac_coord(F, x)
    @test eltype(vals) == T
    @test Matrix(sparse(rows, cols, vals, n, n)) == J
    @test obj(F, x) == zero(T)
    @test grad(F, x) == zeros(T, n)
  end
end

@testitem "IdentityFModel: selection mapping" tags=[:fmodel] begin
  using LinearAlgebra, NLPModels, SparseArrays
  for T in (Float32, Float64)
    nx, nr = 3, 2
    n = nx + nr
    F = IdentityFModel(ones(T, n), nx)
    @test get_nvar(F) == n
    @test get_ncon(F) == nx
    x = T[1, 2, 3, 4, 5]
    J = [Matrix{T}(I, nx, nx) zeros(T, nx, nr)]
    @test cons(F, x) == x[1:nx]
    v = T[-1, -2, -3, -4, -5]
    @test jprod(F, x, v) == J * v
    w = T[10, 20, 30]
    @test jtprod(F, x, w) == J' * w
    rows, cols = jac_structure(F)
    vals = jac_coord(F, x)
    @test Matrix(sparse(rows, cols, vals, nx, n)) == J
  end
end

@testitem "IdentityFModel: invalid dimensions" tags=[:fmodel] begin
  @test_throws ErrorException IdentityFModel(zeros(3), 0)
  @test_throws ErrorException IdentityFModel(zeros(3), 4)
end

@testitem "IdentityFModel: counters" tags=[:fmodel] begin
  using LinearAlgebra, NLPModels, SparseArrays
  F = IdentityFModel(zeros(2))
  x = [1.0, 2.0]
  cons(F, x)
  jtprod(F, x, x)
  @test neval_cons(F) == 1
  @test neval_jtprod(F) == 1
  reset!(F)
  @test neval_cons(F) == 0
end

@testitem "IdentityFModel: default F preserves element type" tags=[:fmodel] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1)^2 + 100 * (x[2] - x[1]^2)^2, Float32[-1.2; 1.0])
  env_model = EnveloptNLPModel(model, NormL1(1.0f0))
  @test env_model.F isa IdentityFModel{Float32, Vector{Float32}}
  @test eltype(obj(env_model, get_x0(env_model))) == Float32
end
