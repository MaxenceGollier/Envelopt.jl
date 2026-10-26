@testitem "single subproblem solve with R2" tags=[:solver, :unconstrained, :r2] begin
  using ADNLPModels, NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  r2_solver = R2EnveloptSubSolver(env_model, NormL1(0.1))
  stats = r2_solver(env_model, get_x0(env_model), tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "single subproblem solve with R2N" tags=[:solver, :unconstrained, :r2n] begin
  using ADNLPModels, NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  for hessian_approximation in (:lbfgs, :lsr1)
    r2n_solver =
      R2NEnveloptSubSolver(env_model, NormL1(0.1); hessian_approximation = hessian_approximation)
    stats = r2n_solver(env_model, get_x0(env_model), tol = 1.0e-2)
    @test Envelopt.first_order(stats)
  end
end

@testitem "simple unconstrained envelopt solve with R2" tags=[:solver, :unconstrained, :r2] begin
  using ADNLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats = envelopt(
    env_model;
    subsolver = R2EnveloptSubSolver(env_model, NormL1(0.1)),
    verbose = false,
  )
  @test stats.status == :first_order
end

@testitem "simple unconstrained envelopt solve with R2N" tags=[:solver, :unconstrained, :r2n] begin
  using ADNLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats = envelopt(
    env_model;
    subsolver = R2NEnveloptSubSolver(env_model, NormL1(0.1)),
    verbose = false,
  )
  @test stats.status == :first_order
end

@testitem "simple bound-constrained envelopt solve with R2" tags=[
  :solver,
  :boundconstrained,
  :r2,
] begin
  using ADNLPModels,
    OptimizationProblems,
    OptimizationProblems.ADNLPProblems,
    ProximalOperators,
    RegularizedOptimization,
    RegularizedProblems
  model = hs1()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats = envelopt(
    env_model;
    subsolver = R2EnveloptSubSolver(env_model, NormL1(0.1)),
    verbose = false,
  )
  @test stats.status == :first_order
end

@testitem "simple bound-constrained envelopt solve with R2N" tags=[
  :solver,
  :boundconstrained,
  :r2n,
] begin
  using ADNLPModels,
    OptimizationProblems,
    OptimizationProblems.ADNLPProblems,
    ProximalOperators,
    RegularizedOptimization,
    RegularizedProblems
  model = hs1()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats = envelopt(
    env_model;
    subsolver = R2NEnveloptSubSolver(env_model, NormL1(0.1)),
    verbose = false,
  )
  @test stats.status == :first_order
end

@testitem "R2 and R2N do not support constrained problems" tags=[
  :solver,
  :constrained,
  :r2,
  :r2n,
] begin
  using ADNLPModels,
    OptimizationProblems,
    OptimizationProblems.ADNLPProblems,
    ProximalOperators,
    RegularizedOptimization,
    RegularizedProblems
  model = hs13()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  @test_throws ErrorException R2EnveloptSubSolver(env_model, NormL1(0.1))
  @test_throws ErrorException R2NEnveloptSubSolver(env_model, NormL1(0.1))
end
