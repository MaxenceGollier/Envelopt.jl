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

@testitem "simple unconstrained envelopt solve of a RegularizedNLPModel" tags=[
  :solver,
  :unconstrained,
  :r2,
  :r2n,
] begin
  using ADNLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  reg_nlp = RegularizedNLPModel(env_model, NormL1(0.1))
  stats = envelopt(reg_nlp, verbose = false)  # R2N by default
  @test stats.status == :first_order
  stats = envelopt(reg_nlp, subsolver = R2EnveloptSubSolver(reg_nlp), verbose = false)
  @test stats.status == :first_order
end

@testitem "solve! with a preallocated EnveloptSolver for a RegularizedNLPModel" tags=[
  :solver,
  :unconstrained,
  :r2,
  :r2n,
] begin
  using ADNLPModels,
    NLPModels, ProximalOperators, RegularizedOptimization, RegularizedProblems, SolverCore
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  reg_nlp = RegularizedNLPModel(env_model, NormL1(0.1))
  for subsolver in (R2NEnveloptSubSolver, R2EnveloptSubSolver)
    stats_ref = envelopt(reg_nlp, subsolver = subsolver(reg_nlp), verbose = false)
    @test stats_ref.status == :first_order
    @test get_x0(env_model) == [-1.2; 1.0]  # the initial guess is restored

    # solving again with a preallocated workspace gives the same results
    solver = EnveloptSolver(reg_nlp; subsolver = subsolver(reg_nlp))
    stats = EnveloptExecutionStats(reg_nlp)
    for _ = 1:2
      solve!(solver, reg_nlp, stats, verbose = false)
      @test stats.status == :first_order
      @test stats.solution == stats_ref.solution
      @test stats.objective == stats_ref.objective
      @test stats.iter == stats_ref.iter
      @test stats.inner_iter == stats_ref.inner_iter
      @test stats.u == stats_ref.u
    end
  end

  # a smooth subproblem solver would ignore the regularizer
  @test_throws TypeError EnveloptSolver(reg_nlp; subsolver = MadNLPEnveloptSubSolver(env_model))
  @test_throws ArgumentError solve!(EnveloptSolver(env_model), reg_nlp, verbose = false)
end
