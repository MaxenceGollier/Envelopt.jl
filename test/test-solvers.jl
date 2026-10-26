# using NLPModelsTest
# using Test

@testitem "single subproblem solve with Trunk" tags=[:solver, :unconstrained, :trunk] begin
  using ADNLPModels, JSOSolvers, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  trunk_solver = TrunkEnveloptSubSolver(env_model)
  stats = trunk_solver(env_model, get_x0(env_model), tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "single subproblem solve with TRON" tags=[:solver, :unconstrained, :tron] begin
  using ADNLPModels, JSOSolvers, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  tron_solver = TronEnveloptSubSolver(env_model)
  stats = tron_solver(env_model, get_x0(env_model), tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "single subproblem solve with MadNLP" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, NLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  madnlp_solver = MadNLPEnveloptSubSolver(env_model)
  stats = madnlp_solver(env_model, get_x0(env_model), 0, tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "single subproblem solve with IPOPT" tags=[:solver, :unconstrained, :ipopt] begin
  using ADNLPModels, NLPModels, NLPModelsIpopt, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  ipopt_solver = IPOPTEnveloptSubSolver(env_model)
  stats = ipopt_solver(env_model, get_x0(env_model), 0, tol = 1.0e-2)
  @test Envelopt.first_order(stats)
end

@testitem "simple solve with MadNLP" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, MadNLP, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  mad_solver =
    MadNLPSolver(env_model; hessian_approximation = MadNLP.CompactLBFGS, print_level = MadNLP.ERROR)
  stats = solve!(mad_solver)
  @test stats.status == MadNLP.SOLVE_SUCCEEDED
end

@testitem "simple unconstrained envelopt solve with IPOPT" tags=[:solver, :unconstrained, :ipopt] begin
  using ADNLPModels, NLPModelsIpopt, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats =
    envelopt(env_model, subsolver = IPOPTEnveloptSubSolver(env_model), verbose = false)
  @test stats.status == :first_order
end

@testitem "simple unconstrained envelopt solve with MadNLP" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats = envelopt(env_model, verbose = false)
  @test stats.status == :first_order
end

@testitem "simple unconstrained envelopt solve with Trunk" tags=[:solver, :unconstrained, :trunk] begin
  using ADNLPModels, JSOSolvers, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats =
    envelopt(env_model; subsolver = TrunkEnveloptSubSolver(env_model), verbose = false)
  @test stats.status == :first_order
end

@testitem "simple unconstrained envelopt solve with Tron" tags=[:solver, :unconstrained, :tron] begin
  using ADNLPModels, JSOSolvers, ProximalOperators
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats =
    envelopt(env_model; subsolver = TronEnveloptSubSolver(env_model), verbose = false)
  @test stats.status == :first_order
end

@testitem "simple bound-constrained envelopt solve with Tron" tags=[
  :solver,
  :boundconstrained,
  :tron,
] begin
  using ADNLPModels,
    JSOSolvers,
    OptimizationProblems,
    OptimizationProblems.ADNLPProblems,
    ProximalOperators
  model = hs1()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats =
    envelopt(env_model; subsolver = TronEnveloptSubSolver(env_model), verbose = false)
  @test stats.status == :first_order
end

@testitem "test constrained problem with MadNLP" tags=[:solvers, :constrained, :madnlp] begin
  using ADNLPModels, OptimizationProblems, OptimizationProblems.ADNLPProblems, ProximalOperators
  model = hs13()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats = envelopt(env_model, verbose = false)
  @test stats.status == :first_order
end

@testitem "test constrained problem with IPOPT" tags=[:solvers, :constrained, :ipopt] begin
  using ADNLPModels,
    NLPModelsIpopt,
    OptimizationProblems,
    OptimizationProblems.ADNLPProblems,
    ProximalOperators
  model = hs13()
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats =
    envelopt(env_model, subsolver = IPOPTEnveloptSubSolver(env_model), verbose = false)
  @test stats.status == :first_order
end

@testitem "test complementarity problem with NCL" tags=[:solvers, :constrained, :mpcc] begin
  using ADNLPModels, NCL, ProximalOperators
  # min (x₁ - 1)² + (x₂ - 1)²  s.t. x₁ * x₂ = 0, x₁ ≥ 0, x₂ \geq 0.
  model = ADNLPModel(
    x -> (x[1] - 1)^2 + (x[2] - 1)^2,
    [1.2; 1.2],
    [0.0, 0.0],
    [Inf, Inf],
    x -> [x[1] * x[2]],
    [0.0],
    [0.0],
  )
  ncl_model = NCLModel(model)
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(ncl_model, h)  # F(x, r) = x and μ = 1 by default
  stats = envelopt(env_model, verbose = false)
  @test stats.status == :first_order
end

@testitem "solve! with a preallocated EnveloptSolver" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, NLPModels, ProximalOperators, SolverCore
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)  # F(x) = x and μ = 1 by default
  stats_ref = envelopt(env_model, verbose = false)
  @test stats_ref.status == :first_order
  @test get_x0(env_model) == [-1.2; 1.0]  # the initial guess is restored

  # solving again with a preallocated workspace gives the same results
  solver = EnveloptSolver(env_model)
  stats = EnveloptExecutionStats(env_model)
  for _ = 1:2
    solve!(solver, env_model, stats, verbose = false)
    @test stats.status == :first_order
    @test stats.solution == stats_ref.solution
    @test stats.objective == stats_ref.objective
    @test stats.iter == stats_ref.iter
    @test stats.inner_iter == stats_ref.inner_iter
    @test stats.u == stats_ref.u
    @test stats.smooth_obj + stats.nonsmooth_obj == stats.objective
  end
end

@testitem "EnveloptExecutionStats is type stable and forwards to GenericExecutionStats" tags=[
  :solver,
  :unconstrained,
  :madnlp,
] begin
  using ADNLPModels, NLPModels, ProximalOperators, SolverCore, Test
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)
  stats = EnveloptExecutionStats(env_model)
  @test stats.stats isa GenericExecutionStats

  # field access is concretely typed
  get_u(s) = s.u
  get_inner_iter(s) = s.inner_iter
  get_smooth_obj(s) = s.smooth_obj
  get_status(s) = s.status
  get_solution(s) = s.solution
  get_objective(s) = s.objective
  @inferred get_u(stats)
  @inferred get_inner_iter(stats)
  @inferred get_smooth_obj(stats)
  @inferred get_status(stats)
  @inferred get_solution(stats)
  @inferred get_objective(stats)

  # setters and reset! are forwarded
  set_status!(stats, :user)
  @test stats.status == :user
  set_iter!(stats, 3)
  @test stats.iter == 3
  stats.inner_iter = 7
  @test stats.inner_iter == 7
  SolverCore.reset!(stats)
  @test stats.inner_iter == 0
  @test !stats.status_reliable

  # updating the Envelopt-specific fields does not allocate
  function update!(s, fk, hk, uk)
    s.inner_iter += 1
    s.smooth_obj = fk
    s.nonsmooth_obj = hk
    copyto!(s.u, uk)
    return s
  end
  uk = ones(2)
  update!(stats, 1.0, 2.0, uk)
  @test (@allocated update!(stats, 1.0, 2.0, uk)) == 0

  # printing includes the Envelopt-specific fields
  io = IOBuffer()
  print(io, stats)
  @test occursin("inner iterations", String(take!(io)))
end

@testitem "envelopt callback can stop the solver" tags=[:solver, :unconstrained, :madnlp] begin
  using ADNLPModels, ProximalOperators, SolverCore
  model = ADNLPModel(x -> (x[1] - 1.0)^2 + 100 * (x[2] - x[1]^2)^2, [-1.2; 1.0])
  h = NormL1(1.0)
  env_model = EnveloptNLPModel(model, h)
  cb = (nlp, solver, stats) -> (stats.iter ≥ 1 && set_status!(stats, :user))
  stats = envelopt(env_model, verbose = false, callback = cb)
  @test stats.status == :user
  @test stats.iter == 1
end
