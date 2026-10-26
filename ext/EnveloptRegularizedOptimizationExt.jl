module EnveloptRegularizedOptimizationExt

using Envelopt, NLPModels, NLPModelsModifiers, SolverCore
using RegularizedOptimization
using RegularizedProblems

include("RegularizedOptimization/regopt_sub.jl")
include("RegularizedOptimization/r2.jl")
include("RegularizedOptimization/r2n.jl")

end
