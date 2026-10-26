module EnveloptRegularizedOptimizationExt

using Envelopt, NLPModels, NLPModelsModifiers, SolverCore
using RegularizedOptimization, RegularizedProblems

include("RegularizedOptimization/r2.jl")
include("RegularizedOptimization/r2n.jl")

end
