module EnveloptJSOSolversExt

using Envelopt, NLPModels, SolverCore
using JSOSolvers

include("JSOSolvers/tron.jl")
include("JSOSolvers/trunk.jl")

end