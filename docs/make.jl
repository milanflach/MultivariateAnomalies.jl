cd("/Users/mflach/.julia/v0.4/MultivariateAnomalies/docs")

using Documenter, MultivariateAnomalies

makedocs(modules = [MultivariateAnomalies])


deploydocs(
  repo = "github.com/milanflach/MultivariateAnomalies.jl.git"
)
