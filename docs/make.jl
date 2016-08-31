# cd(joinpath(Pkg.dir("MultivariateAnomalies"), "docs"))

using Documenter, MultivariateAnomalies

makedocs(modules = [MultivariateAnomalies])


deploydocs(
#  deps   = Deps.pip("mkdocs", "python-markdown-math"),
  repo = "github.com/milanflach/MultivariateAnomalies.jl.git",
#  julia = "0.4"
)
