# cd(joinpath(Pkg.dir("MultivariateAnomalies"), "docs"))

using Documenter, MultivariateAnomalies

makedocs(modules = [MultivariateAnomalies],
         format   = Documenter.Formats.HTML,
         sitename = "MultivariateAnomalies.jl",
             pages    = Any[
           "Home" =>  "index.md",
           "Manual" => Any[
               "man/FeatureExtraction.md"
             , "man/DetectionAlgorithms.md"
             , "man/AUC.md"
             , "man/DistDensity.md"
             , "man/Scores.md"
             ]
           ]
         )


deploydocs(
  repo = "github.com/milanflach/MultivariateAnomalies.jl.git",
  julia = "release",
  osname = "osx",
  target = "build", # not needed for mkdocs
  make = nothing, # not needed for mkdocs
  deps = nothing #Deps.pip("mkdocs", "python-markdown-math"), # change for mkdocs
)
