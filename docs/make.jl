# cd(joinpath(Pkg.dir("MultivariateAnomalies"), "docs"))

using Documenter, MultivariateAnomalies

makedocs(modules = [MultivariateAnomalies]
#        , doctest = false
#         , format   = Documenter.Formats.HTML,
#          , sitename = "MultivariateAnomalies.jl",
#          , pages    = Any[
#              "Home" =>  "index.md",
#              "Manual" => Any[
#                  "man/Preprocessing.md"
#                , "man/DetectionAlgorithms.md"
#                , "man/AUC.md"
#                , "man/DistancesDensity.md"
#                , "man/Postprocessing.md"
#                ]
#              ]
          )


deploydocs(
  deps = Deps.pip("mkdocs", "python-markdown-math") #nothing # change to nothing for mkdocs
  , repo = "github.com/milanflach/MultivariateAnomalies.jl.git"
  , julia = "release"
  
  #, osname = "osx"
  #, target = "build" # not needed for mkdocs
  #, make = nothing # not needed for mkdocs
)
