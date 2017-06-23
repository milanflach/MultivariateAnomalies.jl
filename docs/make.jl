# cd(joinpath(Pkg.dir("MultivariateAnomalies"), "docs"))
using Documenter, MultivariateAnomalies

#makedocs(modules = [MultivariateAnomalies]
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
#          )



          makedocs(
              modules = [MultivariateAnomalies],
              clean   = false,
              format   = :html,
              sitename = "MultivariateAnomalies.jl",
              pages    = Any[ # Compat: `Any` for 0.4 compat
                  "Home" => "index.md",
                  "Manual" => Any[
                      "man/AUC.md",
                      "man/DetectionAlgorithms.md",
                      "man/DistancesDensity.md",
                      "man/HighLevelFunctions.md",
                      "man/Postprocessing.md",
                      "man/Preprocessing.md"
                  ]
                  ]
          )

          deploydocs(
              #deps   = Deps.pip("mkdocs", "python-markdown-math"),
              repo   = "github.com/milanflach/MultivariateAnomalies.jl.git",
              julia  = "0.5",
              deps   = nothing,
              make   = nothing,
              target = "build"
          )


#deploydocs(
#  deps = Deps.pip("mkdocs", "python-markdown-math") #nothing # change to nothing for mkdocs
#  , repo = "github.com/milanflach/MultivariateAnomalies.jl.git"
#  , julia = "0.5"
  #, osname = "osx"
  #, target = "build" # not needed for mkdocs
  #, make = nothing # not needed for mkdocs
#)
