using Documenter, MultivariateAnomalies


          makedocs(
              modules = [MultivariateAnomalies],
              clean   = false,
              format   = :html,
              sitename = "MultivariateAnomalies.jl",
              pages    = Any[ # Compat: `Any` for 0.4 compat
                  "Home" => "index.md",
                  "Manual" => Any[
                    "High Level Functions" => "man/HighLevelFunctions.md",
                    "Anomaly Detection Algorithms" => "man/DetectionAlgorithms.md",
                    "Distance and Densities" =>  "man/DistancesDensity.md",
                    "Postprocessing" => "man/Postprocessing.md",
                    "Preprocessing" =>  "man/Preprocessing.md",
                    "AUC" => "man/AUC.md",
                    "OnlineAlgorithms" => "man/OnlineAlgorithms.md"
                  ]
                  ]
          )

          deploydocs(
              repo   = "github.com/milanflach/MultivariateAnomalies.jl.git",
              #julia  = "0.7",
              #deps   = nothing,
              #make   = nothing,
              target = "build"
          )
