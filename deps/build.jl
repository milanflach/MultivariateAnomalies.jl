Pkg.rm("LIBSVM")
Pkg.clone("https://github.com/milanflach/LIBSVM.jl.git")
Pkg.checkout("LIBSVM", "mutating_versions")
Pkg.build("LIBSVM")
