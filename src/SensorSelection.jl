module SensorSelection

using LinearAlgebra, ProgressMeter, Random

include("Utils.jl")
include("FIM.jl")
include("Measurement.jl")
include("OptSamplingFW.jl")
include("OptSamplingBW.jl")

BLAS.set_num_threads(1)

end
