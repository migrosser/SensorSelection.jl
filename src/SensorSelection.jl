module SensorSelection

using LinearAlgebra, ProgressMeter, Random

include("Utils.jl")
include("FIM.jl")
include("Measurement.jl")
include("SeqMeasurement.jl")
include("OptSamplingFW.jl")
include("OptSamplingBW.jl")
include("OptSamplingBWSeq.jl")
end
