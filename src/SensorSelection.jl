module SensorSelection

using LinearAlgebra, ProgressMeter, Random

include("Utils.jl")
include("FIM.jl")
include("Measurement.jl")
# include("Measurement2.jl")
include("SeqMeasurement.jl")
include("OptSamplingFW.jl")
# include("OptSamplingFW2.jl")
include("OptSamplingBW.jl")
include("OptSamplingBWSeq.jl")

# idea1: approximate IFIMs by diagonal matrices (seems not to work)

# idea2: select samples according to a pseudo-random subset of the IFIM
#        select the ~5% largest rows of the IFIM and ~5% randomly selected rows
end
