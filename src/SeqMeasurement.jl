export SeqMeasurement

"""
  data structure describing a measurement

  parameters:
  * `w::Vector{Bool}` -  msk determining which samples are measured
  * `Ht::Array{T,3}`  -  adjoint of measurement matrices. Components stacked along 3. dim
  * `Σy::Matrix{T}`   -  variances of the measurements. Components stacked along 2. dim
  * `fim::Vector{Array{T,2}}` - Array containing the FIM for each measurement component
  * `fim::Vector{Hermitian{T,Array{T,2}}}` - Array containing the IFIM for each measurement component
"""
mutable struct SeqMeasurement{T}
  w::Matrix{Bool}
  Ht::Array{T,3}
  Σy::Matrix{T}
  fim::Array{T,2}
  ifim::Matrix{T}
end

function SeqMeasurement(Ht::Array{T,3}, Σy::Matrix{T}, Σx::Matrix{T}) where T
  numPar, numCand, numMeas = size(Ht)
  w = zeros(Bool,numCand, numMeas)
  fim = initFIM(Σx)
  ifim = zeros(T,size(fim))

  return SeqMeasurement(w, Ht, Σy, fim, ifim)
end

function SeqMeasurement(Ht::Array{T,3}, Σy::Matrix{T}, Σx::Vector{T},w::Matrix{Bool}) where T
  numPar, numCand, numMeas = size(Ht)
  fim = calculateFIM(reshape(Σx,:,1), reshape(Σy,:,1),reshape(Ht,numPar,numCand*numMeas,1),vec(w))[1]
  ifim= inv(fim)

  return SeqMeasurement(w, Ht, Σy, fim, ifim)
end
