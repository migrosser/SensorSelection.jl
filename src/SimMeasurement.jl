export SimMeasurement

"""
  data structure describing a measurement

  parameters:
  * `w::Vector{Bool}` -  msk determining which samples are measured
  * `Ht::Array{T,3}`  -  adjoint of measurement matrices. Simultaneous measurements stacked along 3. dim
  * `Σy::Matrix{T}`   -  variances of the measurements. Simultaneous measurements stacked along 2. dim
  * `fim::Array{T,2}` - Array containing the FIM
  * `fim::Hermitian{T,Array{T,2}}` - Array containing the inverse FIM
"""
mutable struct SimMeasurement{T}
  w::Vector{Bool}
  Ht::Array{T,3}
  Σy::Matrix{T}
  fim::Matrix{T}
  ifim::Matrix{T}
  G::Matrix{T}
  tmp::Matrix{T}
end

function SimMeasurement(Ht::Array{T,3}, Σy::Matrix{T}, fim::Array{T,2}; diagFIM::Bool=false) where T
  numCand = size(Ht,2)
  w = zeros(Bool,numCand)
  ifim = zeros(T,size(fim,1),size(fim,2))
  if diagFIM
    fimDiag = Diagonal(diag(fim))
    ifim .= inv(fimDiag)
  else
    ifim .= inv(fim)
  end

  G = zeros(T,size(fim,1),size(Ht,3))
  tmp = zeros(T,size(Ht,3), size(Ht,3))

  return SimMeasurement(w, Ht, Σy, fim, ifim, G, tmp)
end

function SimMeasurement(Ht::Array{T,3}, Σy::Matrix{T}, Σx::Vector{T}; diagFIM::Bool=false) where T
  numCand = size(Ht,2)
  w = zeros(Bool,numCand)
  fim = initFIM(reshape(Σx,:,1))[1]
  ifim = zeros(T,size(fim,1),size(fim,2))
  if diagFIM
    fimDiag = Diagonal(diag(fim))
    ifim .= inv(fimDiag)
  else
    ifim .= inv(fim)
  end

  G = zeros(T,size(fim,1),size(Ht,3))
  tmp = zeros(T,size(Ht,3), size(Ht,3))

  return SimMeasurement(w, Ht, Σy, fim, ifim, G, tmp)
end
