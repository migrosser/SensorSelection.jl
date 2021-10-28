export SimExperiment

"""
  data structure describing a measurement

  parameters:
  * `w::Vector{Bool}` -  msk determining which samples are measured
  * `Ht::Array{T,3}`  -  adjoint of measurement matrices. Simultaneous measurements stacked along 3. dim
  * `Σy::Matrix{T}`   -  variances of the measurements. Simultaneous measurements stacked along 2. dim
  * `fim::Array{T,2}` - Array containing the FIM
  * `fim::Hermitian{T,Array{T,2}}` - Array containing the inverse FIM
"""
mutable struct SimExperiment{T}
  w::Vector{Bool}
  Ht::Array{T,3}
  Σy::Matrix{T}
  fim::Matrix{T}
  ifim::Matrix{T}
  G::Matrix{T}
  tmp::Matrix{T}
end

function SimExperiment(Ht::Array{T,3}, Σy::Matrix{T}, fim::Array{T,2}; diagFIM::Bool=false) where T
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

  return SimExperiment(w, Ht, Σy, fim, ifim, G, tmp)
end

function SimExperiment(Ht::Array{T,3}, Σy::Matrix{T}, Σx::Vector{T}; diagFIM::Bool=false) where T
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

  return SimExperiment(w, Ht, Σy, fim, ifim, G, tmp)
end

function SimExperiment(Ht::Array{T,3}, Σy::Matrix{T}, Σx::Vector{T}, w::Vector{Bool}; diagFIM::Bool=false) where T
  numCand = size(Ht,2)
  fim = calculateFIMSim(Σx,Σy,Ht,w)
  ifim = zeros(T,size(fim,1),size(fim,2))
  if diagFIM
    fimDiag = Diagonal(diag(fim))
    ifim .= inv(fimDiag)
  else
    ifim .= inv(fim)
  end

  G = zeros(T,size(fim,1),size(Ht,3))
  tmp = zeros(T,size(Ht,3), size(Ht,3))

  return SimExperiment(w, Ht, Σy, fim, ifim, G, tmp)
end