export Measurement

"""
  data structure describing a measurement

  parameters:
  * `w::Vector{Bool}` -  msk determining which samples are measured
  * `Ht::Array{T,3}`  -  adjoint of measurement matrices. Components stacked along 3. dim
  * `Σy::Matrix{T}`   -  variances of the measurements. Components stacked along 2. dim
  * `fim::Vector{Array{T,2}}` - Array containing the FIM for each measurement component
  * `fim::Vector{Hermitian{T,Array{T,2}}}` - Array containing the IFIM for each measurement component
"""
mutable struct Measurement{T}
  w::Vector{Bool}
  Ht::Array{T,3}
  Σy::Matrix{T}
  fim::Vector{Matrix{T}}
  ifim::Vector{Matrix{T}}
end

function Measurement(Ht::Array{T,3}, Σy::Matrix{T}, fim::Array{T,3}) where T
  numCand = size(Ht,2)
  w = zeros(Bool,numCand)
  ifim = [ zeros(T,size(fim[1],1),size(fim[1],2)) for k=1:length(fim) ]

  return Measurement(w, Ht, Σy, fim, ifim)
end

function Measurement(Ht::Array{T,3}, Σy::Matrix{T}, Σx::Matrix{T}) where T
  numCand = size(Ht,2)
  w = zeros(Bool,numCand)
  fim = initFIM(Σx)
  ifim = [ zeros(T,size(fim[1],1),size(fim[1],2)) for k=1:length(fim) ]

  return Measurement(w, Ht, Σy, fim, ifim)
end

function Measurement(Ht::Array{T,3}, Σy::Matrix{T}, Φ::Array{T,3}, Σx::Matrix{T}) where T
  numCand = size(Ht,2)
  w = zeros(Bool,numCand)
  fim = initFIM(Φ,Σx)
  ifim = [ zeros(T,size(fim[1],1),size(fim[1],2)) for k=1:length(fim) ]

  return Measurement(w, Ht, Σy, fim, ifim)
end

function Measurement(Ht::Array{T,3}, Σy::Matrix{T}, Σx::Matrix{T},w::Vector{Bool};diagFIM::Bool=false) where T
  numCand = size(Ht,2)
  fim = calculateFIM(Σx,Σy,Ht,w)
  ifim = [ zeros(T,size(fim[1],1),size(fim[1],2)) for k=1:length(fim) ]
  for k=1:length(fim)
    if diagFIM
      fimDiag = diagm(diag(fim[k]))
      ifim[k] = inv(fimDiag)
    else
      ifim[k] = inv(fim[k])
    end
  end

  return Measurement(w, Ht, Σy, fim, ifim)
end

function Measurement(Ht::Array{T,3}, Σy::Matrix{T}, Φ::Array{T,3}, Σx::Matrix{T}, w::Vector{Bool};diagFIM::Bool=false) where T
  numCand = size(Ht,2)
  fim = calculateFIM(Σx,Φ,Σy,Ht,w)
  ifim = [ zeros(T,size(fim[1],1),size(fim[1],2)) for k=1:length(fim) ]
  for k=1:length(fim)
    if diagFIM
      fimDiag = diagm(diag(fim[k]))
      ifim[k] = inv(fimDiag)
    else
      ifim[k] = inv(fim[k])
    end
  end

  return Measurement(w, Ht, Σy, fim, ifim)
end
