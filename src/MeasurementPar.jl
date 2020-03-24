export Measurement

mutable struct MeasurementPar{T}
  w::Vector{Bool}
  Ht::Vector{Vector{T}}
  Σy::Vector{T}
  fim::Array{T,2}
  ifim::Hermitian{T,Array{T,2}}
end

function MeasurementPar(Ht::Array{T,2}, Σy::Vector{T}, fim::Array{T,2}) where T
  numCand = size(Ht,2)
  w = zeros(Bool,numCand)
  ifim = Hermitian(zeros(T,size(fim[1],1),size(fim[1],2)))

  Ht_vec = [Ht[:,j] for j=1:size(Ht,2)]

  return MeasurementPar(w, Ht_vec, Σy, fim, ifim)
end

function MeasurementPar(Ht::Array{T,2}, Σy::Vector{T}, Σx::Vector{T}) where T
  numCand = size(Ht,2)
  w = zeros(Bool,numCand)
  fim = initFIM(reshape(Σx,:,1))[1]
  ifim = Hermitian(zeros(T,size(fim[1],1),size(fim[1],2)))

  Ht_vec = [Ht[:,j] for j=1:size(Ht,2)]

  return MeasurementPar(w, Ht_vec, Σy, fim, ifim)
end

function MeasurementPar(Ht::Array{T,2}, Σy::Vector{T}, Σx::Vector{T},w::Vector{Bool};diagFIM::Bool=false) where T
  numCand = size(Ht,2)
  hx,hy = size(Ht)
  fim = calculateFIM(reshape(Σx,:,1), reshape(Σy,:,1), reshape(Ht,hx,hy,1),w)[1]
  ifim = Hermitian(zeros(T,size(fim[1],1),size(fim[1],2)))
  if diagFIM
    fimDiag = diagm(diag(fim))
    ifim = Hermitian(inv(fimDiag))
  else
    ifim = Hermitian(inv(fim))
  end

  Ht_vec = [Ht[:,j] for j=1:size(Ht,2)]

  return MeasurementPar(w, Ht_vec, Σy, fim, ifim)
end
