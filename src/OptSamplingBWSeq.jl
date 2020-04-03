export optSamplingBWSeq!

"""
    `optSampling!(m::Measurement, numSamples::Int64)`

select sensors such that the estimation of the parameter-vector `x` has minimum uncertainty.

# Arguments:
* `m::Measurement`            - Measurement object
* `numSamples::Vector{Int64}` - number of samples
"""
function optSamplingBW!(m::SeqMeasurement{T}, numSamples::Int64, numSimSamp::Int64, numCand::Int64, batch::Int64=numCand) where T
  @info "optSamplingBW"
  # candidate samples to be removed
  cand = [collect(1:size(m.Ht,2)) for k=1:size(m.w,2)]
  # preallocate temporary arrays
  tmp = zeros(T, size(m.fim,1), batch)
  tmp_ifim = zeros(T, size(m.ifim))
  δj = zeros(T,length(cand[1]))
  # remove batches of samples
  numBatches = div(length(cand[1])-numSamples,numSimSamp)
  numComp = size(m.w,2)
  @showprogress 1 "Select Sensors..." for i=1:numBatches
    δj .= 0.0
    for k=1:numComp
      removeSamples!(m,k,cand[k],tmp,tmp_ifim,δj,numSimSamp,numCand)
    end
  end
end

"""
  `removeSample!(w::Vector{Bool}, k::Int64, cand::Vector{Int64}, fim::Array{Float64,3}, Ht::Array{Float64,3},r::Matrix{Float64})`

remove `numSamp` samples from the k-th component in `cand`.
Samples are chosen such that the trace of the inverse FIM is minimized
(i.e. minimizes uncertainty).

# Arguments:
* `m::Measurement`         - Measurement object
* `cand::Vector{Int64}`    - candidate samples
"""
function removeSamples!(m::SeqMeasurement{T}, k::Int64, cand::Vector{Int64}
                      , tmp::Matrix{T}, tmp_ifim::Matrix{T}
                      , δj::Vector{T}, numSamp::Int64, numCand::Int64) where T
  numComp = length(m.fim)

  # find index which yields the least increase in average variance (trace of inverse FIM)
  incIFIM!(k, δj, m, cand, tmp)

  # find the smallest elements and iteratively remove the best candidates
  p = sortperm( real.(δj[1:length(cand)]) )[1:numCand]
  removeSample!(m, k, cand, 1, p, tmp[:,1], tmp_ifim)
  for i=1:numSamp-1
    # re-estimate the changes in uncertainty
    incIFIM!(k, δj, m, cand[p], tmp)
    # remove least informative sample
    δfim, idx = findmin(real.(δj[1:length(p)]))
    removeSample!(m, k, cand, idx, p, tmp[:,1], tmp_ifim)
  end

  return nothing
end

"""
  `incIFIM!(δj::Vector{T}, m::Measurement{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T`

calculate the change in the trace of inverse FIM when removing a sample from the candidate set.
The results are stored in the first entries of `δj`
"""
function incIFIM!(k::Int64, δj::Vector{T}, m::SeqMeasurement{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T
  batch = size(tmp,2)
  nprod = div(length(cand),batch)
  δj .= 0.0
  for j=1:nprod
    x0 = (j-1)*batch+1
    x1 = j*batch
    δjb = @view δj[x0:x1]
    incIFIMBatchSeq!(δjb, m.ifim, m.Ht[:,cand[x0:x1],k], m.Σy[x0:x1,k], tmp)
  end
  if nprod*batch+1 <= length(cand)
    x0 = nprod*batch+1
    x1 = length(cand)
    δjb = @view δj[x0:x1]
    incIFIMBatchSeq!(δjb, m.ifim, m.Ht[:,cand[x0:x1],k], m.Σy[x0:x1,k], tmp[:,1:x1-x0+1])
  end
end

function incIFIMBatchSeq!(δ::U, ifim::Hermitian{T,Array{T,2}}, Ht::Matrix{T}, Σy::Vector{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T
  # add contributions for all components
  mul!(tmp, ifim, Ht)
  δ .+= ddot(tmp)./( Σy .- ddot(Ht,tmp) )
end

function removeSample!(m::SeqMeasurement{T}, k::Int64, cand::Vector{Int64},idx::Int64,p::Vector{Int64},tmp::Vector{T},tmp_ifim::Matrix{T}) where T
  p_i = p[idx]
  # update samplig scheme
  m.w[cand[p_i],k] = 0
  # update FIM
  m.fim .-= 1.0/m.Σy[cand[p_i],k] .* m.Ht[:,cand[p_i],k]*adjoint(m.Ht[:,cand[p_i],k])
  # update IFIM
  mul!(tmp, m.ifim, m.Ht[:,cand[p_i],k])
  denom = m.Σy[cand[p_i],k]-dot( m.Ht[:,cand[p_i],k],tmp )
  tmp_ifim .= tmp*adjoint(tmp) ./ denom
  m.ifim += Hermitian(0.5*(tmp_ifim.+adjoint(tmp_ifim)))
  # remove sample from candidate set
  deleteat!(cand,p_i)
  p[findall(x->x>p_i,p)] .-=1
  deleteat!(p,idx)
end
