export optSamplingBW!, removeSample!

"""
    `optSampling!(m::Measurement, numSamples::Int64)`

select sensors such that the estimation of the parameter-vector `x` has minimum uncertainty.

# Arguments:
* `m::Measurement`            - Measurement object
* `numSamples::Vector{Int64}` - number of samples
"""
function optSamplingBW!(m::Measurement{T}, numSamples::Int64, Σx::Matrix{T}, numSimSamp::Int64, numCand::Int64, batch::Int64=numCand) where T
  @info "optSamplingBW"
  # remove samples
  cand = collect(1:size(m.Ht,2))
  tmp = zeros(T, size(m.fim[1],1), batch)
  tmp_ifim = zeros(T, size(m.ifim[1]))
  δj = zeros(T,length(cand))
  @showprogress 1 "Select Sensors..." for i=1:div(length(cand)-numSamples,numSimSamp)
    δj .= 0.0
    removeSamples!(m,cand,tmp,tmp_ifim,δj,numSimSamp,numCand)
  end
end

"""
  `removeSample!(w::Vector{Bool}, cand::Vector{Int64}, fim::Array{Float64,3}, Ht::Array{Float64,3},r::Matrix{Float64})`

remove the sample from `cand` which minimizes maximizes the trace of the FIM (i.e. minimizes uncertainty).

# Arguments:
* `m::Measurement`         - Measurement object
* `cand::Vector{Int64}`    - candidate samples
"""
function removeSamples!(m::Measurement{T}, cand::Vector{Int64}
                      , tmp::Matrix{T}, tmp_ifim::Matrix{T}
                      , δj::Vector{T}, numSimSamp::Int64, numCand::Int64) where T
  numComp = length(m.fim)

  # find index which yields the least increase in average variance (trace of inverse FIM)
  incIFIM!(δj,m,cand,tmp)

  # find the smallest elements and iteratively remove the best candidates
  p = sortperm( real.(δj) )[1:numCand]
  cand2 = copy( cand[p] )
  removeSample!(m, cand2, 1, tmp[:,1], tmp_ifim)
  for i=1:numSimSamp-1
    # re-estimate the changes in uncertainty
    incIFIM!(δj, m, cand2, tmp)
    # remove least informative sample
    δfim, idx = findmin(real.(δj[1:length(cand2)]))
    removeSample!(m, cand2, idx, tmp[:,1], tmp_ifim)
  end

  return nothing
end

"""
  calculate the change in the trace of inverse FIM when removing a sample from the candidate set
"""
function incIFIM!(δj::Vector{T}, m::Measurement{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T
  batch = size(tmp,2)
  nprod = div(length(cand),batch)
  δj .= 0.0
  for j=1:nprod
    x0 = (j-1)*batch+1
    x1 = j*batch
    δjb = @view δj[x0:x1]
    incIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[x0:x1,:], tmp)
  end
  if nprod*batch+1 <= length(cand)
    x0 = nprod*batch+1
    x1 = length(cand)
    δjb = @view δj[x0:x1]
    incIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[x0:x1,:], tmp[:,1:x1-x0+1])
  end
end

function incIFIMBatch!(δ::U, ifim::Vector{Hermitian{T,Array{T,2}}}, Ht::Array{T,3}, Σy::Matrix{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T
  # add contributions for all components
  for k=1:length(ifim)
    mul!(tmp, ifim[k], Ht[:,:,k])
    δ .+= ddot(tmp)./( Σy[:,k] .- ddot(Ht[:,:,k],tmp) )
  end
end

function removeSample!(m::Measurement{T}, cand::Vector{Int64},idx::Int64,tmp::Vector{T},tmp_ifim::Matrix{T}) where T
  numComp = length(m.fim)
  # update samplig scheme
  m.w[cand[idx]] = 0
  # update FIM
  for k=1:numComp
    m.fim[k] .-= 1.0/m.Σy[cand[idx],k] .* m.Ht[:,cand[idx],k]*adjoint(m.Ht[:,cand[idx],k])
  end
  # update IFIM
  for k=1:numComp
    mul!(tmp, m.ifim[k], m.Ht[:,cand[idx],k])
    denom = m.Σy[cand[idx],k]-dot( m.Ht[:,cand[idx],k],tmp )
    tmp_ifim .= tmp*adjoint(tmp) ./ denom
    m.ifim[k] += Hermitian(0.5*(tmp_ifim.+adjoint(tmp_ifim)))
  end
  # remove sample from candidate set
  deleteat!(cand,idx)
end
