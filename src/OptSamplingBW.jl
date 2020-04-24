export optSamplingBW!

"""
    `optSampling!(m::Measurement, numSamples::Int64)`

select sensors such that the estimation of the parameter-vector `x` has minimum uncertainty.

# Arguments:
* `m::Measurement`            - Measurement object
* `numSamples::Vector{Int64}` - number of samples
"""
function optSamplingBW!(m::Measurement{T}, numSamples::Int64, numSimSamp::Int64, numCand::Int64, batch::Int64=numCand; numericInv::Bool=false) where T
  @info "optSamplingBW"
  # candidate samples to be removed
  cand = collect(1:size(m.Ht,2))
  # preallocate temporary arrays
  tmp = zeros(T, size(m.fim[1],1), batch)
  δj = zeros(T,length(cand))
  # remove batches of samples
  numBatches = div(length(cand)-numSamples,numSimSamp)
  wlog = Int64[]
  @showprogress 1 "Select Sensors..." for i=1:numBatches
    δj .= 0.0
    removeSamples!(m,cand,tmp,δj,numSimSamp,numCand,wlog,numericInv=numericInv)
  end
  # remove samples from m.w
  m.w[wlog] .= 0
  return wlog
end

"""
  `removeSample!(w::Vector{Bool}, cand::Vector{Int64}, fim::Array{Float64,3}, Ht::Array{Float64,3},r::Matrix{Float64})`

remove the sample from `cand` which minimizes the trace of the inverse FIM
(i.e. minimizes uncertainty).

# Arguments:
* `m::Measurement`         - Measurement object
* `cand::Vector{Int64}`    - candidate samples
"""
function removeSamples!(m::Measurement{T}, cand::Vector{Int64}
                      , tmp::Matrix{T}
                      , δj::Vector{T}, numSamp::Int64, numCand::Int64,wlog::Vector{Int64}; numericInv::Bool=false) where T
  numComp = length(m.fim)

  # find index which yields the least increase in average variance (trace of inverse FIM)
  incIFIM!(δj,m,cand,tmp)

  # find the smallest elements and iteratively remove the best candidates
  p = sortperm( real.(δj[1:length(cand)]) )[1:numCand]
  removeSample!(m, cand, 1, p, tmp[:,1], wlog, numericInv=numericInv)
  for i=1:numSamp-1
    # re-estimate the changes in uncertainty
    incIFIM!(δj, m, cand[p], tmp)
    # remove least informative sample
    δfim, idx = findmin(real.(δj[1:length(p)]))
    removeSample!(m, cand, idx, p, tmp[:,1], wlog, numericInv=numericInv)
  end

  return nothing
end

"""
  `incIFIM!(δj::Vector{T}, m::Measurement{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T`

calculate the change in the trace of inverse FIM when removing a sample from the candidate set.
The results are stored in the first entries of `δj`
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

function incIFIMBatch!(δ::U, ifim::Vector{Matrix{T}}, Ht::Array{T,3}, Σy::Matrix{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T
  # add contributions for all components
  for k=1:length(ifim)
    mul!(tmp, ifim[k], Ht[:,:,k])
    δ .+= ddot(tmp)./( Σy[:,k] .- ddot(Ht[:,:,k],tmp) )
  end
end

function removeSample!(m::Measurement{T}, cand::Vector{Int64},idx::Int64
                      ,p::Vector{Int64},tmp::Vector{T}
                      ,wlog::Vector{Int64}; numericInv::Bool=false) where T
  numComp = length(m.fim)
  p_i = p[idx]
  # update samplig log
  push!(wlog,cand[p_i])
  # update FIM
  for k=1:numComp
    m.fim[k] .-= 1.0/m.Σy[cand[p_i],k] .* m.Ht[:,cand[p_i],k]*adjoint(m.Ht[:,cand[p_i],k])
  end
  # update IFIM
  if numericInv
    for k=1:numComp
        m.ifim[k] = inv(m.fim[k])
    end
  else
    for k=1:numComp
      mul!(tmp, m.ifim[k], m.Ht[:,cand[p_i],k])
      denom = m.Σy[cand[p_i],k]-dot( m.Ht[:,cand[p_i],k],tmp )
      m.ifim[k] .+= tmp*adjoint(tmp) ./ denom
    end
  end
  # remove sample from candidate set
  deleteat!(cand,p_i)
  p[findall(x->x>p_i,p)] .-=1
  deleteat!(p,idx)
end
