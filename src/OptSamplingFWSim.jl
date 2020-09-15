export optSamplingFW!, addSample!

"""
    `optSampling!(m::Measurement, numSamples::Int64)`

select sensors such that the estimation of the parameter-vector `x` has minimum uncertainty.

# Arguments:
* `m::Measurement`            - Measurement object
* `numSamples::Vector{Int64}` - number of samples
"""
function optSamplingFW!(m::SimMeasurement{T}, numSamples::Int64, numSimSamp::Int64
                        , numCand::Int64, batch::Int64=numCand
                        ; measCost::Vector{Float64}=Float64[]) where T
  cand = collect(1:size(m.Ht,2))
  G_tmp = zeros(T,size(m.fim,1),batch,size(m.Ht,3))
  GhG_tmp = zeros(T,size(m.Ht,3),size(m.Ht,3))
  δj = zeros(T,length(cand))
  # remove batches of samples
  numBatches = div(numSamples,numSimSamp)
  wlog = Int64[]
  @showprogress 1 "Select Sensors..." for i=1:numBatches
    nSamp = min(length(cand),numSimSamp)
    nCand = min(length(cand),numCand)
    addSamples!(m,cand,G_tmp,GhG_tmp,δj,nSamp,nCand,wlog,measCost=measCost)
  end
  m.w[wlog] .= 1
  return wlog
end

"""
  `addSamples!(w::Vector{Bool}, cand::Vector{Int64}, fim::Array{Float64,3}, Ht::Array{Float64,3},r::Matrix{Float64})`

adds the sample from `cand` which minimizes maximizes the trace of the FIM (i.e. minimizes uncertainty).

# Arguments:
* `m::Measurement`         - Measurement object
* `cand::Vector{Int64}`    - candidate samples
"""
function addSamples!(m::SimMeasurement{T}, cand::Vector{Int64}
                      , G_tmp::Array{T,3}, GhG_tmp::Matrix{T},δj::Vector{T}
                      , numSamp::Int64, numCand::Int64,wlog::Vector{Int64}
                      ; measCost::Vector{Float64}=Float64[]) where T

  # find index which yields the largest decrease in average variance (trace of inverse FIM)
  redIFIM!(δj,m,cand,G_tmp,GhG_tmp)
  # add regularization (measCost)
  if !isempty(measCost)
    δj[1:length(cand)] .-= measCost[cand]
  end

  # find the largest elements and iteratively add the best candidates
  p = sortperm( real.(δj[1:length(cand)]), rev=true )[1:numCand]
  addSample!(m, cand, 1, p, wlog)
  for i=1:numSamp-1
    # re-estimate the changes in uncertainty
    redIFIM!(δj, m, cand[p],G_tmp,GhG_tmp)
    if !isempty(measCost)
      δj[1:length(p)] .-= measCost[cand[p]]
    end
    # add most informative sample
    δfim, idx = findmax(real.(δj[1:length(p)]))
    addSample!(m, cand, idx, p, wlog)
  end

  return nothing
end

"""
  `redIFIM!(δj::Vector{T}, m::Measurement{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T`

calculate the change in the trace of inverse FIM when adding a sample from the candidate set.
The results are stored in the first entries of `δj`
"""
function redIFIM!(δj::Vector{T}, m::SimMeasurement{T}, cand::Vector{Int64}, G_tmp::Array{T,3},GhG_tmp) where T
  batch = size(G_tmp,2)
  nprod = div(length(cand),batch)
  δj .= 0.0
  for j=1:nprod
    x0 = (j-1)*batch+1
    x1 = j*batch
    δjb = @view δj[x0:x1]
    redIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[cand[x0:x1],:], G_tmp, GhG_tmp, m.tmp)
  end
  if nprod*batch+1 <= length(cand)
    x0 = nprod*batch+1
    x1 = length(cand)
    δjb = @view δj[x0:x1]
    redIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[cand[x0:x1],:], G_tmp[:,1:x1-x0+1,:], GhG_tmp, m.tmp)
  end
end

"""
  `redIFIM!(δj::Vector{T}, m::Measurement{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T`

calculate the change in the trace of inverse FIM when adding a sample from the candidate set.
The results are stored in the first entries of `δj`
"""
function redIFIMBatch!(δj::U, ifim::Matrix{T}, Ht::Array{T,3}, Σy::Matrix{T}, G::Array{T,3}, GhG::Matrix{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T
  # precompute G for all candidates
  for c=1:size(Ht,3) # loop over all channels
    G[:,:,c] .= ifim*Ht[:,:,c]
  end
  # compute change in CRB for all candidates
  for j=1:size(Ht,2) # loop over all candidates
    # compute inner matrix
    tmp .= inv( Diagonal(Σy[j,:]) .+ adjoint(Ht[:,j,:])*G[:,j,:] )
    δj[j] = trp(G[:,j,:],tmp,GhG)
  end
end

function addSample!(m::SimMeasurement{T}, cand::Vector{Int64},idx::Int64
                    ,p::Vector{Int64},wlog::Vector{Int64}) where T
  p_i = p[idx]
  # update samplig log
  push!(wlog,cand[p_i])
  # update IFIM
  m.G .= m.ifim*m.Ht[:,cand[p_i],:]
  m.tmp .= inv( Diagonal(m.Σy[cand[p_i],:]) .+ adjoint(m.Ht[:,cand[p_i],:])*m.G )
  m.ifim .-= m.G*m.tmp*adjoint(m.G)
  # remove sample from candidate set
  deleteat!(cand,p_i)
  p[findall(x->x>p_i,p)] .-=1
  deleteat!(p,idx)
end
