export optSamplingFW!, removeSample!

"""
    `optSamplingBW!(m::SimExperiment, numSamples::Int64, numSimSamp::Int64)`

Iteratively removes the least informative measurements from `m` until only `numSamples`
measurements remain. This method assumes `m` to be an experiment where all `measurements` in `m.Ht` are performed.

# Arguments:
* `m::SimExperiment`             - Experiment object
* `numSamples::Vector{Int64}` - number of samples
* `numSimSamp::Int64`         - number of samples to remove per batch
* `numCand::Int64`            - number candidate measurements to consider per batch
"""
function optSamplingBW!(m::SimExperiment{T}, numSamples::Int64, numSimSamp::Int64
                        , numCand::Int64, batch::Int64=numCand
                        ; measCost::Vector{Float64}=Float64[]) where T

  cand = collect(1:size(m.Ht,2))
  G_tmp = zeros(T,size(m.fim,1),batch,size(m.Ht,3))
  GhG_tmp = zeros(T,size(m.Ht,3),size(m.Ht,3))
  δj = zeros(T,length(cand))
  # remove batches of samples
  numBatches = div(length(cand)-numSamples,numSimSamp)
  wlog = Int64[]
  @showprogress 1 "Select Sensors..." for i=1:numBatches
    nSamp = min(length(cand),numSimSamp)
    nCand = min(length(cand),numCand)
    removeSamples!(m,cand,G_tmp,GhG_tmp,δj,nSamp,nCand,wlog,measCost=measCost)
  end
  m.w[wlog] .= 0
  return wlog
end

"""
  `removeSamples!(m::SimExperiment{T}, cand::Vector{Int64},
                  G_tmp::Array{T,3}, GhG_tmp::Matrix{T}, 
                  δj::Vector{T}, numSamp::Int64, numCand::Int64,
                  wlog::Vector{Int64}; numericInv::Bool=false) where T`

remove `numSamp` samples the experiment `m` such that the trace of the inverse FIM is minimized
(i.e. minimizes uncertainty).

# Arguments:
* `m::Experiment`         - Experiment object
* `cand::Vector{Int64}`   - candidate samples
* `G_tmp::Matrix{T}`      - temporary matrix for intermediate calculations
* `GhG_tmp::Matrix{T}`    - temporary matrix for intermediate calculations
* `δj::Vector{T}`         - vector to store the change in uncertainty for each candidate
* `numSamp::Int64`        - number of measurements to remove
* `numCand::Int64`        - number of candidate measurements to consider per batch
* `wlog::Vector{Int64}`   - vector to keep track which measurements are performed
"""
function removeSamples!(m::SimExperiment{T}, cand::Vector{Int64}
                      , G_tmp::Array{T,3}, GhG_tmp::Matrix{T},δj::Vector{T}
                      , numSamp::Int64, numCand::Int64,wlog::Vector{Int64}
                      ; measCost::Vector{Float64}=Float64[]) where T

  # find index which yields the largest decrease in average variance (trace of inverse FIM)
  incIFIM!(δj,m,cand,G_tmp,GhG_tmp)
  # add regularization (measCost)
  if !isempty(measCost)
    δj[1:length(cand)] .-= measCost[cand]
  end

  # find the largest elements and iteratively add the best candidates
  p = sortperm( real.(δj[1:length(cand)]))[1:numCand]
  removeSample!(m, cand, 1, p, wlog)
  for i=1:numSamp-1
    # re-estimate the changes in uncertainty
    incIFIM!(δj, m, cand[p],G_tmp,GhG_tmp)
    if !isempty(measCost)
      δj[1:length(p)] .-= measCost[cand[p]]
    end
    # add most informative sample
    δfim, idx = findmin(real.(δj[1:length(p)]))
    time = @elapsed removeSample!(m, cand, idx, p, wlog)
  end

  return nothing
end

"""
  `incIFIM!(δj::Vector{T}, m::SimExperiment{T}, cand::Vector{Int64}, G_tmp::Array{T,3}, GhG_tmp) where T`

calculate the change in the trace of inverse FIM when removing a sample from the candidate set.
The results are stored in the first entries of `δj`
"""
function incIFIM!(δj::Vector{T}, m::SimExperiment{T}, cand::Vector{Int64}, G_tmp::Array{T,3},GhG_tmp) where T
  batch = size(G_tmp,2)
  nprod = div(length(cand),batch)
  δj .= 0.0
  for j=1:nprod
    x0 = (j-1)*batch+1
    x1 = j*batch
    δjb = @view δj[x0:x1]
    incIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[cand[x0:x1],:], G_tmp, GhG_tmp, m.tmp)
  end
  if nprod*batch+1 <= length(cand)
    x0 = nprod*batch+1
    x1 = length(cand)
    δjb = @view δj[x0:x1]
    incIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[cand[x0:x1],:], G_tmp[:,1:x1-x0+1,:], GhG_tmp, m.tmp)
  end
end

"""
  `incIFIMBatch!(δ::U, ifim::Vector{Matrix{T}}, Ht::Array{T,3}, Σy::Matrix{T}, G::Array{T,3}, GhG::Matrix{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T`

computes the increase in the trace of the inverse FIM when removing a measurement from the experiment
""" 
function incIFIMBatch!(δj::U, ifim::Matrix{T}, Ht::Array{T,3}, Σy::Matrix{T}, G::Array{T,3}, GhG::Matrix{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T
  # precompute G for all candidates
  for c=1:size(Ht,3) # loop over all channels
    G[:,:,c] .= ifim*Ht[:,:,c]
  end
  # compute change in CRB for all candidates
  for j=1:size(Ht,2) # loop over all candidates
    # compute inner matrix
    tmp .= inv( Diagonal(Σy[j,:]) .- adjoint(Ht[:,j,:])*G[:,j,:] )
    δj[j] = trp(G[:,j,:],tmp,GhG)
  end
end

"""
  `removeSample!(m::SimExperiment{T}, cand::Vector{Int64},idx::Int64 , p::Vector{Int64},
                ,wlog::Vector{Int64}) where T`

removes the measurement with idx `idx` from the experiment `m`.
"""
function removeSample!(m::SimExperiment{T}, cand::Vector{Int64},idx::Int64
                    ,p::Vector{Int64},wlog::Vector{Int64}) where T
  p_i = p[idx]
  # update samplig log
  push!(wlog,cand[p_i])
  # update IFIM
  m.G .= m.ifim*m.Ht[:,cand[p_i],:]
  m.tmp .= inv( Diagonal(m.Σy[cand[p_i],:]) .- adjoint(m.Ht[:,cand[p_i],:])*m.G )
  m.ifim .+= m.G*m.tmp*adjoint(m.G)
  # remove sample from candidate set
  deleteat!(cand,p_i)
  p[findall(x->x>p_i,p)] .-=1
  deleteat!(p,idx)
end
