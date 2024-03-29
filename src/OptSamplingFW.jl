export optSamplingFW!, addSample!

"""
    `optSamplingFW!(m::Experiment{T}, numSamples::Int64, numSimSamp::Int64, numCand::Int64, batch::Int64=numCand; numericInv::Bool=false) where T`

Iteratively adds the most informative measurements from `m` until `numSamples`
measurements are performed.

# Arguments:
* `m::Experiment`             - Experiment object
* `numSamples::Vector{Int64}` - number of samples
* `numSimSamp::Int64`         - number of samples to remove per batch
* `numCand::Int64`            - number candidate measurements to consider per batch
"""
function optSamplingFW!(m::Experiment{T}, numSamples::Int64, numSimSamp::Int64
                        , numCand::Int64, batch::Int64=numCand
                        ; numericInv::Bool=false, measCost::Vector{Float64}=Float64[]) where T
  cand = collect(1:size(m.Ht,2))
  tmp = zeros(T, size(m.fim[1],1), batch)
  δj = zeros(T,length(cand))
  # remove batches of samples
  numBatches = div(numSamples,numSimSamp)
  wlog = Int64[]
  @showprogress 1 "Select Sensors..." for i=1:numBatches
    nSamp = min(length(cand),numSimSamp)
    nCand = min(length(cand),numCand)
    # addSamples!(m,cand,tmp,δj,nSamp,nCand,wlog)
    addSamples!(m,cand,tmp,δj,nSamp,nCand,wlog,numericInv=numericInv,measCost=measCost)
  end
  m.w[wlog] .= 1
  return wlog
end

"""
  `addSamples!(m::Experiment{T}, cand::Vector{Int64},
               tmp::Matrix{T}, δj::Vector{T}, numSamp::Int64, 
               numCand::Int64,wlog::Vector{Int64}; numericInv::Bool=false) where T`

adds the sample from `cand` which minimizes the trace of the FIM (i.e. minimizes uncertainty).

# Arguments:
* `m::Experiment`         - Experiment object
* `cand::Vector{Int64}`   - candidate samples
* `tmp::Matrix{T}`        - temporary matrix for intermediate calculations
* `δj::Vector{T}`         - vector to store the change in uncertainty for each candidate
* `numSamp::Int64`        - number of measurements to remove
* `numCand::Int64`        - number of candidate measurements to consider per batch
* (`numericInv::Bool=false`) - if true the FIM is inverted numerically instead of using
                               the Woodbury Matrix inversion lemma
"""
function addSamples!(m::Experiment{T}, cand::Vector{Int64}
                      , tmp::Matrix{T}, δj::Vector{T}
                      , numSamp::Int64, numCand::Int64,wlog::Vector{Int64}
                      ; numericInv::Bool=false, measCost::Vector{Float64}=Float64[]) where T
  numComp = length(m.fim)

  # find index which yields the largest decrease in average variance (trace of inverse FIM)
  redIFIM!(δj,m,cand,tmp)
  # add regularization (measCost)
  if !isempty(measCost)
    δj[1:length(cand)] .-= measCost[cand]
  end

  # find the largest elements and iteratively add the best candidates
  p = sortperm( real.(δj[1:length(cand)]), rev=true )[1:numCand]
  addSample!(m, cand, 1, p, tmp[:,1], wlog, numericInv=numericInv)
  for i=1:numSamp-1
    # re-estimate the changes in uncertainty
    redIFIM!(δj, m, cand[p], tmp)
    if !isempty(measCost)
      δj[1:length(p)] .-= measCost[cand[p]]
    end
    # add most informative sample
    δfim, idx = findmax(real.(δj[1:length(p)]))
    addSample!(m, cand, idx, p, tmp[:,1], wlog, numericInv=numericInv)
  end

  return nothing
end

"""
  `redIFIM!(δj::Vector{T}, m::Experiment{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T`

calculate the change in the trace of inverse FIM when adding a sample from the candidate set.
The results are stored in the first entries of `δj`
"""
function redIFIM!(δj::Vector{T}, m::Experiment{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T
  batch = size(tmp,2)
  nprod = div(length(cand),batch)
  δj .= 0.0
  for j=1:nprod
    x0 = (j-1)*batch+1
    x1 = j*batch
    δjb = @view δj[x0:x1]
    redIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[cand[x0:x1],:], tmp)
  end
  if nprod*batch+1 <= length(cand)
    x0 = nprod*batch+1
    x1 = length(cand)
    δjb = @view δj[x0:x1]
    redIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[cand[x0:x1],:], tmp[:,1:x1-x0+1])
  end
end

"""
  `redIFIMBatch!(δ::U, ifim::Vector{Matrix{T}}, Ht::Array{T,3}, Σy::Matrix{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T`

computes the decrease in the trace of the inverse FIM when removing a measurement from the experiment
""" 
function redIFIMBatch!(δ::U, ifim::Vector{Matrix{T}}, Ht::Array{T,3}, Σy::Matrix{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T
  # add contributions for all components
  for k=1:length(ifim)
    mul!(tmp, ifim[k], Ht[:,:,k])
    δ .+= ddot(tmp)./( Σy[:,k] .+ ddot(Ht[:,:,k],tmp) )
  end
end

"""
  `addSample!(m::Experiment{T}, cand::Vector{Int64},idx::Int64 ,p::Vector{Int64},tmp::Vector{T}
                ,wlog::Vector{Int64}; numericInv::Bool=false) where T`

adds the measurement with idx `idx` to the experiment `m`.
"""
function addSample!(m::Experiment{T}, cand::Vector{Int64},idx::Int64
                    ,p::Vector{Int64},tmp::Vector{T}
                    ,wlog::Vector{Int64}; numericInv::Bool=false) where T
  numComp = length(m.fim)
  p_i = p[idx]
  # update samplig log
  push!(wlog,cand[p_i])
  # update FIM
  for k=1:numComp
    m.fim[k] .+= 1.0/m.Σy[cand[p_i],k] .* m.Ht[:,cand[p_i],k]*adjoint(m.Ht[:,cand[p_i],k])
  end
  # update IFIM
  if numericInv
    for k=1:numComp
        m.ifim[k] = inv(m.fim[k])
    end
  else
    for k=1:numComp
      mul!(tmp, m.ifim[k], m.Ht[:,cand[p_i],k])
      denom = m.Σy[cand[p_i],k]+dot( m.Ht[:,cand[p_i],k],tmp )
      m.ifim[k] .-= tmp*adjoint(tmp) ./ denom
    end
  end
  # remove sample from candidate set
  deleteat!(cand,p_i)
  p[findall(x->x>p_i,p)] .-=1
  deleteat!(p,idx)
end
