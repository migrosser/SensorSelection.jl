export optSamplingFW!, addSample!

"""
    `optSampling!(m::Measurement, numSamples::Int64)`

select sensors such that the estimation of the parameter-vector `x` has minimum uncertainty.

# Arguments:
* `m::Measurement`            - Measurement object
* `numSamples::Vector{Int64}` - number of samples
"""
function optSamplingFW!(m::Measurement{T}, numSamples::Int64, batch::Int64=1) where T
  cand = collect(1:size(m.Ht,2))
  tmp = zeros(T,size(m.fim[1],1),batch)
  δj = zeros(T,length(cand))
  wlog = Int64[]
  @showprogress 1 "Select Sensors..." for i=1:numSamples
    addSample!(m,cand,tmp,δj,wlog)
  end
  m.w[wlog] .= 1
  return wlog
end

"""
  `addSample!(w::Vector{Bool}, cand::Vector{Int64}, fim::Array{Float64,3}, Ht::Array{Float64,3},r::Matrix{Float64})`

adds the sample from `cand` which minimizes maximizes the trace of the FIM (i.e. minimizes uncertainty).

# Arguments:
* `m::Measurement`         - Measurement object
* `cand::Vector{Int64}`    - candidate samples
"""
function addSample!(m::Measurement{T}, cand::Vector{Int64}, tmp::Matrix{T}, δj::Vector{T}, wlog::Vector{Int64}) where T
  # inverse of current FIMs
  numComp = length(m.fim)
  for k=1:numComp
    m.ifim[k] = Hermitian(inv(m.fim[k]))
  end

  # find index which yields the largest reduction in average variance (trace of inverse FIM)
  redIFIM!(δj,m,cand,tmp)
  δfim, idx = findmax(real.(δj[1:length(cand)]))

  # add the corresponding sample
  addSample!(m, cand, idx, wlog)

  return nothing
end

"""
  `redIFIM!(δj::Vector{T}, m::Measurement{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T`

calculate the change in the trace of inverse FIM when adding a sample from the candidate set.
The results are stored in the first entries of `δj`
"""
function redIFIM!(δj::Vector{T}, m::Measurement{T}, cand::Vector{Int64}, tmp::Matrix{T}) where T
  batch = size(tmp,2)
  nprod = div(length(cand),batch)
  δj .= 0.0
  for j=1:nprod
    x0 = (j-1)*batch+1
    x1 = j*batch
    δjb = @view δj[x0:x1]
    redIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[x0:x1,:], tmp)
  end
  if nprod*batch+1 <= length(cand)
    x0 = nprod*batch+1
    x1 = length(cand)
    δjb = @view δj[x0:x1]
    redIFIMBatch!(δjb, m.ifim, m.Ht[:,cand[x0:x1],:], m.Σy[x0:x1,:], tmp[:,1:x1-x0+1])
  end
end

function redIFIMBatch!(δ::U, ifim::Vector{Hermitian{T,Array{T,2}}}, Ht::Array{T,3}, Σy::Matrix{T}, tmp::Matrix{T}) where U<:AbstractVector{T} where T
  # add contributions for all components
  for k=1:length(ifim)
    mul!(tmp, ifim[k], Ht[:,:,k])
    δ .+= ddot(tmp)./( Σy[:,k] .+ ddot(Ht[:,:,k],tmp) )
  end
end

function addSample!(m::Measurement{T}, cand::Vector{Int64}, idx::Int64, wlog::Vector{Int64}) where T
  numComp = length(m.fim)
  # update samplig log
  push!(wlog,cand[idx])
  # update FIM
  for k=1:numComp
    m.fim[k] .+= 1.0/m.Σy[cand[idx],k] .* m.Ht[:,cand[idx],k]*adjoint(m.Ht[:,cand[idx],k])
  end
  # remove sample from candidate set
  deleteat!(cand,idx)
end
