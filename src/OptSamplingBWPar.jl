export optSamplingBW!, removeSample!

"""
    `optSampling!(m::Measurement, numSamples::Int64)`

select sensors such that the estimation of the parameter-vector `x` has minimum uncertainty.

# Arguments:
* `m::Measurement`            - Measurement object
* `numSamples::Vector{Int64}` - number of samples
"""
function optSamplingBW!(m::MeasurementPar{T}, numSamples::Int64, Σx::Matrix{T}, numSimSamp::Int64, batch::Int64) where T
  @info "optSamplingBW"
  # remove samples
  cand = collect(1:length(m.Ht))
  tmp = zeros(T,size(m.fim[1],1))
  tmp_ifim = zeros(T, size(m.ifim[1]))
  δj = zeros(T,length(cand))
  @showprogress 1 "Select Sensors..." for i=1:div(length(cand)-numSamples,numSimSamp)
    δj .= 0.0
    removeSamples!(m,cand,tmp,tmp_ifim,δj,numSimSamp,batch)
  end
end

"""
  `removeSample!(w::Vector{Bool}, cand::Vector{Int64}, fim::Array{Float64,3}, Ht::Array{Float64,3},r::Matrix{Float64})`

remove the sample from `cand` which minimizes maximizes the trace of the FIM (i.e. minimizes uncertainty).

# Arguments:
* `m::Measurement`         - Measurement object
* `cand::Vector{Int64}`    - candidate samples
"""
function removeSamples!(m::MeasurementPar{T}, cand::Vector{Int64}, tmp::Vector{Vector{T}}, tmp_ifim::Matrix{T}, δj::Vector{T}, numSimSamp::Int64, batch::Int64) where T
  numComp = length(m.fim)

  # find index which yields the largest reduction in average variance (trace of inverse FIM)
  # randomize order of candidates to avoid systemic errors
  getFIMChange!(δj,m,cand,tmp)

  # find the smallest elements and iteratively remove the best candidates
  p = sortperm( real.(δj) )[1:batch]
  cand2 = copy( cand[p] )
  δj2 = copy(δj[p])
  for i=1:numSimSamp-1
    δfim, idx = findmin(real.(δj2))
    removeSample!(m, cand2, δj2, idx, tmp[1], tmp_ifim)
    # re-estimate the changes in uncertainty
    getFIMChange!(δj2, m, cand2)
  end
  # remove the last sample
  δfim, idx = findmin(real.(δj2))
  removeSample!(m, cand2, δj2, idx, tmp[1], tmp_ifim)

  return nothing
end

"""
  calculate the change in the trace of inverse FIM when removing a sample from the candidate set
"""
function getFIMChange!(δj::Vector{T}, m::MeasurementPar{T}, cand::Vector{Int64}) where T
  numComp = length(m.fim)
  # shuffle!(cand)
  for k=1:numComp
    # here ht should be a Distributed Array containing the measurement vectors
    δj += map(x->FIMChange(m,x),ht)
  end
end

function FIMChange(m::MeasurementPar,ht::Vector{T}) where T
  tmp = m.ifim[k]*m.Ht[:,c,k]
  return dot(tmp,tmp)/( m.Σy[c,k] - dot(m.Ht[:,c,k], tmp ) )
end

function removeSample!(m::MeasurementPar{T}, cand::Vector{Int64},δj::Vector{T},idx::Int64,tmp::Vector{T},tmp_ifim::Matrix{T}) where T
  numComp = length(m.fim)
  # update samplig scheme
  m.w[cand[idx]] = 0
  # update FIM
  for k=1:numComp
    m.fim[k] .-= 1.0/m.Σy[cand[idx],k] .* m.Ht[:,cand[idx],k]*adjoint(m.Ht[:,cand[idx],k])
  end
  # update IFIM
  for k=1:numComp
    tmp .= m.ifim[k]*m.Ht[:,cand[idx],k]
    denom = m.Σy[cand[idx],k]-dot( m.Ht[:,cand[idx],k],tmp )
    tmp_ifim .= tmp*adjoint(tmp) ./ denom
    m.ifim[k] += Hermitian(0.5*(tmp_ifim.+adjoint(tmp_ifim)))
  end


  # remove sample from candidate set
  deleteat!(cand,idx)
  deleteat!(δj,idx)
end
