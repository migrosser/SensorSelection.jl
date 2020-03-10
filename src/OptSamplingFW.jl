export optSamplingFW!, addSample!

"""
    `optSampling!(m::Measurement, numSamples::Int64)`

select sensors such that the estimation of the parameter-vector `x` has minimum uncertainty.

# Arguments:
* `m::Measurement`            - Measurement object
* `numSamples::Vector{Int64}` - number of samples
"""
function optSamplingFW!(m::Measurement{T}, numSamples::Int64) where T
  cand = collect(1:size(m.Ht,2))
  tmp = zeros(T,size(m.fim[1],1))
  @showprogress 1 "Select Sensors..." for i=1:numSamples
    addSample!(m,cand,tmp)
  end
end

"""
  `addSample!(w::Vector{Bool}, cand::Vector{Int64}, fim::Array{Float64,3}, Ht::Array{Float64,3},r::Matrix{Float64})`

adds the sample from `cand` which minimizes maximizes the trace of the FIM (i.e. minimizes uncertainty).

# Arguments:
* `m::Measurement`         - Measurement object
* `cand::Vector{Int64}`    - candidate samples
"""
function addSample!(m::Measurement{T}, cand::Vector{Int64}, tmp::Vector{T}) where T
  # inverse of current FIMs
  numComp = length(m.fim)
  for k=1:numComp
    m.ifim[k] = Hermitian(inv(m.fim[k]))
  end

  # find index which yields the largest reduction in average variance (trace of inverse FIM)
  idx = 0
  δfim = 0.0
  # randomize order of candidates to avoid systemic errors
  shuffle!(cand)
  for (j,c) in enumerate(cand)
    # change for this candidate
    δj = 0.0
    for k=1:numComp
      # exploit that the inverse FIM is symmetric
      tmp[:] .= m.ifim[k]*m.Ht[:,c,k]
      δj += dot(tmp,tmp)/( m.Σy[c,k] + dot(m.Ht[:,c,k], tmp ) )
    end
    # compare to current best estimate
    if real(δj) > δfim
      idx = j
      δfim = real(δj)
    end
  end

  # update samplig scheme
  m.w[cand[idx]] = 1

  # update FIM
  for k=1:numComp
    m.fim[k] .+= 1.0/m.Σy[cand[idx],k] .* m.Ht[:,cand[idx],k]*adjoint(m.Ht[:,cand[idx],k])
  end

  # remove sample from candidate set
  deleteat!(cand,idx)

  return nothing
end


# """
#     `optSampling!(m::Measurement, numSamples::Int64)`
#
# select sensors such that the estimation of the parameter-vector `x` has minimum uncertainty.
#
# # Arguments:
# * `m::Measurement`            - Measurement object
# * `numSamples::Vector{Int64}` - number of samples
# """
# function optSamplingFW!(m::Measurement{T}, numSamples::Int64) where T
#   @info "optSampling2"
#   cand = collect(1:size(m.Ht,2))
#   tmp = zeros(T,size(m.fim[1],1))
#   δj = zeros(T,length(cand))
#   @showprogress 1 "Select Sensors..." for i=1:numSamples
#     δj .= 0.0
#     addSample!(m,cand,tmp,δj)
#   end
# end
#
# """
#   `addSample!(w::Vector{Bool}, cand::Vector{Int64}, fim::Array{Float64,3}, Ht::Array{Float64,3},r::Matrix{Float64})`
#
# adds the sample from `cand` which minimizes maximizes the trace of the FIM (i.e. minimizes uncertainty).
#
# # Arguments:
# * `m::Measurement`         - Measurement object
# * `cand::Vector{Int64}`    - candidate samples
# """
# function addSample!(m::Measurement{T}, cand::Vector{Int64}, tmp::Vector{T}, δj::Vector{T}) where T
#   # inverse of current FIMs
#   numComp = length(m.fim)
#   for k=1:numComp
#     m.ifim[k] = Hermitian(inv(m.fim[k]))
#   end
#
#   # find index which yields the largest reduction in average variance (trace of inverse FIM)
#   # randomize order of candidates to avoid systemic errors
#   shuffle!(cand)
#   for k=1:numComp
#     for (j,c) in enumerate(cand)
#       # exploit that the inverse FIM is symmetric
#       tmp[:] .= m.ifim[k]*m.Ht[:,c,k]
#       δj[j] += dot(tmp,tmp)/( m.Σy[c,k] + dot(m.Ht[:,c,k], tmp ) )
#     end
#   end
#   δfim, idx = findmax(real.(δj))
#
#   # update samplig scheme
#   m.w[cand[idx]] = 1
#
#   # update FIM
#   for k=1:numComp
#     m.fim[k] .+= 1.0/m.Σy[cand[idx],k] .* m.Ht[:,cand[idx],k]*transpose(m.Ht[:,cand[idx],k])
#   end
#
#   # remove sample from candidate set
#   deleteat!(cand,idx)
#
#   return nothing
# end
