export initFIM, calculateFIM, calculateFIMSim

"""
    `initFIM(Φ::Array{T,3}, Σx::Matrix{T}) where T`

returns the Fisher information matrices for Gaussian priors with variances 
stored in the columns of `Σx`. Moreover, the parameters are linearly transformed 
using the matrices stored in the first two dimentions of `Φ`.
"""
function initFIM(Φ::Array{T,3}, Σx::Matrix{T}) where T
  nx,ny,nc = size(Φ)

  Σ0 = [zeros(T,nx,nx) for k=1:nc]# prior covariance matrix
  for k=1:nc
    Σ0[k] .= inv( Φ[:,:,k]*( diagm(Σx[:,k])*transpose(Φ[:,:,k]) ) )
  end

  return Σ0
end

"""
    `initFIM(Σx::Matrix{T}) where T`

returns the Fisher information matrices for a Gaussian priors with variances 
stored in the columns of `Σx`.
"""
function initFIM(Σx::Matrix{T}) where T
  nx,nc = size(Σx)

  Σ0 = [zeros(T,nx,nx) for k=1:nc]# prior covariance matrix
  for k=1:nc
    Σ0[k] .= inv(diagm(Σx[:,k]))
  end

  return Σ0
end


"""
  `calculateFIM(Σx::Matrix{T},Σy::Matrix{T},Ht::Array{T,3}, w::Vector{Bool})  where T`

computes the Fisher information matrices for the prior variances stored for in the columns of `Σx`
and the noise variances stored in the column of `Σy`. `Ht` contains the transposed of the 
measurement matrices (stacked along the third dimension) and `w` is a mask indicating
which measurements (i.e. column of the matrices in Ht) are to be included in the FIMs.
"""
function calculateFIM(Σx::Matrix{T},Σy::Matrix{T},Ht::Array{T,3}, w::Vector{Bool}) where T
  nx,nc = size(Σx)

  fim = [inv(diagm(Σx[:,k])) for k=1:nc]
  for k=1:nc
    @showprogress 1 "Calculate $(k). FIM..." for i=1:length(w)
      if w[i]==0
        continue
      else
        fim[k] .+= 1.0/Σy[i,k] .* Ht[:,i,k]*adjoint(Ht[:,i,k])
      end
    end
  end

  return fim
end

"""
  `calculateFIMSim(Σx::Matrix{T},Σy::Matrix{T},Ht::Array{T,3}, w::Vector{Bool})  where T`

computes the Fisher information matrices for the prior variances stored for in the columns of `Σx`
and the noise variances stored in the column of `Σy`. `Ht` contains the transposed of the 
measurement matrices (stacked along the third dimension) and `w` is a mask indicating
which measurements (i.e. column of the matrices in Ht) are to be included in the FIMs.
In contrast to `calcuateFIM`, this method assumes that the measurems concatenated along the third
dimension of `Ht` are acquired simultaneously.
"""
function calculateFIMSim(Σx::Vector{T},Σy::Matrix{T},Ht::Array{T,3}, w::Vector{Bool}) where T
  fim = inv(diagm(Σx))
  @showprogress 1 "Calculate FIM..." for i=1:length(w)
    if w[i]==0
      continue
    else
      # add simultaneously acquired measurements
      for k=1:size(Ht,3)
        fim .+= 1.0/Σy[i,k] .* Ht[:,i,k]*adjoint(Ht[:,i,k])
      end
    end
  end
  return fim
end

"""
  `calculateFIM(Σx::Matrix{T},Σy::Matrix{T},Ht::Array{T,3}, w::Vector{Bool})  where T`

computes the Fisher information matrices for the prior variances stored for in the columns of `Σx`
and the noise variances stored in the column of `Σy`. The prior covariance matrices are additionally
transformed using the matrices stored in `Φ`.
Ht` contains the transposed of the measurement matrices (stacked along the third dimension) 
and `w` is a mask indicating which measurements (i.e. column of the matrices in Ht) 
are to be included in the FIMs.
"""
function calculateFIM(Σx::Matrix{T}, Φ::Array{T,3}, Σy::Matrix{T}, Ht::Array{T,3}, w::Vector{Bool}) where T
  nx,ny,nc = size(Φ)

  fim = [zeros(T,nx,nx) for k=1:nc]# prior covariance matrix
  for k=1:nc
    fim[k] .= inv( Φ[:,:,k]*( diagm(Σx[:,k])*transpose(Φ[:,:,k]) ) )
  end

  for k=1:nc
    @showprogress 1 "Calculate $(k). FIM..." for i=1:length(w)
      if w[i]==0
        continue
      else
        fim[k] .+= 1.0/Σy[i,k] .* Ht[:,i,k]*adjoint(Ht[:,i,k])
      end
    end
  end

  return fim
end
