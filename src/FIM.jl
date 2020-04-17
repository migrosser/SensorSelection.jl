export initFIM, calculateFIM, calculateIFIM

"""
    `initFIM(Φ::Array{T,3}, Σx::Matrix{T}) where T`

returns the Fisher information matrix for a Gaussian prior with variances `Σx`.
Moreover, the parameters are linearly transformed using `Φ`.
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
    `initFIM(Φ::Array{T,3}, Σx::Matrix{T}) where T`

returns the Fisher information matrix for a Gaussian prior with variances `Σx`.
"""
function initFIM(Σx::Matrix{T}) where T
  nx,nc = size(Σx)

  Σ0 = [zeros(T,nx,nx) for k=1:nc]# prior covariance matrix
  for k=1:nc
    Σ0[k] .= inv(diagm(Σx[:,k]))
  end

  return Σ0
end

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
