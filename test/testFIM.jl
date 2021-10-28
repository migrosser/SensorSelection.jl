
@testset "FIM" begin
  Random.seed!(1234)
  N=128
  # measurement matrix
  Ht = collect( reshape( (1.0*I(N)),N,N,1 ) )
  # prior variances
  σx = reshape( sort(rand(N).+eps()), N, 1)
  # measurement noise variances
  σy = rand(N,1).+eps()
  # measurements
  numSamp = rand(1:N)
  sampIdx = shuffle(collect(1:N))[1:numSamp]
  w = zeros(Bool,N); w[sampIdx] .= 1

  # FIM
  fim1_ref = Diagonal(vec(σx).^(-1))
  for i in sampIdx
    fim1_ref[i,i] += σy[i]^(-1)
  end
  fim1 = calculateFIM(σx, σy, Ht, w)[1]
  fim1_sim = calculateFIMSim(vec(σx), σy, Ht, w)

  @test fim1 == fim1_ref
  @test fim1_sim == fim1_ref

  # FIM after a random basis transformation
  p = randperm(N)
  pMat = Matrix{Float64}( I(N)[p,:] )
  fim2_ref = pMat*Diagonal(vec(σx).^(-1))*transpose(pMat)
  for i in sampIdx
    fim2_ref[i,i] += σy[i]^(-1)
  end
  fim2 = calculateFIM(σx, reshape(pMat,N,N,1), σy, Ht, w)[1]

  @test fim2 == fim2_ref
end
