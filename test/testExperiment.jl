using SensorSelection, LinearAlgebra, Random, Test

function testForwardSelection(N=128; var=1.e-2, numSimSamp=1, numSimCand=N)
  # measurement matrix
  Ht = collect( reshape( (1.0*I(N)),N,N,1 ) )
  # prior variances
  σx = reshape( sort(rand(N).+eps()), N, 1)
  # measurement noise variances
  σy = var*ones(N,1)
  # number of samples
  numSamp = rand(numSimSamp:numSimSamp:N)

  # experiment object
  exp1 = Experiment(Ht, σy, σx, zeros(Bool, N)) # non-uniform prior
  exp2 = Experiment(Ht, σx, σy, zeros(Bool, N)) # non-uniform noise

  # perform forward optimization
  wlog1 = optSamplingFW!(exp1, numSamp, numSimSamp, numSimCand)
  msk1 = zeros(Bool,N); msk1[wlog1] .= 1
  wlog2 = optSamplingFW!(exp2, numSamp, numSimSamp, numSimCand)
  msk2 = zeros(Bool,N); msk2[wlog2] .= 1

  # tests wlog
  @test wlog1==N:-1:(N-numSamp+1)
  @test wlog2==1:numSamp
  # test consistency of internal sampling mask
  @test exp1.w==msk1
  @test exp2.w==msk2
end

function testBackwardSelection(N=128; var=1.e-2, numSimSamp=1, numSimCand=N)
  # measurement matrix
  Ht = collect( reshape( (1.0*I(N)),N,N,1 ) )
  # prior variances
  σx = reshape( sort(rand(N).+eps()), N, 1)
  # measurement noise variances
  σy = var*ones(N,1)
  # number of samples
  numSamp = rand(numSimSamp:numSimSamp:N)
  
  # experiment object
  exp1 = Experiment(Ht, σy, σx, ones(Bool, N)) # non-uniform prior
  exp2 = Experiment(Ht, σx, σy, ones(Bool, N)) # non-uniform noise

  # perform backwards optimization
  wlog1 = optSamplingBW!(exp1, numSamp, numSimSamp, numSimCand)
  msk1 = ones(Bool,N); msk1[wlog1] .= 0
  wlog2 = optSamplingBW!(exp2, numSamp, numSimSamp, numSimCand)
  msk2 = ones(Bool,N); msk2[wlog2] .= 0

  # test wlog
  numRmSamp = div(N-numSamp,numSimSamp)*numSimSamp
  @test wlog1==1:(numRmSamp)
  @test wlog2==N:-1:(N-numRmSamp+1)
  # test consistency of internal sampling mask
  @test exp1.w==msk1
  @test exp2.w==msk2
end

@testset "Experiment" begin
  Random.seed!(1234)
  # optimization without batching
  testForwardSelection(128)
  testBackwardSelection(128)
  # optimization with batching
  numSimSamp = rand(1:10)
  numSimCand = numSimSamp+rand(0:128-numSimSamp)
  testForwardSelection(128, numSimSamp=numSimSamp, numSimCand=numSimCand)
  testBackwardSelection(128, numSimSamp=numSimSamp, numSimCand=numSimCand)
end