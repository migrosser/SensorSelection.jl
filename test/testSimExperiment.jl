function testForwardSelection(N=128; var=1.e-2, numSimSamp=1, numSimCand=N)
  # measurement matrices => perform each measurement with 2 sensors with different SNR
  Ht1 = collect( (1.0*I(N)) )
  Ht2 = collect( (1.0*I(N)) )
  Ht = cat(Ht1,Ht2,dims=3)
  # prior variances
  σx1 = sort(rand(N).+eps())
  σx2 = var*ones(N)
  # measurement noise variances
  σy1 = cat(var*ones(N), 2*var*ones(N), dims=2)
  σy2 = cat(σx1, 2*σx1, dims=2)
  # number of samples
  numSamp = rand(numSimSamp:numSimSamp:N)

  # experiment object
  exp1 = SimExperiment(Ht, σy1, σx1, zeros(Bool, N)) # non-uniform prior
  exp2 = SimExperiment(Ht, σy2, σx2, zeros(Bool, N)) # non-uniform noise

  # perform forward optimization
  wlog1 = optSamplingFW!(exp1, numSamp, numSimSamp, numSimCand)
  msk1 = zeros(Bool,N); msk1[wlog1] .= 1
  wlog2 = optSamplingFW!(exp2, numSamp, numSimSamp, numSimCand)
  msk2 = zeros(Bool,N); msk2[wlog2] .= 1

  # tests wlog
  numActSamp = div(numSamp,numSimSamp)*numSimSamp
  @test wlog1==N:-1:(N-numActSamp+1)
  @test wlog2==1:numActSamp
  # test consistency of internal sampling mask
  @test exp1.w==msk1
  @test exp2.w==msk2
end

function testBackwardSelection(N=128; var=1.e-2, numSimSamp=1, numSimCand=N)
  # measurement matrices => perform each measurement with 2 sensors with different SNR
  Ht1 = collect( (1.0*I(N)) )
  Ht2 = collect( (1.0*I(N)) )
  Ht = cat(Ht1,Ht2,dims=3)
  # prior variances
  σx1 = sort(rand(N).+eps())
  σx2 = var*ones(N)
  # measurement noise variances
  σy1 = cat(var*ones(N), 2*var*ones(N), dims=2)
  σy2 = cat(σx1, 2*σx1, dims=2)
  # number of samples
  numSamp = rand(numSimSamp:numSimSamp:N)
  
  # experiment object
  exp1 = SimExperiment(Ht, σy1, σx1, ones(Bool, N)) # non-uniform prior
  exp2 = SimExperiment(Ht, σy2, σx2, ones(Bool, N)) # non-uniform noise

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

@testset "SimExperiment" begin
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