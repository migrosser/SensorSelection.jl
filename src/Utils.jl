export maxIndices

"""
    `maxIndices(x::Array{T}, numIndices::Int64) where T`

find the indices with the largest elements in an Array
"""
function maxIndices(x::Array{T}, numIndices::Int64) where T
  p = sortperm(vec(x),rev=true)
  return p[1:numIndices]
end

# compute dot products for each column of x
function ddot(x::Matrix{T}) where T
  return dot.(eachcol(x),eachcol(x))
end

# compute dot products for each column of x and y
function ddot(x::Matrix{T}, y::Matrix{T}) where T
  return dot.(eachcol(x),eachcol(y))
end

# compute tr(A*B*A^H) using the cyclic permutation invariances of the trace
# compute only elements relevant for the product
function trp(A::Matrix{T}, B::Matrix{T}, AhA::Matrix{T}) where T
  AhA .= adjoint(A)*A
  return dot(AhA,B)
end