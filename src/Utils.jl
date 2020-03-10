export maxIndices

"""
    `maxIndices(x::Array{T}, numIndices::Int64) where T`

find the indices with the largest elements in an Array
"""
function maxIndices(x::Array{T}, numIndices::Int64) where T
  p = sortperm(vec(x),rev=true)
  return p[1:numIndices]
end
