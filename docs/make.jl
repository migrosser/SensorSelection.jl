using Documenter, SensorSelection

makedocs(
    modules = [SensorSelection],
    sitename = "SensorSelection.jl",
    authors = "Mirco Grosser",
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(repo   = "github.com/migrosser/SensorSelection.jl")
           