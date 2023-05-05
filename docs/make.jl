using LSShadows
using Documenter

DocMeta.setdocmeta!(LSShadows, :DocTestSetup, :(using LSShadows); recursive=true)

makedocs(;
    modules=[LSShadows],
    authors="Raimel A. Medina",
    repo="https://github.com/RaimelMedina/LSShadows.jl/blob/{commit}{path}#{line}",
    sitename="LSShadows.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://RaimelMedina.github.io/LSShadows.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/RaimelMedina/LSShadows.jl",
    devbranch="main",
)
