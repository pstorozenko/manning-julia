using Pkg; Pkg.activate(".")
using Downloads, DataFrames, CSV, SHA, FreqTables, StatsPlots, StatsBase, GLM, Arrow
urls = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names"
]
files = Downloads.download.(urls)
data_sha = [
    0xad, 0xfa, 0x6b, 0x6d, 0xca,
    0x24, 0xa6, 0x3f, 0xe1, 0x66,
    0xa9, 0xe7, 0xfa, 0x01, 0xce,
    0xe4, 0x33, 0x58, 0x57, 0xd1
]
open(files[1]) do f
    @assert sha1(f) == data_sha
end
print(read(files[2], String))
colnames = [:CRIM, :ZN, :INDUS, :CHAS, :NOX, :RM, :AGE, :DIS, :RAD, :TAX, :PTRATIO, :B, :LSTAT, :MEDV]

df = CSV.read(files[1], DataFrame, header=colnames, delim =" ", ignorerepeated = true)

nominal = [:CHAS, :RAD]
continnuous = [c for c in colnames if !(c in nominal)]


freqtable.(Ref(df), nominal)
hists = Dict([c => histogram(df[:,c], title = c) for c in continnuous])
hists[:MEDV]

filter!(:MEDV => !=(50), df)
medv_cor = [corkendall(df[!, :MEDV], df[!, c]) for c in continnuous]
sp = sortperm(medv_cor, rev=true)

cor_M = [corkendall(df[!, c1], df[!, c2]) for c1 in continnuous[sp], c2 in continnuous[sp]]
heatmap(cor_M)

df_cor = DataFrame(var=continnuous, cor = medv_cor)
sort!(df_cor, :cor, by=abs, rev=true)

function scatter_linear(df, c1, c2)
    x = df[!, c1]
    y = df[!, c2]
    X = [ones(length(x)) x]
    m = lm(X, y)
    y_hat = predict(m)
    p = plot(x, y_hat, title="$c1 vs $c2")
    scatter!(x, y)
    p
end

scatter_linear(df, :MEDV, :INDUS)
sc_plots = Dict([c=>scatter_linear(df, :MEDV, c) for c in continnuous])
sc_plots[:B]
select!(df, Not(:B))
sc_plots[:CRIM]
sc_plots[:DIS]
sc_plots[:ZN]
transform!(df, :CRIM => ByRow(log) => :CRIM)
transform!(df, :DIS => ByRow(log) => :DIS)
transform!(df, :ZN => ByRow(x -> ifelse(x == 0, 0, 1)) => :ZN)
mean(df[!, :MEDV])

function ±(x, dx)
    x - dx, x + dx
end

function boot_mean(x)
    m = mean(x)
    m - quantile(x, 0.05), m, m + quantile(x, 0.95)
end
aux = ["lower","mean","upper"]
res = []

function conf_plot(df, c)
    dfc = flatten(combine(groupby(df, c), :MEDV => boot_mean => :MEDV), :MEDV)
    dfc[!, "type"] = repeat(aux, length(unique(df[!, c])))
    d = unstack(dfc, :type, :MEDV) 
    scatter(d[!, c], d[!, :mean], yerror=(d[!, :lower], d[!, :upper]))
end

confplots = Dict([
    c => conf_plot(df, c) for c in [:ZN, nominal...]
])

# If those nominal values would be more spread, I'd use some kind of categorical array
confplots[:ZN]
confplots[:CHAS]
confplots[:RAD]

select!(df, Not(:RAD))

Arrow.write("results.arrow", df)
