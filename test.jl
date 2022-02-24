using DynamicalSystems, PyPlot

plt.style.use("seaborn-paper")
PyPlot.rc("font", family="serif")
PyPlot.rc("text", usetex=true)
PyPlot.matplotlib.rcParams["axes.titlesize"] = 10
PyPlot.matplotlib.rcParams["axes.labelsize"] = 10
PyPlot.matplotlib.rcParams["xtick.labelsize"] = 9
PyPlot.matplotlib.rcParams["ytick.labelsize"] = 9
const sw = 3.40457
const dw = 7.05826

# now we define the cardiac map
function f(x; p=[100.0,43.54])
	c = p[1]
	d_min = p[2]
	d = c-x
	while d < d_min
		d = d + c
	end
	return 258.0 + 125.0*exp(-0.068*(d-d_min)) - 350.0*exp(-0.028*(d-d_min))
end

# let's wrap this in a covenience function for sequence generation
function F(;p=[500.0,43.54], n=10000, x0=230.0)
	x = zeros(n)
	x[1] = f(x0;p)
	for i in 1:(n-1)
		x[i+1] = f(x[i];p)
	end
	return x
end

BCLs = 40.0:(2.0^-3):400.0
m, τ = 6, 1 # Symbol size/dimension and embedding lag
xs = zeros(Float64,length(BCLs),100)
hs_perm = Float64[]
hs_wtperm = Float64[]
hs_ampperm = Float64[]
hs_UN = Float64[]

base = Base.MathConstants.e
for (n,BCL) in enumerate(BCLs)
	xs[n,:] .= F(n=1000,p=[BCL,43.54])[end+1-100:end]
	hperm = Entropies.genentropy(xs[n,:], SymbolicPermutation(m = m, τ = τ), base = base)
	hwtperm = Entropies.genentropy(xs[n,:], SymbolicWeightedPermutation(m = m, τ = τ), base = base)
	hampperm = Entropies.genentropy(xs[n,:], SymbolicAmplitudeAwarePermutation(m = m, τ = τ), base = base)
	#un = minimum(xs[n,:]) .+ (maximum(xs[n,:])-minimum(xs[n,:])).*rand(Float64,size(xs,2))
	
	push!(hs_perm, hperm); 
	push!(hs_wtperm, hwtperm); 
	push!(hs_ampperm, hampperm);
	#push!(hs_UN, Entropies.genentropy(un, SymbolicPermutation(m = m, τ = τ), base = base));
end

fig,axs = plt.subplots(2,1,figsize=(sw,sw),constrained_layout=true,sharex=true)
axs[1].plot(BCLs, xs, ".k", markersize=2, alpha=0.05, label="map")
axs[2].plot(BCLs, hs_perm, "-C0", alpha=0.5, label="\$h_6 (SP)\$")
axs[2].plot(BCLs, hs_wtperm, "-C1", alpha=0.5, label="\$h_6 (WT)\$")
axs[2].plot(BCLs, hs_ampperm, "-C2", alpha=0.5, label="\$h_6 (SAAP)\$")
#axs[2].plot(BCLs, hs_UN, "-k", alpha=0.5, label="Uniform noise")
axs[2].set_xlabel("BCL [ms]")
axs[2].set_ylabel("Entropies")
axs[1].set_ylabel("APD [ms]")
axs[1].set_xlim([0.0,400.0])
yl = axs[1].get_ylim()
axs[1].set_ylim([0.0,yl[2]])
axs[2].legend(loc=0, edgecolor="none")
plt.savefig("./julia_fig.pdf", bbox_inches="tight")
plt.close()
