import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc

plt.style.use('seaborn-paper')

rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)
matplotlib.rcParams['axes.titlesize'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9

sw = 3.40457
dw = 7.05826

import ordpy
import igraph
import cairo


# now we define the cardiac map
def f(x=230.0, p=np.array([100.0,43.54]), s=0.0):
	c = p[0]
	d_min = p[1]
	d = c-x
	while d < d_min:
		d = d + c
	return 258.0 + 125.0*np.exp(-0.068*(d-d_min)) - 350.0*np.exp(-0.028*(d-d_min)) + s*np.random.randn()

# let's wrap this in a covenience function for sequence generation
def map(p=np.array([500.0,43.54]), n=10000, x0=230.0, s=0.0):
	x = np.zeros(n)
	x[0] = f(x0,p,s=s)
	for i in range(n-1):
		x[i+1] = f(x[i],p,s=s)
	return x

def bifplot():
	fig,axs = plt.subplots(2,1,figsize=(sw,sw),sharex=True,constrained_layout=True)
	BCLs= np.arange(40.0,400.0,step=2.0**-3)
	pts = np.zeros((len(BCLs),100))
	hRE = np.zeros(len(BCLs))
	#hTE = np.zeros(len(BCLs))
	#hSE = np.zeros(len(BCLs))
	hGN = np.zeros(len(BCLs))
	#hUN = np.zeros(len(BCLs))
	for n,BCL in enumerate(BCLs):
		print(f"BCL = {BCL}\n")
		pts[n,:] = map(p=np.array([BCL,43.54]),n=300,x0=40+np.random.rand()*200.0,s=1e-15)[-100:]
		hRE[n] = ordpy.renyi_entropy(pts[n,:], dx=6)
		#hTE[n] = ordpy.tsallis_entropy(pts[n,:], dx=6)
		#hSE[n] = ordpy.permutation_entropy(pts[n,:], dx=6)
		hGN[n] = ordpy.global_node_entropy(pts[n,:], dx=6)
		#hUN[n] = ordpy.permutation_entropy(40.0 + 210.0*np.random.rand(100), dx=5)
	axs[0].plot(BCLs, pts,  ".k", markersize=2, alpha=0.05)
	axs[1].plot(BCLs, hRE, '-C0', markersize=2, alpha=0.5, label=r"Renyi Entropy ($d=6$)")
	#axs[1].plot(BCLs, hTE, '-C1', markersize=2, alpha=0.5, label=r"Tsallis Entropy ($d=6$)")
	#axs[1].plot(BCLs, hSE, '-C2', markersize=2, alpha=0.5, label=r"Shannon Entropy ($d=6$)")
	axs[1].plot(BCLs, hGN, '-C3', markersize=2, alpha=0.5, label=r"Global Node Entropy ($d=6$)")
	#axs[1].plot(BCLs, hUN, '-k', markersize=2, alpha=0.5, label=r"Uniform Noise Entropy ($d=6$)")
	axs[0].set_xlim([0.0,400.0])
	yl = axs[0].get_ylim()
	axs[0].set_ylim([0.0,yl[1]])
	yl = axs[1].get_ylim()
	axs[1].set_ylim([0.0,yl[1]])
	axs[1].set_xlabel("BCL [ms]")
	axs[0].set_ylabel("APD [ms]")
	axs[1].legend(loc=0, edgecolor="none")
	plt.savefig("./bifplot.pdf",bbox_inches="tight")
	plt.close() #plt.show()
bifplot() # maybe unnecessary to be replotting the map bif diagram

BCLs = [400.0, 346.0, 344.0, 301.0, 300.0, 174.0, 170.0, 150.0, 145.0, 120.0, 114.0, 100.0, 90.0, 83.0, 40.0]
time_series = [map(p=np.array([BCL,43.54]),n=11000)[-10000:] for BCL in BCLs]
time_series += [40.0 + 210.0*np.abs(np.random.uniform(size=len(time_series[0])))]

labels = [f'{BCL} ms' for BCL in BCLs]
labels.append("Uniform noise")

# now we define the cardiac map (I don't remember what this one is for?)
"""
def g(x, p):
	a = x[0]
	c = x[1]
	d_min = p[1]
	d = c-x
	while d < d_min:
		d = d + c
	return 258.0 + 125.0*np.exp(-0.068*(d-d_min)) - 350.0*np.exp(-0.028*(d-d_min))
"""
fig, axs = plt.subplots(1,2, figsize=(dw, sw), constrained_layout=True)
for series, label in zip(time_series[-1:], labels[-1:]):
	for dx in range(1,9):
		f = ordpy.missing_patterns(series, dx=dx, taux=1, return_fraction=True, return_missing=False, probs=False, tie_precision=0)
		

def asymptoticMissingConnections(data, max_dim=9):
	missingFractions = []
	for dx in range(1,max_dim+1):
		f = ordpy.missing_patterns(data, dx=dx, taux=1, return_fraction=True, return_missing=False, probs=False, tie_precision=0)
		missingFractions.append(f)
	return np.array(missingFractions)

HC = [ordpy.complexity_entropy(series, dx=5) for series in time_series]

fig, axs = plt.subplots(1,2, figsize=(dw, sw), constrained_layout=True)
for HC_, label_, BCL_ in zip(HC, labels, BCLs):
	axs[0].scatter(HC_[0], HC_[1], label=label_, s=10*np.sqrt(BCL_/400), alpha=0.67)
axs[0].scatter(HC[-1][0], HC[-1][1], s=3, color="k", label=labels[-1])
axs[0].set_xlabel('Permutation entropy, $H$')
axs[0].set_ylabel('Statistical complexity, $C$')
axs[0].legend(loc=0, ncol=2, edgecolor="none")

#for TS_, label_ in zip(time_series, labels):
#	axs[1].scatter(range(len(TS_)), TS_, label=label_, s=3, alpha=0.67)
#axs[1].set_xlabel('Iteration, $n$')
#axs[1].set_ylabel('APD [ms]')
##axs[1].legend(loc=0, edgecolor="none")

"""
for series_, label_, BCL_ in zip(time_series, labels, BCLs):
	f = asymptoticMissingConnections(series_)
	axs[1].plot(np.arange(1,len(f)+1), f, "-", linewidth=3*np.sqrt(BCL_/400), label=label_, alpha=0.67)
f = asymptoticMissingConnections(time_series[-1])
axs[1].plot(np.arange(1,len(f)+1), f, "-k", linewidth=1, label=labels[-1], alpha=0.67)
#axs[1].set_xlim([1,len(f)])
axs[1].set_xlabel('Embedding dimension')
axs[1].set_ylabel('Missing fraction, $f$')
"""

series = map(p=np.array([BCL,43.54]),n=10100,x0=40+np.random.rand()*200.0,s=1e-15)[-10000:]
for dx in range(2,9):
	curve = []
	for N in range(100,10000,100):
		f = ordpy.missing_patterns(series[:N], dx=dx, taux=1, return_fraction=True, return_missing=False, probs=False, tie_precision=0)
		curve.append(f)
	axs[1].plot(range(100,10000,100), curve, "-", label=r"dx = {dx}")
plt.savefig("./HC_fig.pdf", bbox_inches="tight")
plt.close()

# make ordinal networks
vertex_list, edge_list, edge_weight_list = list(), list(), list()
for series in time_series:
	v_, e_, w_   = ordpy.ordinal_network(series, dx=5, taux=1, tie_precision=0, overlapping=True)
	vertex_list += [v_]
	edge_list   += [e_]
	edge_weight_list += [w_]

def create_ig_graph(vertex_list, edge_list, edge_weight):

	G = igraph.Graph(directed=True)

	for v_ in vertex_list:
		G.add_vertex(v_)

	for [in_, out_], weight_ in zip(edge_list, edge_weight):
		G.add_edge(in_, out_, weight=weight_)

	return G

graphs = []

for v_, e_, w_ in zip(vertex_list, edge_list, edge_weight_list):
	graphs += [create_ig_graph(v_, e_, w_)]

def igplot(g,filename=""):
	f = igraph.plot(g,
			filename,
			layout=g.layout_kamada_kawai(), #layout_circle(),
			bbox=(500,500),
			margin=(40, 40, 40, 40),
			vertex_label = [s.replace('|','') for s in g.vs['name']],
			vertex_label_color='#202020',
			vertex_color='#969696',
			vertex_size=20,
			vertex_font_size=6,
			edge_width=(1 + 8*np.asarray(g.es['weight'])).tolist(),
		)
	return f

for graph_, label_ in zip(graphs, labels):
	print(label_)
	#display(SVG(igplot(graph_)._repr_svg_()))
	igplot(graph_,filename=f"./graph_{label_}.pdf");

