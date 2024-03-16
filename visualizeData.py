import numpy as np
import os, sys, time
from glob import glob
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import plotly.graph_objects as go
from cycler import cycler
import argparse
# IBM's colorblind-friendly colors
#           |   Red  |   Blue  |  Purple |  Orange | Yellow  |   Green |   Teal  | Grey
hexcolors = ['DC267F', '648FFF', '785EF0', 'FE6100', 'FFB000', '009E73', '3DDBD9', '808080']
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])
class justADictionary():
    def __init__(self, my_name):
        self.name = my_name
        self.c  = 2.99792458 # 1e8   m/s speed of lgiht
        self.h  = 6.62607015 # 1e-34 J/s Plancks constant, 
        self.kB = 1.380649   # 1e-23 J/K Boltzmanns constant, 
        self.Rinf = 10973731.56816  # /m rydberg constant
        self.neutron_proton_mass_ratio = 1.00137842     # m_n / m_p
        self.proton_electron_mass_ratio = 1836.15267343 # m_p / m_e
        self.e = 1.60217663 # 1e-19 C electron charge in coulombs
        self.a = 6.02214076 # 1e23  /mol avogadros number
        self.wien = 2.89777 # 1e-3  m*K  peak_lambda = wien_const / temp
    
    def __str__(self):
        return self.name
CVALS = justADictionary("Useful Physics constants, indexed in class for easy access")
def timeIt(func):
    """@ timeIt: Wrapper to print run time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.clock_gettime_ns(0)
        res = func(*args, **kwargs)
        end_time = time.clock_gettime_ns(0)
        diff = (end_time - start_time) * 10**(-9)
        print(func.__name__, 'ran in %.6fs' % diff)
        return res
    return wrapper

# copied from some other thing, if we want to make a 3D graph of clusters
def graph_3d(min_X, model, coeffs, x_res=10, amps_arnd_center=1.5, x_dim=3):
    model_X = []
    for i in range(x_dim):
        c = min_X[i]
        model_X.append(np.linspace(c-amps_arnd_center, c+amps_arnd_center, x_res))
    # make 3D figures around center
    for i in range(x_dim):
        lbllst = ["X Current (A)","Y Current (A)","Z Current (A)"]
        # get this iteration of (x, y) at fixed z
        model_lst = [model_X[(i+j)%x_dim] for j in range(1,x_dim)]
        Xs = np.array(np.meshgrid(*model_lst, indexing='ij')).T.reshape(-1,x_dim-1)
        xx = Xs[:,0]
        yy = Xs[:,1]
        zz = np.repeat([min_X[i]],len(xx))
        ordered = [xx, yy, zz]
        # restore actual order to the model to get Y
        XX = (ordered[(2-i)%3], ordered[(-i) % 3], ordered[(1-i)%3])
        YY = model(XX, *coeffs)

        # plot the model
        scatter = go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='markers',
            marker=dict(
                size=6,
                color=-YY,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.5
            ),
            name='model'
        )
        surface = go.Surface(x=xx, y=yy, z=YY, opacity=0.8,
                             contours = {"x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color":"white"},
                                         "y": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05},
                                         "z": {"show": True, "usecolormap": True, "highlightcolor": "limegreen", "project_z": True}
                                        }
                            )
        x_lbl = lbllst[(i+1)%3]; y_lbl = lbllst[(i+2)%3]
        layout = go.Layout(
            title=f"Transition Frequency vs. Induced Current in {x_lbl[0]}-{y_lbl[0]}",
            margin=dict(l=0, r=0, b=20, t=50), # tight layout
            width=500, height=500,
            #contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
            scene=dict(
                xaxis=dict(
                    showbackground=False,
                    title=x_lbl,
                    showspikes=False
                ),
                yaxis=dict(
                    showbackground=False,
                    title=y_lbl,
                    showspikes=False
                ),
                zaxis=dict(
                    showbackground=True,
                    title='Frequency (MHz)',
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    backgroundcolor='rgb(230, 230,230)'
                )
            )
        )
        data = [scatter]
        fig = go.Figure(data=data, layout=layout)
        fig.show()

# graph fV.fY,....
def graph_clusters(clusters, event_idxs):
    events = np.unique(clusters["event"])
    numevents = len(events)
    print(f"there are {numevents} events in clusters")
    axislabels = ["fV.fX","fV.fY","fV.fZ"]
    event_colors = ['#' + c for c in hexcolors]
    layout = go.Layout(
        title=f"Clusters",
        margin=dict(l=0, r=0, b=20, t=50), # tight layout
        width=900, height=900,
        #contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
        scene=dict(
            xaxis=dict(
                showbackground=False,
                title=axislabels[0],
                showspikes=False
            ),
            yaxis=dict(
                showbackground=False,
                title=axislabels[1],
                showspikes=False
            ),
            zaxis=dict(
                showbackground=True,
                title=axislabels[2],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )
    fig = go.Figure(layout=layout)
    
    if not len(event_idxs):
        event_idxs = np.random.randint(0,numevents,(10))
    for i in event_idxs:
        if i < 0 or i > numevents: continue
        event = events[i]
        points = clusters[clusters["event"] == event]
        X = points["fV.fX"]; Y = points["fV.fY"]; Z = points["fV.fZ"]
        retLabel = lambda x: str(x["fDetId"])+","+str(x["fSubdetId"])+","+str(x["fLabel[3]"])
        labels = np.array([retLabel(points[i]) for i in range(points.size)])
        fig.add_trace(go.Scatter3d(
            x=Z, y=X, z=Y,
            mode='markers',
            #hover_data=labels,
            marker=dict(
                size=6,
                color = event_colors[i % len(event_colors)], # set color to an array/list of desired values
                opacity=0.8,
            ),
            name="Ev "+str(event)
        ))
    
    fig.show()

def graph_tracks(tracks):
    # go, how do you construct a line tho???
    fields = ['fLabel','fIndex','fStatus','fSign']
    Vs = ['fV.fX','fV.fY','fV.fZ']
    Ps = ['fP.fX','fP.fY','fP.fZ']
    Es = ['fBeta','fDcaXY','fDcaZ','fPVX','fPVY','fPVZ']
    
    pass

@ timeIt
def loadFromNPZ(name):
    return np.load(name+".npz")['arr_0']

# ----- MAIN ----- #
def main(config):
    tracks = loadFromNPZ("../tracks")
    print(f"loaded {tracks.size} tracks with fields {tracks.dtype.fields}")

    clusters = loadFromNPZ("../clusters")
    print(f"loaded {clusters.size} clusters with fields {clusters.dtype.fields}")

    event_idxs = config.evs1.extend(config.evs2)
    if not event_idxs:
        event_idxs = []
    graph_clusters(clusters, event_idxs)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", dest="evs1", default=[], nargs="+",
                        help="List of events to plot in clusters. Default is 10 random ones")
    parser.add_argument("--ev", dest="evs2", default=[], nargs="+",
                        help="List of events to plot in clusters. Default is 10 random ones")
    '''
    parser.add_argument("--label", dest="label", default="",
                        help="Label of pkl file (after seed part).") 
    parser.add_argument("--seeds", dest="seeds", default=[-1], nargs="+",
                        help="List of target seeds to plot.") 
    parser.add_argument("--ext", dest="ext", default="png",
                        help="Image extension (e.g., pdf or png)") 
    parser.add_argument("--convergence", dest='convergence', default='max',
                        help="How normalized parameter iterations are combined into a total convergence level: 'sum', 'max', 'min', 'mean'")
    parser.add_argument("--plot", dest='plot', default=[], nargs="+",
                        help="List of plot specifications: \nloss [UNIF_LEN] ['avg']: make loss plots, [int length for smoothing] [make moving average loss plot] \nall: plot parameter iterations, \
                              \nconv: make parameter convergence plots\n dedx [NUM_BINS]: make histogram of energy data [int number of histogram bins], edit configurations in main() 471 \
                              ")
    parser.add_argument("--linewidth", dest='linewidth', default=None,
                        help="List of plot specifications: \nloss: make loss plots, \nall: plot parameter iterations, \nconvergence: make parameter convergence plots")
    parser.add_argument("--ladd", dest='label_add', default='',
                        help="List of plot specifications: \nloss: make loss plots, \nall: plot parameter iterations, \nconvergence: make parameter convergence plots")
    '''
    args = parser.parse_args()
    main(args)