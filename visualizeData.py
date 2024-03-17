import numpy as np
import os, sys, time
from glob import glob
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt 
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import plotly.graph_objects as go
import plotly.express as px
from cycler import cycler
import argparse
# IBM's colorblind-friendly colors
#           |   Red  |   Blue  |  Purple |  Orange | Yellow  |   Green |   Teal  | Grey
hexcolors = ['DC267F', '648FFF', '785EF0', 'FE6100', 'FFB000', '009E73', '3DDBD9', '808080']
mpl.rcParams['axes.prop_cycle'] = cycler('color', [mpl.colors.to_rgba('#' + c) for c in hexcolors])

#plotly colorscales at https://plotly.com/python/builtin-colorscales/
def nRGBsFromPoints(color_points: list[tuple], n_colors: int) -> np.ndarray:
    """return ndarr of n_colors rgb points discretizing the colorspace parametrized by color_points"""
    ret = False
    N = len(color_points)
    nstep = max(1,int(np.ceil(n_colors / (N-1)))) # make more than needed and cut later\
    for i in range(1,N):
        new = np.linspace(color_points[i-1], color_points[i], nstep, False)
        ret = new if isinstance(ret, bool) else np.vstack((ret, new)) 
    ret = np.vstack((ret, color_points[-1])) # add final color :)

    colors = np.array(["rgb(%.2f,%.2f,%.2f)" % tuple(i) for i in ret[:n_colors]])
    return colors
viridis_points = [px.colors.hex_to_rgb(i) for i in px.colors.sequential.Viridis_r]
turbo_points   = [px.colors.hex_to_rgb(i) for i in px.colors.sequential.Turbo_r]

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
Ith = lambda i: str(i) + ("th" if (abs(i) % 100 in (11,12,13)) else ["th","st","nd","rd","th","th","th","th","th","th"][abs(i) % 10])

# graph cylinders in plotly
def goCylinder(r, h, a=0, nt=100, nz=50, color='blue', opacity=0.1):
    """
    parametrized cylinder w radius r, height h, base z a, number of theta points nt, number of z points nz
    """
    theta = np.linspace(0, 2*np.pi, nt)
    z = np.linspace(a, a+h, nz )
    theta, z = np.meshgrid(theta, z)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    cylinder = go.Surface(x=x, y=y, z=z,colorscale = [[0, color],[1, color]],
                          showscale=False, opacity=opacity, showlegend=False, hoverinfo='skip')
    return cylinder

def addCylindersToFig(fig: go.Figure):
    """ plot the boundary cylinders onto a figure """
    opacity = 0.1
    middle = [goCylinder(r, 500, -300, 18, 2, 'blue', opacity) for r in (84,133,248)]
    close = [goCylinder(r, 500, -300, 18, 2, 'red', opacity) for r in (4,7,16,38,43)]
    far = [goCylinder(r, 500, -300, 18, 2, 'purple', opacity) for r in (292.5,383.15)]
    fig.add_traces(middle+close+far)

defaultLayout = lambda axislabels: go.Layout(
        title=f"Clusters",
        margin=dict(l=10, r=10, b=50, t=120), # tight layout, give title some room
        width=1500, height=1000,
        #contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
        scene=dict(
            camera = dict(projection=dict(type = "orthographic")),
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
                showspikes=False,
                showbackground=False,
                title=axislabels[2],
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                backgroundcolor='rgb(230, 230,230)'
            )
        )
    )

fIds = ("fDetId","fLabel[3]","fSubdetId")
retLabel = lambda x: ':'.join([str(x[fid]) for fid in fIds])
MAXEVENTS = 300000
def getFieldPoints(clusters: np.ndarray, field: str, idxs: list[int], maxnumevents=MAXEVENTS, printinfo=0):
    ''' concatenates [clusters[clusters[field] == thing] where
     - idxs: specifies idxs into np.unique(clusters[field])
     - printinfo: 0-no info, 1-only size of concatenated, 2-all fields info
     - maxnumevents: maximum number of events to include so plotly doesn't crash 
    '''
    events = np.unique(clusters[field])
    if not len(idxs):  # maximize number of events possible
        idxs = np.linspace(0,events.shape[0],events.shape[0],endpoint=False,dtype=int)
        np.random.shuffle(idxs)
    points = False
    events_used = []
    for evidx in idxs:
        event = events[evidx]
        ev_clusters = clusters[clusters[field] == event]
        if printinfo == 2: print(f"{evidx} {field} {event} contains {ev_clusters.size} clusters")
        if ev_clusters.size > MAXEVENTS:
            print(f"{field} {event} has {ev_clusters.size} points!")
            continue
        if isinstance(points, bool):
            points = ev_clusters
        else:
            if ev_clusters.size + points.size > MAXEVENTS: continue
            points = np.concatenate((points, ev_clusters))
        events_used.append(event)
    if isinstance(points, bool):  # all event idxs were too large
        evidx = idxs[0]
        event = events[evidx]
        print(f"using the first {maxnumevents} pts from {Ith(evidx)} ev {event}...")
        points = clusters[clusters[field] == event][:maxnumevents]
        events_used.append(event)
    if printinfo: print(f"total {points.size} points")
    return (points, events_used)

# graph fV.fY,....
def graph_clusters(clusters, event_idxs, width, height, cylinder, field_idx=2):
    events = np.unique(clusters["event"])
    numevents = len(events)
    print(f"there are {numevents} events in clusters")
    axislabels = ["fV.fX","fV.fY","fV.fZ"]
    event_colors = ['#' + c for c in hexcolors]
    layout = defaultLayout(axislabels)
    layout.width = width; layout.height = height
    fig = go.Figure(layout=layout)
    if cylinder: addCylindersToFig(fig)
    if not len(event_idxs):
        event_idxs = np.random.randint(0,numevents,(10))
    for fId in fIds: print(fId,max(clusters[fId]))
    for idx in event_idxs:
        if idx < 0 or idx > numevents: continue
        event = events[idx]
        points = clusters[clusters["event"] == event]
        print(f"{Ith(idx)} event {event} has {len(points)} points")
        X = points["fV.fX"]; Y = points["fV.fY"]; Z = points["fV.fZ"]
        # color based on fId
        colors = [event_colors[x % len(event_colors)] for x in points[fIds[field_idx]]]
        labels = np.array([retLabel(pt) for pt in points])
        fig.add_trace(go.Scatter3d(x=X, y=Y, z=Z,mode='markers',name="Ev "+str(event),
                                    marker=dict(size=6, opacity=0.8, color = colors),
                                    customdata=labels,
                                    hovertemplate='<br>'.join(
                                        ['x: %{x:.2f}','y: %{y:.2f}','z: %{z:.2f}','%{customdata}'])
                                    ))
    fig.show()


def graphID(clusters, event_idxs, field_idx, groupwidth, width, height, cylinder):
    ''' field_idx between 0 and 2 for clusters '''
    if field_idx >= len(fIds): field_idx = len(fIds) - 1
    if field_idx < 0: field_idx = 0
    field = fIds[field_idx]
    print(f"graphing fId {field}")
    points, events_used = getFieldPoints(clusters, "event", event_idxs, printinfo=0)
    axislabels = ["fV.fX","fV.fY","fV.fZ"]
    layout = defaultLayout(axislabels)
    layout.title=field+": "+', '.join([str(ev) for ev in events_used])
    layout.width = width; layout.height = height
    fig = go.Figure(layout=layout)
    if cylinder: 
        print('adding cylinders...')
        print(width,height)
        addCylindersToFig(fig)
    unique_fields = np.unique(points[field])
    cmin = min(unique_fields); cmax = max(unique_fields)
    # graph colors
    #event_colors = ['#' + c for c in hexcolors]
    Ncolors = len(unique_fields)
    #colorspace = nRGBsFromPoints(turbo_points, Ncolors)
    #colorspace = nRGBsFromPoints(viridis_points, Ncolors)
    if groupwidth < 10: groupwidth = 10
    if field_idx != 2: groupwidth = 1
    # reduce lag for all the damn traces
    for i in range(0,Ncolors,groupwidth):
        fieldidxs = range(i,min(i+groupwidth,Ncolors))
        pts, fields_used = getFieldPoints(points, field, fieldidxs, printinfo=0)
        labels = np.array([retLabel(pt) for pt in pts])
        X = pts["fV.fX"]; Y = pts["fV.fY"]; Z = pts["fV.fZ"]
        #color = event_colors[i % len(event_colors)]
        #color = colorspace[i]
        trace = go.Scatter3d(x=X, y=Y, z=Z,mode='markers',name=field+":"+str(min(fields_used))+"-"+str(max(fields_used)),
                            marker=dict(size=4,cmin=cmin,cmax=cmax,color=pts[field],colorscale="Turbo_r",opacity=0.8),
                            customdata=labels,
                            hovertemplate='<br>'.join(
                                ['x: %{x:.2f}','y: %{y:.2f}','z: %{z:.2f}','%{customdata}'])
                            )
        fig.add_trace(trace)
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
    # load data
    tracks = loadFromNPZ("../tracks")
    print(f"loaded {tracks.size} tracks with fields {tracks.dtype.names}")
    if config.datainfo:
        events = np.unique(tracks["event"])
        evsizes = [tracks[tracks["event"] == event].size for event in events]
        print("\tavg: %.3f std: %.3f min: %d max: %d" % (
            np.mean(evsizes), np.std(evsizes), np.min(evsizes), np.max(evsizes)))

    clusters = loadFromNPZ("../clusters")
    print(f"loaded {clusters.size} clusters with fields {clusters.dtype.names}")
    if config.datainfo:
        events = np.unique(clusters["event"])
        evsizes2 = np.array([[clusters[clusters["event"] == events[i]].size, events[i], i] for i in range(len(events))])
        evsizes2 = evsizes2[evsizes2[:,0].argsort()]
        # print largest event things
        num_extrema_to_show = 10
        print("size statistics:")
        print("\tavg: %d std: %d min: %d max: %d" % (
            np.mean(evsizes2[:,0]), np.std(evsizes2[:,0]), np.min(evsizes2[:,0]), np.max(evsizes2[:,0])))
        print("least # clusters:")
        for i in range(num_extrema_to_show):
            small = evsizes2[i]
            print(f"\t{Ith(small[2])} event #{small[1]} has {small[0]} clusters")
        print("least # clusters:")
        for i in range(num_extrema_to_show):
            large = evsizes2[-i-1]
            print(f"\t{Ith(large[2])} event #{large[1]} has {large[0]} clusters")
        print("events sorted in increasing # of clusters:")
        print(evsizes2[:,1].tolist())

    # --- plotting criteria
    cylinder = False; width = 1500; height = 1000
    ev_idxs = [int(i) for i in config.evs]
    i = 0
    while i < len(config.plot):
        if config.plot[i].startswith('cyl'):
            cylinder = True
        elif config.plot[i].startswith('w'):
            i += 1
            try:
                width = int(config.plot[i])
            except IndexError or ValueError:
                pass
        elif config.plot[i].startswith('h'):
            i += 1
            try:
                height = int(config.plot[i])
            except IndexError or ValueError:
                pass
        i += 1

    if config.clusters:
        print("making clusters")
        graph_clusters(clusters, ev_idxs, width, height, cylinder, config.fieldidx)
    else:
        print("making ID graphs for events",', '.join(str(ev) for ev in ev_idxs))
        graphID(clusters, ev_idxs, config.fieldidx, config.groupwidth, width, height, cylinder)

# largest events: 124,123,101,20,95,132,128,71,86,107   all greater than 1M
# smallest events: 102,197,75,226,227,220,313,22,19,255  all less than 6k

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", "--ev", dest="evs", default=[], nargs="+", 
                        help="List of events to plot in clusters. Default is 10 random ones")
    parser.add_argument("--clusters", "--c", dest="clusters", default=False, action='store_true',
                        help="Graph 10 random clusters or those labeled in events tag")
    parser.add_argument("--field", "--f", dest="fieldidx", default=2, type=int,
                        help="field idx corresponding to (fDetid, fLabel[3], fSubdetid)")
    parser.add_argument("--info", dest='datainfo', default=False, action='store_true',
                        help="show info on distribtuion of # clusters per event")
    parser.add_argument("--groupwidth", "--gw", dest='groupwidth',default=100,type=int,
                        help="width of group for plotting IDs. smallest is 5")
    parser.add_argument("--plot", dest='plot',default=[], nargs="+",
                        help="List of additional plot arguments. 'cylinder' draws boundary cylinders, 'width' and 'height' specify plot dimensions")
    '''
    parser.add_argument("--label", dest="label", default="",
                        help="Label of pkl file (after seed part).") 
    parser.add_argument("--seeds", dest="seeds", default=[-1], nargs="+",
                        help="List of target seeds to plot.") 
    parser.add_argument("--ext", dest="ext", default="png",
                        help="Image extension (e.g., pdf or png)") 
    '''
    args = parser.parse_args()
    main(args)