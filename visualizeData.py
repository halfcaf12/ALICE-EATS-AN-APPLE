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
from fractions import Fraction

# ---- CONSTANTS ---- #
PLOT_DIR = "./xgboost_plots"
RBOUNDARIES = [0,5,10,17,35,40,45,84,134,250,292.5,373.5,500]
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

# ----- PLOTTING FUNCTIONS ----- #
def goCylinder(r, h, a=0, nt=100, nz=50, color='blue', opacity=0.1):
    """
    parametrized cylinder w radius r, height h, base z a, number of theta points nt, number of z points nz
    """
    theta = np.linspace(0, 2*np.pi, nt+1)
    z = np.linspace(a, a+h, nz )
    theta, z = np.meshgrid(theta, z)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    cylinder = go.Surface(x=x, y=y, z=z,colorscale = [[0, color],[1, color]],
                          showscale=False, opacity=opacity, showlegend=False, hoverinfo='skip')
    return cylinder

def goCircle(r, z_offset, nt, color, width, opacity=0.5):
    theta = np.linspace(0, 2*np.pi, nt+1)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    circ = go.Scatter3d(x = x.tolist(), y = y.tolist(), z = [z_offset]*(nt+1),
                        mode ='lines', line = dict(color=color, width=width),
                        opacity=opacity, showlegend=False, hoverinfo='skip')
    return circ

def goLines(rs, z_offset, nt, color, width, opacity=0.5):
    theta = np.linspace(0, 2*np.pi, nt, endpoint=False)
    x = []; y = []; z = []
    for th in theta:
        x.extend([rs[0]*np.cos(th),rs[1]*np.cos(th),None])
        y.extend([rs[0]*np.sin(th),rs[1]*np.sin(th),None])
        z.extend([z_offset, z_offset,None])
    line = go.Scatter3d(x = x, y = y, z = z, mode ='lines', line = dict(color=color, width=width),
                        opacity=opacity, showlegend=False, hoverinfo='skip')
    return line

def addBoundaries(fig: go.Figure, make_cylinders=False):
    """ plot the boundary cylinders onto a figure """
    opacity = 0.1; linewidth = 3
    z_offset = -400
    colors = ['blue','red','purple']
    for r in RBOUNDARIES[1:7]:  # close
        if make_cylinders: fig.add_trace(goCylinder(r, -z_offset*2, z_offset, 18, 2, colors[1], opacity))
        fig.add_trace(goCircle(r, z_offset, 18, colors[1], linewidth))
    fig.add_trace(goLines((RBOUNDARIES[1],RBOUNDARIES[6]), z_offset, 18, colors[1], linewidth))
    for r in RBOUNDARIES[7:10]:   # middle
        if make_cylinders: fig.add_trace(goCylinder(r, -z_offset*2, z_offset, 18, 2, colors[0], opacity))
        fig.add_trace(goCircle(r, z_offset, 18, colors[0], linewidth))
    fig.add_trace(goLines((RBOUNDARIES[7],RBOUNDARIES[9]),  z_offset, 18, colors[0], linewidth))
    for r in RBOUNDARIES[10:-1]:    # far
        if make_cylinders: fig.add_trace(goCylinder(r, -z_offset*2, z_offset, 18, 2, colors[2], opacity))
        fig.add_trace(goCircle(r, z_offset, 18, colors[2], linewidth))
    fig.add_trace(goLines((RBOUNDARIES[10],RBOUNDARIES[-2]), z_offset, 18, colors[2], linewidth))
    
defaultLayout = lambda axislabels: go.Layout(
        title=f"Clusters",
        margin=dict(l=10, r=10, b=50, t=120), # tight layout, give title some room
        width=1500, height=1000,
        #contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True),
        scene=dict(
            camera = dict(eye=dict(x=0., y=0., z=1.5),  # XY plane looking down
                          projection=dict(type = "orthographic")),
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

# ----- HANDLING POINTS ------ #
fIds = ("fDetId","fLabel[3]","fSubdetId")
retLabel = lambda x: ':'.join([str(x[fid]) for fid in fIds])
MAXEVENTS = 300000
def getFieldPoints(clusters: np.ndarray, field: str | int, idxs: list[int], maxnumevents=MAXEVENTS, printinfo=0):
    ''' concatenates [clusters[clusters[field] == thing] where
     - idxs: specifies idxs into np.unique(clusters[field])
       - if not specified, will fill with as many events up to maxnumevents 
     - printinfo: 0-no info, 1-only size of concatenated, 2-all fields info
     - maxnumevents: maximum number of events to include so plotly doesn't crash 
    '''
    if isinstance(field,str):
        events = np.unique(clusters[field])
    else:
        assert isinstance(field, int)
        events = np.unique(clusters[:,field])
    if not len(idxs):  # maximize number of events possible
        idxs = np.linspace(0,events.shape[0],events.shape[0],endpoint=False,dtype=int)
        np.random.shuffle(idxs)
    points = False
    events_used = []
    for evidx in idxs:
        event = events[evidx]
        if isinstance(field, str):
            ev_clusters = clusters[clusters[field] == event]
        else:
            ev_clusters = clusters[clusters[:,field] == event]
        if printinfo == 2: print(f"{evidx} {field} {event} contains {ev_clusters.size} clusters")
        if ev_clusters.size > maxnumevents:
            if printinfo:
                print(f"f {field} ev {event} has {ev_clusters.size} points!")
            continue
        if isinstance(points, bool):
            points = ev_clusters
        else:
            if ev_clusters.size + points.size > maxnumevents: continue
            points = np.concatenate((points, ev_clusters))
        events_used.append(event)
    if isinstance(points, bool):  # all event idxs were too large
        evidx = idxs[0]
        event = events[evidx]
        print(f"using the first {maxnumevents} pts from {Ith(evidx)} ev {event}...")
        if isinstance(field,str):
            points = clusters[clusters[field] == event][:maxnumevents]
        else:
            points = clusters[clusters[:,field] == event][:maxnumevents]
        events_used.append(event)
    if printinfo: print(f"total {points.size} points")
    return (points, events_used)

def graph_clusters(clusters, event_idxs, width, height, cylinder, field_idx=2):
    events = np.unique(clusters["event"])
    numevents = len(events)
    print(f"there are {numevents} events in clusters")
    axislabels = ["fV.fX","fV.fY","fV.fZ"]
    event_colors = ['#' + c for c in hexcolors]
    layout = defaultLayout(axislabels)
    layout.width = width; layout.height = height
    fig = go.Figure(layout=layout)
    addBoundaries(fig, cylinder)
    if not len(event_idxs):
        event_idxs = np.random.randint(0,numevents,(10))
    for fId in fIds: print(fId,max(clusters[fId]))
    for idx in event_idxs:
        if idx < 0 or idx > numevents: continue
        event = events[idx]
        points = clusters[clusters["event"] == event]
        print(f"{Ith(idx)} event {event} has {len(points)} points")
        X = points[axislabels[0]]; Y = points[axislabels[1]]; Z = points[axislabels[2]]
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
    addBoundaries(fig, cylinder)
    unique_fields = np.unique(points[field])
    cmin = min(unique_fields); cmax = max(unique_fields)
    if cmin == -3141593:
        points[points[field] == -3141593][field] = -3
    Ncolors = len(unique_fields)
    if groupwidth < 10: groupwidth = 10
    if field_idx != 2: groupwidth = 1
    for i in range(0,Ncolors,groupwidth):
        fieldidxs = range(i,min(i+groupwidth,Ncolors))
        pts, fields_used = getFieldPoints(points, field, fieldidxs, printinfo=0)
        labels = np.array([retLabel(pt) for pt in pts])
        X = pts[axislabels[0]]; Y = pts[axislabels[1]]; Z = pts[axislabels[2]]
        name = field
        if groupwidth > 1:
            name += f":{len(pts)}:{min(fields_used)}-{max(fields_used)}"
        trace = go.Scatter3d(x=X, y=Y, z=Z,mode='markers',name=name,
                            marker=dict(size=4,cmin=cmin,cmax=cmax,color=pts[field],colorscale="Turbo_r",opacity=0.8),
                            customdata=labels,
                            hovertemplate='<br>'.join(
                                ['x: %{x:.2f}','y: %{y:.2f}','z: %{z:.2f}','%{customdata}'])
                            )
        fig.add_trace(trace)
    fig.show()

def graphCylindricalPoints(clusters, labelidx, title="", markersize=4, width=1500, height=1000, cylinder=False):
    """graph points from clusters w/ entries as (radius, theta, z, sector, label, fSubdetId)
     - labelidx is 3,4,5
     - plotly: name, markersize, width, height, cylinder will add cylinder objects
      - traces are by SECTOR """
    axislabels = ["fV.fX","fV.fY","fV.fZ"]
    layout = defaultLayout(axislabels)
    layout.width = width; layout.height = height
    layout.title = title
    fig = go.Figure(layout=layout)
    addBoundaries(fig, cylinder)
    # TRACE BY SECTOR BC THAT IS THE BESSSTTTT
    labels = np.unique(clusters[:,labelidx])
    cmin = min(labels); cmax = max(labels)
    sectors = np.unique(clusters[:,4])
    for sector in sectors:
        points = clusters[clusters[:,4] == sector]
        ring = points[0,3]  # display ring and sector information
        name=f"R{int(ring)} S{int(sector)}: {len(points)} pts"
        # convert from cylindrical to cartesian coordinates
        X = points[:,0]*np.cos(points[:,1])
        Y = points[:,0]*np.sin(points[:,1])
        Z = points[:,2]
        possible_label_idxs = [3,4,5]
        #possible_label_idxs.remove(labelidx)
        colors = points[:,labelidx]
        name += ", avg: %.3f" % np.mean(colors)  # and average subdetId
        ids = []
        for pt in points:
            id_txt = []
            for id in pt[3:]:
                if id == int(id):
                    id_txt.append("%d" % id)
                else:
                    id_txt.append("%.3f" % id)
            ids.append(', '.join(id_txt))
        trace = go.Scatter3d(x=X, y=Y, z=Z,mode='markers',name=name,
                            marker=dict(size=markersize,cmin=cmin,cmax=cmax,color=colors,colorscale="Turbo_r",opacity=0.8),
                            customdata=ids,
                            hovertemplate='<br>'.join(
                                ['x: %{x:.2f}','y: %{y:.2f}','z: %{z:.2f}','id: %{customdata}'])
                            )
        fig.add_trace(trace)
    fig.show()

def plotRTheta(coords, label_idx, title, radius_bounds=(0,0), rticks=[], showplot=True, polar=True):
    PLOT_SIZE = (10,8)
    numradialticks = 8  # not including largest radius
    numthetaticks = 18
    rticks=RBOUNDARIES
    
    rmin = 0
    if radius_bounds[1]:
        coords = coords[coords[:,0] < radius_bounds[1]]
    if radius_bounds[0]:
        coords = coords[coords[:,0] > radius_bounds[0]]
        rmin = max(0,min(coords[:,0])-2)

    labels = coords[:,label_idx]
    R = coords[:,0]; rmax = max(R)
    T = coords[:,1]

    turbo_map = mpl.colormaps.get_cmap('turbo')
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=PLOT_SIZE) if polar else plt.subplots(figsize=PLOT_SIZE)
    sc = ax.scatter(T, R, c=labels, s=5, cmap=turbo_map)
    
    # tick that bih
    if len(rticks):
        start = 0; end = 0
        for i in range(len(rticks)):
            if rticks[i] < rmin:  # start tick after first rmin
                start += 1
            if rticks[i] < rmax:
                end += 1
        radius_ticks = rticks[start:end]
    else:
        radius_ticks = np.linspace(rmin,rmax,numradialticks,endpoint=False)
    
    angles = []; anglelabels = []
    for i in range(numthetaticks):
        if polar:
            angles.append(i*360/numthetaticks)
        else:
            angles.append(i*2*np.pi/numthetaticks)
        label = "$"
        frac = Fraction(i*2/numthetaticks).limit_denominator(numthetaticks)  # in terms of pi
        if i and i != numthetaticks/2:
            label += r"\frac{"+str(frac.numerator)+"}{"+str(frac.denominator)+"}"
        label += r"\pi$"
        anglelabels.append(label)
    if polar:
        ax.set(rmin=rmin, rmax=rmax)
        ax.set_rgrids(radii=radius_ticks, labels=[""]*len(radius_ticks))
        ax.set_thetagrids(angles, labels=anglelabels)
    else:
        ax.set(ylim=(rmin,rmax), yticks=radius_ticks, xticks=angles, xticklabels=anglelabels)
    ax.grid(alpha=0.1)
    ax.set(title=title)
    plt.colorbar(sc)

    pltname = PLOT_DIR+"/plot-"+("polar" if polar else "2d")+"_"+"_".join(title.lower().split(' '))+".pdf"
    fig.savefig(pltname, bbox_inches="tight")
    print("Saved figure to "+pltname)
    if showplot: plt.show()


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
        evsizes2 = np.array([[i, events[i], clusters[clusters["event"] == events[i]].size] for i in range(len(events))])
        evsizes2 = evsizes2[evsizes2[:,2].argsort()]
        # print largest event things
        num_extrema_to_show = 10
        print("size statistics:")
        print("\tavg: %d std: %d min: %d max: %d" % (
            np.mean(evsizes2[:,2]), np.std(evsizes2[:,2]), np.min(evsizes2[:,2]), np.max(evsizes2[:,2])))
        print("least # clusters:")
        for i in range(num_extrema_to_show):
            small = evsizes2[i]
            print(f"\t{Ith(small[0])} event #{small[1]} has {small[2]} clusters")
        print("least # clusters:")
        for i in range(num_extrema_to_show):
            large = evsizes2[-i-1]
            print(f"\t{Ith(large[0])} event #{large[1]} has {large[2]} clusters")
        print("events sorted in increasing # of clusters:")
        print(evsizes2[:,1].tolist())
        print("event idxs sorted in increasing # of clusters:")
        print(evsizes2[:,0].tolist())
        np.savez_compressed("events_increasing_size.npz",evsizes2)

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