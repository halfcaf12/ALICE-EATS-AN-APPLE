import numpy as np
import os, sys
from glob import glob
from tqdm import tqdm
from numpy.lib import recfunctions as rf

# -------- CONTROL DA DAMN ISH ----------- #
# make points / Clusters
make_points = False
make_tracks = False
make_csvs   = False

curdir = os.getcwd()
dir0 = curdir+"/Clusters"
dir1 = curdir+"/RecTracks"
# making huge dataset of all data together

def getDtype(filename):
    with open(filename, "r") as f:
        labels = f.readline().strip().split(';')
        types = []
        firstColumn = True
        for column in f.readline().strip().split(';'):
            if firstColumn:
                if labels[0] == "fDetId":
                    types.append(np.dtype(np.int8))
                firstColumn = False
            try:
                int(column)
                types.append(np.int32)
                continue
            except ValueError:
                pass
            try:
                float(column)
                types.append(np.float32)
                continue
            except ValueError:
                pass
            print("u fucked up somehow... new type???",column)
            types.append(np.dtype.str)
    lst = [("event",np.dtype(np.int16))]
    for i in range(len(labels)):
        field = (labels[i], types[i])
        lst.append(field)
    dtype = np.dtype(lst)
    return dtype

clusters_fnames = glob(dir0+"/*")
tracks_fnames = glob(dir1+"/*")
points_dtype = getDtype(clusters_fnames[0])
tracks_dtype = getDtype(tracks_fnames[0])

if make_points:
    points = False
    print("making clusters csv to npz")
    with tqdm(total = len(clusters_fnames)) as pbar:
        for filename in clusters_fnames: 
            event = int(filename.split('_')[-2][5:])
            data = np.loadtxt(filename, delimiter=';', skiprows=1)
            event_column = np.atleast_2d(np.tile(event,data.shape[0])).T
            data_w_event = rf.unstructured_to_structured(np.hstack((event_column, data)),points_dtype)
            if isinstance(points, bool):
                points = np.array(data_w_event, dtype=points_dtype)
            else:
                points = np.append(points, data_w_event, axis=0)
            pbar.update(1)
    print(points.shape)
    print(points[0])
    print(points.dtype)
    np.savez_compressed('clusters.npz', points)

print(tracks_dtype)
if make_tracks:
    tracks = False
    print("making tracks csv to npz")
    with tqdm(total = len(tracks_fnames)) as pbar:
        for filename in tracks_fnames: 
            event = int(filename.split('_')[-2][5:])
            data = np.loadtxt(filename, delimiter=';', skiprows=1)
            if event == 107:
                print(data)
                print(len(data.shape))
            if data.size:  # data may be empty - no tracks
                if len(data.shape) == 1: # need to make 2D at least
                    data = np.array([data])
                    print(data)
                event_column = np.atleast_2d(np.tile(event,data.shape[0])).T
                data_w_event = rf.unstructured_to_structured(np.hstack((event_column, data)),tracks_dtype)
                if isinstance(tracks, bool):
                    tracks = np.array(data_w_event, dtype=tracks_dtype)
                else:
                    tracks = np.append(tracks, data_w_event, axis=0)
            pbar.update(1)
    print(tracks.shape)
    print(tracks[0])
    print(tracks.dtype)
    np.savez_compressed('tracks.npz', tracks)

