# ALICE-EATS-AN-APPLE
GNNs for HEP Particle Tracking


TODO:
1. GNN in this folder DAVIDo~~!
1. XGBoost to do radial stuff
1. XGBoost + KMeansClustering to identify the 18-sided structures present in data
1. Generalize `treeToCSV` to any ROOT structure, and use arguments

## Data processing
1. ROOT files, stored in `roots/*.root` and scraped from CERN's open data EOS, are converted into CSVs using `root treeToCSV.C` after downloading `root` from https://root.cern/install/. This stores each ROOT tree as a separate csv in `csvs/Clusters/` and `csvs/RecTracks`. 
1. Convert CSVs to NPZs using `processCSVtoNPZ.py`, which stores `../clusters.npz` and `../tracks.npz`. 

## Data visualization
Assumes that you have your data of the form `../clusters.npz` to store particle clusters.
run `visualizeData.py --events 0 1 2 3 4` to print the ``fSubdetId'' field of the clusters. This uses plotly in order to plot the clusters and colors them using this field.
Specifications to `visualizeData` are as follows:
- events are indexed 0 through 317, and visualizeData will automatically generate only 300,000 points at a time. Over 500,000 points using plotly can crash your computer. 
- 