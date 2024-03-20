# ALICE-EATS-AN-APPLE
**ALICE E**xperiment and **T**estbed **S**imulation: **A N**ovel **A**lgorithmic **P**article **P**hysics **L**earning **E**xperiment

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

## Learning Data Parameters with XGBoost Framework
In `boostData.py`, tweaking around with the optimal parameters used for mapping various parameters of the dataset loaded in `../clusters.npz`, namely the fields of `fDetId`, `fSubdetId`, and `fLabel[3]`. `fSubdetId` is the most interesting and shows preliminary track processing. 
- an ensemble tree method with `XGBoost` is found that has 99.6% accuracy to within a single integer classification for `fSubdetId`, which ranges from 0 to up to 7000 and isn't trivially spatially correlated. 