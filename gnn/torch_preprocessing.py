print("")

import os

# Suppress INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# system import
import pkg_resources
import yaml
import pprint

# 3rd party
import torch
from trackml.dataset import load_event
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# local import
from gnn_classes import utils_dir
from utils_torch import config_dict # for accessing predefined configuration files
from utils_torch import config_dic # for accessing predefined output directories

# for preprocessing
from gnn_classes import FeatureStore

# for embedding
from gnn_classes import LayerlessEmbedding
from gnn_classes import EmbeddingInferenceCallback
# for filtering
from gnn_classes import VanillaFilter
from gnn_classes import FilterInferenceCallback

# excluded necessary import
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Walking through GNN track labeling pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Output each sanity check print statement")
    #parser.add_argument("path", type=str, help="The path to the file")
    return parser.parse_args()

def build(args, HOMEDIR):
    #BUILD
    print("BUILD\n")
    action = 'build'

    config_file = pkg_resources.resource_filename(
                        "exatrkx",
                        os.path.join('configs', config_dict[action]))
    with open(config_file) as f:
        b_config = yaml.load(f, Loader=yaml.FullLoader)

    if args.verbose:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(b_config)
        print("\n")

    b_config['endcaps'] = False
    b_config['pt_min'] = 1.0 
    b_config['n_workers'] = 1
    b_config['cell_information'] = True
    b_config['n_files'] = 3 
    b_config['noise'] = False
    
    preprocess_dm = FeatureStore(b_config)
    preprocess_dm.prepare_data()
    print("\n")

    feature_data = torch.load(HOMEDIR+"/content/iml2020/feature_store/1000", map_location='cpu')
    if args.verbose: 
        print(feature_data)
        print("\n")

#EMBEDDING
def embedding(args, HOMEDIR, eventnum):
    print("EMBEDDING\n")
    action = 'embedding'

    config_file = pkg_resources.resource_filename(
                        "exatrkx",
                        os.path.join('configs', config_dict[action]))
    with open(config_file) as f:
        e_config = yaml.load(f, Loader=yaml.FullLoader)

    if args.verbose:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(e_config)
        print("\n")

    e_config['train_split'] = [1, 1, 1]

    e_model = LayerlessEmbedding(e_config)

    e_checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filepath=os.path.join(utils_dir.embedding_outdir,'ckpt-{epoch:02d}-{val_loss:.2f}') ,
        save_top_k=3,
        mode='min')
    e_callback_list = [EmbeddingInferenceCallback()]

    e_trainer = Trainer(
    max_epochs = 2,
    limit_train_batches=8,
    limit_val_batches=8,
    callbacks=e_callback_list,
    gpus=1,
    checkpoint_callback=e_checkpoint_callback
    )
    print("\netrainer")

    e_trainer.fit(e_model)
    print("\nefit")

    if args.verbose:
        os.system("ls "+HOMEDIR+"/content/iml2020/embedding_output/train/")
        print("\n")

    embed_outfile = os.path.join(utils_dir.embedding_outdir, "train", eventnum)
    dd = torch.load(embed_outfile)
    if args.verbose:
        print(dd)
        print("\n")

#FILTERING
def filtering(args, HOMEDIR, eventnum):
    print("FILTERING\n")

    action = 'filtering'

    config_file = pkg_resources.resource_filename(
                        "exatrkx",
                        os.path.join('configs', config_dict[action]))
    with open(config_file) as f:
        f_config = yaml.load(f, Loader=yaml.FullLoader)

    if args.verbose:
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(f_config)

    f_config['train_split'] = [1,1, 1]

    f_model = VanillaFilter(f_config)
    f_callback_list = [FilterInferenceCallback()]

    f_trainer = Trainer(
    max_epochs = 2,
    limit_train_batches=1,
    limit_val_batches=1,
    callbacks=f_callback_list,
    gpus=0,
    )
    print("\nftrainer")

    f_trainer.fit(f_model)
    print("\nffit")

    embed_outfile = os.path.join(utils_dir.embedding_outdir, "train", eventnum)
    dd = torch.load(embed_outfile)
    if args.verbose:
        print(dd)
        print("\n")


def main():
    args = parse_args()
    if args.verbose:
        print("Verbose mode is on.")

    HOMEDIR="/home/davidgn/exatrkx-iml2020"
    eventnum="1000"

    if args.verbose:
        hits, cell, particles, truth = load_event(HOMEDIR+"/content/train_10evts/event00000"+eventnum)
        print(hits.head(3))
        print(cell.head(3))
        print(particles.head(3))
        print(truth.head(3))
        print("\n")

    build(args, HOMEDIR)

    embedding(args, HOMEDIR,eventnum)

    filtering(args, HOMEDIR,eventnum)

    # Now that we've successfully embedded and filtered our input data into an appropriate format, 
    # we must execute the training of the GNN via tensorflow
    # $ convert2tf.py --edge-name "e_radius" --truth-name "y_pid"
    # $ train_gnn_tf.py --max-epochs 8
    # $ eval_gnn_tf.py
    # $ tracks_from_gnn.py


if __name__ == "__main__":
    print("\n")
    main()
