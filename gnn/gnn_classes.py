import time
import os
import sys
import multiprocessing as mp

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets import graphs

import matplotlib.pyplot as plt
import sklearn.metrics

import networkx as nx
import pandas as pd

import tensorflow as tf
from graph_nets import modules
from graph_nets import blocks
from graph_nets.graphs import GraphsTuple 
import sonnet as snt

import torch
import torch.nn as nn
from torch.nn import Linear

from torch_cluster import radius_graph
from torch_geometric.data import DataLoader

from pytorch_lightning.callbacks import Callback

from utils_torch import graph_intersection
import utils_torch

device = 'cuda'

class utils_dir():
    try:
        inputdir = os.environ['TRKXINPUTDIR']
    except KeyError as e:
        print("Require the directory of tracking ML dataset")
        print("Given by environment variable: TRKXINPUTDIR")
        raise(e)

    try:
        output_base = os.environ['TRKXOUTPUTDIR']
    except KeyError as e:
        print("Require the directory for outputs")
        print("Given by environment variable: TRKXOUTPUTDIR")
        raise(e)

    # print("Input: {}".format(inputdir))
    # print("Output: {}".format(output_base))
    detector_path = os.path.join(inputdir, '..', 'detectors.csv')

    feature_outdir   = os.path.join(output_base, "feature_store") # store converted input information
    embedding_outdir = os.path.join(output_base, "embedding_output") # directory outputs after embedding
    filtering_outdir = os.path.join(output_base, "filtering_output") # directory outputs after filtering
    gnn_inputs       = os.path.join(output_base, "gnn_inputs")       # directory for converted filtering outputs
    gnn_models       = os.path.join(output_base, "gnn_models")       # GNN model outputs
    gnn_output       = os.path.join(output_base, "gnn_eval")         # directory for outputs after evalating GNN
    trkx_output      = os.path.join(output_base, "trkx_output")      # directory for outputs of track candidates
    trkx_eval        = os.path.join(output_base, "trkx_eval")        # directory for evaluating track candidates

    outdirs = [feature_outdir, embedding_outdir, filtering_outdir,
            gnn_inputs, gnn_models, gnn_output, trkx_output]

    if not os.path.exists(feature_outdir):
        [os.makedirs(x, exist_ok=True) for x in outdirs]

    datatypes = ['train', 'val', 'test']

    config_dict = {
        "build": 'prepare_feature_store.yaml',
        'embedding': 'train_embedding.yaml', 
        'filtering': 'train_filter.yaml',
    }

    outdir_dict = {
        "build": feature_outdir,
        'embedding': embedding_outdir,
        'filtering': filtering_outdir,
    }

class graph():

    graph_types = {
    'n_node': tf.int32,
    'n_edge': tf.int32,
    'nodes': tf.float32,
    'edges': tf.float32,
    'receivers': tf.int32,
    'senders': tf.int32,
    'globals': tf.float32,
    }

    def parse_tfrec_function(example_proto):
        graph_types = {
            'n_node': tf.int32,
            'n_edge': tf.int32,
            'nodes': tf.float32,
            'edges': tf.float32,
            'receivers': tf.int32,
            'senders': tf.int32,
            'globals': tf.float32,
            }

        features_description = dict(
            [(key+"_IN",  tf.io.FixedLenFeature([], tf.string)) for key in graphs.ALL_FIELDS] + 
            [(key+"_OUT", tf.io.FixedLenFeature([], tf.string)) for key in graphs.ALL_FIELDS])

        example = tf.io.parse_single_example(example_proto, features_description)
        input_dd = graphs.GraphsTuple(**dict([(key, tf.io.parse_tensor(example[key+"_IN"], graph_types[key]))
            for key in graphs.ALL_FIELDS]))
        out_dd = graphs.GraphsTuple(**dict([(key, tf.io.parse_tensor(example[key+"_OUT"], graph_types[key]))
            for key in graphs.ALL_FIELDS]))
        return input_dd, out_dd


    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def serialize_graph(G1, G2):
        feature = {}
        for key in graphs.ALL_FIELDS:
            feature[key+"_IN"] = tf.train.Feature(bytes_list=tf.train.BytesList((tf.io.serialize_tensor(getattr(G1, key)))))
            feature[key+"_OUT"] = tf.train.Feature(bytes_list=tf.train.BytesList((tf.io.serialize_tensor(getattr(G2, key)))))
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()


    def specs_from_graphs_tuple(
        graphs_tuple_sample, with_batch_dim=False,
        dynamic_num_graphs=False,
        dynamic_num_nodes=True,
        dynamic_num_edges=True,
        description_fn=tf.TensorSpec,
        ):
        graphs_tuple_description_fields = {}
        edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]

        for field_name in graphs.ALL_FIELDS:
            field_sample = getattr(graphs_tuple_sample, field_name)
            if field_sample is None:
                raise ValueError(
                    "The `GraphsTuple` field `{}` was `None`. All fields of the "
                    "`GraphsTuple` must be specified to create valid signatures that"
                    "work with `tf.function`. This can be achieved with `input_graph = "
                    "utils_tf.set_zero_{{node,edge,global}}_features(input_graph, 0)`"
                    "to replace None's by empty features in your graph. Alternatively"
                    "`None`s can be replaced by empty lists by doing `input_graph = "
                    "input_graph.replace({{nodes,edges,globals}}=[]). To ensure "
                    "correct execution of the program, it is recommended to restore "
                    "the None's once inside of the `tf.function` by doing "
                    "`input_graph = input_graph.replace({{nodes,edges,globals}}=None)"
                    "".format(field_name))

            shape = list(field_sample.shape)
            dtype = field_sample.dtype

            # If the field is not None but has no field shape (i.e. it is a constant)
            # then we consider this to be a replaced `None`.
            # If dynamic_num_graphs, then all fields have a None first dimension.
            # If dynamic_num_nodes, then the "nodes" field needs None first dimension.
            # If dynamic_num_edges, then the "edges", "senders" and "receivers" need
            # a None first dimension.
            if shape:
                if with_batch_dim:
                    shape[1] = None
                elif (dynamic_num_graphs \
                    or (dynamic_num_nodes \
                        and field_name == graphs.NODES) \
                    or (dynamic_num_edges \
                        and field_name in edge_dim_fields)): shape[0] = None

            print(field_name, shape, dtype)
            graphs_tuple_description_fields[field_name] = description_fn(
                shape=shape, dtype=dtype)

        return graphs.GraphsTuple(**graphs_tuple_description_fields)
    
    def dtype_shape_from_graphs_tuple(input_graph, with_batch_dim=False,\
                                with_padding=True, debug=False, with_fixed_size=False):
        graphs_tuple_dtype = {}
        graphs_tuple_shape = {}

        edge_dim_fields = [graphs.EDGES, graphs.SENDERS, graphs.RECEIVERS]
        for field_name in graphs.ALL_FIELDS:
            field_sample = getattr(input_graph, field_name)
            shape = list(field_sample.shape)
            dtype = field_sample.dtype
            print(field_name, shape, dtype)

            if not with_fixed_size and shape and not with_padding:
                if with_batch_dim:
                    shape[1] = None
                else:
                    if field_name == graphs.NODES or field_name in edge_dim_fields:
                        shape[0] = None

            graphs_tuple_dtype[field_name] = dtype
            graphs_tuple_shape[field_name] = tf.TensorShape(shape)
            if debug:
                print(field_name, shape, dtype)
        
        return graphs.GraphsTuple(**graphs_tuple_dtype), graphs.GraphsTuple(**graphs_tuple_shape)

class DoubletsDataset(object):
    def __init__(self, num_workers=1, with_padding=False,
                n_graphs_per_evt=1, overwrite=False, edge_name='edge_index',
                truth_name='y'
        ):
        self.input_dtype = None
        self.input_shape = None
        self.target_dtype = None
        self.target_shape = None
        self.with_padding = False
        self.num_workers = num_workers
        self.overwrite = overwrite
        self.edge_name = edge_name
        self.truth_name = truth_name

    def make_graph(self, event, debug=False):
        """
        Convert the event into a graphs_tuple. 
        """
        edge_name = self.edge_name
        n_nodes = event['x'].shape[0]
        n_edges = event[edge_name].shape[1]
        nodes = event['x']
        edges = np.zeros((n_edges, 1), dtype=np.float32)
        senders =  event[edge_name][0, :]
        receivers = event[edge_name][1, :]
        edge_target = event[self.truth_name].numpy().astype(np.float32)
        
        input_datadict = {
            "n_node": n_nodes,
            "n_edge": n_edges,
            "nodes": nodes,
            "edges": edges,
            "senders": senders,
            "receivers": receivers,
            "globals": np.array([n_nodes], dtype=np.float32)
        }
        n_edges_target = 1
        target_datadict = {
            "n_node": 1,
            "n_edge": n_edges_target,
            "nodes": np.zeros((1, 1), dtype=np.float32),
            "edges": edge_target,
            "senders": np.zeros((n_edges_target,), dtype=np.int32),
            "receivers": np.zeros((n_edges_target,), dtype=np.int32),
            "globals": np.zeros((1,), dtype=np.float32),
        }
        input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
        target_graph = utils_tf.data_dicts_to_graphs_tuple([target_datadict])
        return [(input_graph, target_graph)]        

    def _get_signature(self, tensors):
        if self.input_dtype and self.target_dtype:
            return 

        ex_input, ex_target = tensors[0]
        self.input_dtype, self.input_shape = graph.dtype_shape_from_graphs_tuple(
            ex_input, with_padding=self.with_padding)
        self.target_dtype, self.target_shape = graph.dtype_shape_from_graphs_tuple(
            ex_target, with_padding=self.with_padding)
    

    def process(self, indir, outdir):
        files = os.listdir(indir)
        ievt = 0
        now = time.time()
        for filename in files:
            infile = os.path.join(indir, filename)
            outname = os.path.join(outdir, filename)
            if os.path.exists(outname) and not self.overwrite:
                continue
            if "npz" in infile:
                array = np.load(infile)
            else:
                import torch
                array = torch.load(infile, map_location='cpu')
            # print(array)
            tensors = self.make_graph(array)
            def generator():
                for G in tensors:
                    yield (G[0], G[1])
            self._get_signature(tensors)
            dataset = tf.data.Dataset.from_generator(
                generator, 
                output_types=(self.input_dtype, self.target_dtype),
                output_shapes=(self.input_shape, self.target_shape),
                args=None
            )

            writer = tf.io.TFRecordWriter(outname)
            for data in dataset:
                example = graph.serialize_graph(*data)
                writer.write(example)
            writer.close()
            ievt += 1

        read_time = time.time() - now
        print("{} added {:,} events, in {:.1f} mins".format(self.__class__.__name__,
            ievt, read_time/60.))
        

NUM_LAYERS = 2
LATENT_SIZE = 128
def make_mlp_model():
  """Instantiates a new MLP, followed by LayerNorm.

  The parameters of each new MLP are not shared with others generated by
  this function.

  Returns:
    A Sonnet module which contains the MLP and LayerNorm.
  """
  return snt.Sequential([
      snt.nets.MLP([128, 64],
                   activation=tf.nn.relu,
                   activate_final=True),
      # snt.LayerNorm()
  ])


class InteractionNetwork(snt.Module):
  """Implementation of an Interaction Network.

  An interaction networks computes interactions on the edges based on the
  previous edges features, and on the features of the nodes sending into those
  edges. It then updates the nodes based on the incomming updated edges.
  See https://arxiv.org/abs/1612.00222 for more details.

  This model does not update the graph globals, and they are allowed to be
  `None`.
  """

  def __init__(self,
               edge_model_fn,
               node_model_fn,
               reducer=tf.math.unsorted_segment_sum,
               name="interaction_network"):
    """Initializes the InteractionNetwork module.

    Args:
      edge_model_fn: A callable that will be passed to `EdgeBlock` to perform
        per-edge computations. The callable must return a Sonnet module (or
        equivalent; see `blocks.EdgeBlock` for details), and the shape of the
        output of this module must match the one of the input nodes, but for the
        first and last axis.
      node_model_fn: A callable that will be passed to `NodeBlock` to perform
        per-node computations. The callable must return a Sonnet module (or
        equivalent; see `blocks.NodeBlock` for details).
      reducer: Reducer to be used by NodeBlock to aggregate edges. Defaults to
        tf.unsorted_segment_sum.
      name: The module name.
    """
    super(InteractionNetwork, self).__init__(name=name)
    self._edge_block = blocks.EdgeBlock(
        edge_model_fn=edge_model_fn, use_globals=False)
    self._node_block = blocks.NodeBlock(
        node_model_fn=node_model_fn,
        use_received_edges=True,
        use_sent_edges=True,
        use_globals=False,
        received_edges_reducer=reducer)

  def __call__(self, graph):
    """Connects the InterationNetwork.

    Args:
      graph: A `graphs.GraphsTuple` containing `Tensor`s. `graph.globals` can be
        `None`. The features of each node and edge of `graph` must be
        concatenable on the last axis (i.e., the shapes of `graph.nodes` and
        `graph.edges` must match but for their first and last axis).

    Returns:
      An output `graphs.GraphsTuple` with updated edges and nodes.

    Raises:
      ValueError: If any of `graph.nodes`, `graph.edges`, `graph.receivers` or
        `graph.senders` is `None`.
    """
    return self._edge_block(self._node_block(graph))

class SegmentClassifier(snt.Module):

  def __init__(self, name="SegmentClassifier"):
    super(SegmentClassifier, self).__init__(name=name)

    self._edge_block = blocks.EdgeBlock(
        edge_model_fn=make_mlp_model,
        use_edges=False,
        use_receiver_nodes=True,
        use_sender_nodes=True,
        use_globals=False,
        name='edge_encoder_block'
    )
    self._node_encoder_block = blocks.NodeBlock(
        node_model_fn=make_mlp_model,
        use_received_edges=False,
        use_sent_edges=False,
        use_nodes=True,
        use_globals=False,
        name='node_encoder_block'
    )

    self._core = InteractionNetwork(
        edge_model_fn=make_mlp_model,
        node_model_fn=make_mlp_model,
        reducer=tf.math.unsorted_segment_sum
    )

    # Transforms the outputs into appropriate shapes.
    edge_output_size = 1
    edge_fn =lambda: snt.Sequential([
        snt.nets.MLP([edge_output_size],
                      activation=tf.nn.relu, # default is relu
                      name='edge_output'),
        tf.sigmoid])

    self._output_transform = modules.GraphIndependent(edge_fn, None, None)

  def __call__(self, input_op, num_processing_steps):
    latent = self._edge_block(self._node_encoder_block(input_op))
    latent0 = latent

    output_ops = []
    for _ in range(num_processing_steps):
        core_input = utils_tf.concat([latent0, latent], axis=1)
        latent = self._core(core_input)
        output_ops.append(self._output_transform(latent))
    return output_ops
  
def plot_metrics(odd, tdd, odd_th=0.5, tdd_th=0.5, outname='roc_graph_nets.eps',
                off_interactive=False, alternative=True):
    fontsize=16
    minor_size=14
    
    if off_interactive:
        plt.ioff()

    y_pred, y_true = (odd > odd_th), (tdd > tdd_th)
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, odd)

    if alternative:
        results = []
        labels = ['Accuracy:           ', 'Precision (purity): ', 'Recall (efficiency):']
        thresholds = [0.1, 0.5, 0.8]

        for threshold in thresholds:
            y_p, y_t = (odd > threshold), (tdd > threshold)
            accuracy  = sklearn.metrics.accuracy_score(y_t, y_p)
            precision = sklearn.metrics.precision_score(y_t, y_p)
            recall    = sklearn.metrics.recall_score(y_t, y_p)
            results.append((accuracy, precision, recall))
        
        print("GNN threshold:{:11.2f} {:7.2f} {:7.2f}".format(*thresholds))
        for idx,lab in enumerate(labels):
            print("{} {:6.4f} {:6.4f} {:6.4f}".format(lab, *[x[idx] for x in results]))

    else:
        accuracy  = sklearn.metrics.accuracy_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall    = sklearn.metrics.recall_score(y_true, y_pred)
        print('Accuracy:            %.6f' % accuracy)
        print('Precision (purity):  %.6f' % precision)
        print('Recall (efficiency): %.6f' % recall)

    auc = sklearn.metrics.auc(fpr, tpr)
    print("AUC: %.4f" % auc)
    y_p_5 = odd > 0.5
    print("Fake rejection at 0.5: {:.6f}".format(1-y_true[y_p_5 & ~y_true].shape[0]/y_true[~y_true].shape[0]))

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()
    ax0, ax1, ax2, ax3 = axs

    # Plot the model outputs
    # binning=dict(bins=50, range=(0,1), histtype='step', log=True)
    binning=dict(bins=50, histtype='step', log=True)
    ax0.hist(odd[y_true==False], lw=2, label='fake', **binning)
    ax0.hist(odd[y_true], lw=2, label='true', **binning)
    ax0.set_xlabel('Model output', fontsize=fontsize)
    ax0.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax0.legend(loc=0, fontsize=fontsize)
    ax0.set_title('ROC curve, AUC = %.4f' % auc, fontsize=fontsize)

    # Plot the ROC curve
    ax1.plot(fpr, tpr, lw=2)
    ax1.plot([0, 1], [0, 1], '--', lw=2)
    ax1.set_xlabel('False positive rate', fontsize=fontsize)
    ax1.set_ylabel('True positive rate', fontsize=fontsize)
    ax1.set_title('ROC curve, AUC = %.4f' % auc, fontsize=fontsize)
    ax1.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)


    p, r, t = sklearn.metrics.precision_recall_curve(y_true, odd)
    ax2.plot(t, p[:-1], label='purity', lw=2)
    ax2.plot(t, r[:-1], label='efficiency', lw=2)
    ax2.set_xlabel('Cut on model score', fontsize=fontsize)
    ax2.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)
    ax2.legend(fontsize=fontsize, loc='upper right')

    ax3.plot(p, r, lw=2)
    ax3.set_xlabel('Purity', fontsize=fontsize)
    ax3.set_ylabel('Efficiency', fontsize=fontsize)
    ax3.tick_params(width=2, grid_alpha=0.5, labelsize=minor_size)

    plt.savefig(outname)
    if off_interactive:
        plt.close(fig)

def plot_nx_with_edge_cmaps(G, weight_name='predict', weight_range=(0, 1),
                            alpha=1.0, ax=None,
                            cmaps=plt.get_cmap('Greys'), threshold=0.):

    def get_pos(Gp):
        pos = {}
        for node in Gp.nodes():
            r, phi, z = Gp.nodes[node]['pos'][:3]
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            pos[node] = np.array([x, y])
        return pos
    
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    pos = get_pos(G)
    res = [(edge, G.edges[edge][weight_name]) for edge in G.edges() if G.edges[edge][weight_name] > threshold]
    edges, weights = zip(*dict(res).items())

    vmin, vmax = weight_range

    nx.draw(G, pos, node_color='#A0CBE2', edge_color=weights, edge_cmap=cmaps,
            edgelist=edges, width=0.5, with_labels=False,
            node_size=1, edge_vmin=vmin, edge_vmax=vmax,
            ax=ax, arrows=False, alpha=alpha
           )
    
def np_to_nx(array):
    G = nx.Graph()

    node_features = ['r', 'phi', 'z']
    feature_scales = [1000, np.pi, 1000]

    df = pd.DataFrame(array['x']*feature_scales, columns=node_features)
    node_info = [
        (i, dict(pos=np.array(row), hit_id=array['I'][i])) for i,row in df.iterrows()
    ]
    G.add_nodes_from(node_info)

    receivers = array['receivers']
    senders = array['senders']
    score = array['score']
    truth = array['truth']
    edge_info = [
        (i, j, dict(weight=k, solution=l)) for i,j,k,l in zip(senders, receivers, score, truth)
    ]
    G.add_edges_from(edge_info)
    return G

class LayerlessEmbedding(EmbeddingBase):

    def __init__(self, hparams):
        super().__init__(hparams)
        '''
        Initialise the Lightning Module that can scan over different embedding training regimes
        '''

        # Construct the MLP architecture
        layers = [Linear(hparams["in_channels"], hparams["emb_hidden"])]
        ln = [Linear(hparams["emb_hidden"], hparams["emb_hidden"]) for _ in range(hparams["nb_layer"]-1)]
        layers.extend(ln)
        self.layers = nn.ModuleList(layers)
        self.emb_layer = nn.Linear(hparams["emb_hidden"], hparams["emb_dim"])
        self.norm = nn.LayerNorm(hparams["emb_hidden"])
        self.act = nn.Tanh()


    def forward(self, x):
#         hits = self.normalize(hits)
        for l in self.layers:
            x = l(x)
            x = self.act(x)
#         x = self.norm(x) #Option of LayerNorm
        x = self.emb_layer(x)
        return x

class EmbeddingInferenceCallback(Callback):
    def __init__(self):
        self.output_dir = None
        self.overwrite = False

    def on_train_start(self, trainer, pl_module):
        # Prep the directory to produce inference data to
        self.output_dir = pl_module.hparams.output_dir
        self.datatypes = ["train", "val", "test"]
        os.makedirs(self.output_dir, exist_ok=True)
        [os.makedirs(os.path.join(self.output_dir, datatype), exist_ok=True) for datatype in self.datatypes]

        # Set overwrite setting if it is in config
        if "overwrite" in pl_module.hparams:
            self.overwrite = pl_module.hparams.overwrite

    def on_train_end(self, trainer, pl_module):
        print("Training finished, running inference to build graphs...")

        # By default, the set of examples propagated through the pipeline will be train+val+test set
        datasets = {"train": pl_module.trainset, "val": pl_module.valset, "test": pl_module.testset}
        total_length = sum([len(dataset) for dataset in datasets.values()])
        batch_incr = 0

        pl_module.eval()
        with torch.no_grad():
            for set_idx, (datatype, dataset) in enumerate(datasets.items()):
                for batch_idx, batch in enumerate(dataset):
                    percent = (batch_incr / total_length) * 100
                    sys.stdout.flush()
                    sys.stdout.write(f'{percent:.01f}% inference complete \r')

                    # print(not os.path.exists(os.path.join(self.output_dir, datatype, batch.event_file[-4:])))
                    # print(self.overwrite)
                    #
                    # print(os.path.join(self.output_dir, datatype, batch.event_file[-4:]))
                    if (not os.path.exists(os.path.join(self.output_dir, datatype, batch.event_file[-4:]))) or self.overwrite:
                        batch = batch.to(pl_module.device) #Is this step necessary??
                        batch = self.construct_downstream(batch, pl_module)
                        self.save_downstream(batch, pl_module, datatype)
                        del batch
                        torch.cuda.empty_cache()

                    batch_incr += 1

    def construct_downstream(self, batch, pl_module):

        if 'ci' in pl_module.hparams["regime"]:
            spatial = pl_module(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = pl_module(batch.x)

        e_bidir = torch.cat([batch.layerless_true_edges,
                       torch.stack([batch.layerless_true_edges[1], batch.layerless_true_edges[0]], axis=1).T], axis=-1)

        # This step should remove reliance on r_val, 
        clustering = getattr(utils_torch, pl_module.hparams.clustering)
        # and instead compute an r_build based on the EXACT r required to reach target eff/pur
        e_spatial = clustering(spatial, pl_module.hparams.r_val, pl_module.hparams.knn_val)
        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        # remove edges that point from outter region to inner region
        R_dist = torch.sqrt(batch.x[:,0]**2 + batch.x[:,2]**2) # distance away from origin...
        sel_idx = R_dist[e_spatial[0]] <= R_dist[e_spatial[1]]
        
        batch.e_radius = e_spatial[:, sel_idx]
        batch.y = torch.from_numpy(y_cluster).float()[sel_idx]

        return batch

    def save_downstream(self, batch, pl_module, datatype):

        with open(os.path.join(self.output_dir, datatype, batch.event_file[-4:]), 'wb') as pickle_file:
            torch.save(batch, pickle_file)

def load_datasets(input_dir, train_split, seed = 0):
    '''
    Prepare the random Train, Val, Test split, using a seed for reproducibility. Seed should be
    changed across final varied runs, but can be left as default for experimentation.
    '''
    torch.manual_seed(seed)
    all_events = os.listdir(input_dir)
    all_events = sorted([os.path.join(input_dir, event) for event in all_events])
    loaded_events = [torch.load(event, map_location='cpu') for event in all_events[:sum(train_split)]]
    train_events, val_events, test_events = random_split(loaded_events, train_split)

    return train_events, val_events, test_events

class EmbeddingBase(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        '''
        Initialise the Lightning Module that can scan over different embedding training regimes
        '''
        # Assign hyperparameters
        self.hparams = hparams
        self.hparams['input_dir'] = utils_dir.feature_outdir
        self.hparams['output_dir'] = utils_dir.embedding_outdir
        self.clustering = getattr(utils_torch, hparams['clustering'])

    def setup(self, stage):
        self.trainset, self.valset, self.testset = load_datasets(self.hparams["input_dir"], self.hparams["train_split"])


    def train_dataloader(self):
        if len(self.trainset) > 0:
            return DataLoader(self.trainset, batch_size=1, num_workers=self.hparams['n_workers'])
        else:
            return None

    def val_dataloader(self):
        if len(self.valset) > 0:
            return DataLoader(self.valset, batch_size=1, num_workers=self.hparams['n_workers'])
        else:
            return None

    def test_dataloader(self):
        if len(self.testset):
            return DataLoader(self.testset, batch_size=1, num_workers=self.hparams['n_workers'])
        else:
            return None

    def configure_optimizers(self):
        optimizer = [torch.optim.AdamW(self.parameters(), lr=(self.hparams["lr"]), betas=(0.9, 0.999), eps=1e-08, amsgrad=True)]
        scheduler = [
            {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer[0],\
                                                    factor=self.hparams["factor"], patience=self.hparams["patience"]),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        ]
#         scheduler = [torch.optim.lr_scheduler.StepLR(optimizer[0], step_size=1, gamma=0.3)]
        return optimizer, scheduler

    def training_step(self, batch, batch_idx):

        # apply the embedding neural network on the hit features
        # and return hidden features in the embedding space.
        if 'ci' in self.hparams["regime"]:
            spatial = self(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = self(batch.x)

        # create another direction for true doublets
        e_bidir = torch.cat([batch.layerless_true_edges,
                            torch.stack([batch.layerless_true_edges[1],
                                        batch.layerless_true_edges[0]], axis=1).T
                            ], axis=-1)

        # construct doublets for training
        e_spatial = torch.empty([2,0], dtype=torch.int64, device=self.device)

        if 'rp' in self.hparams["regime"]:
            # randomly select two times of total true edges
            n_random = int(self.hparams["randomisation"]*e_bidir.shape[1])
            e_spatial = torch.cat([e_spatial,
                torch.randint(e_bidir.min(), e_bidir.max(), (2, n_random), device=self.device)], axis=-1)

        # use a clustering algorithm to connect hits based on the embedding information
        # euclidean distance is used. 
        if 'hnm' in self.hparams["regime"]:
            e_spatial = torch.cat([e_spatial,
                            self.clustering(spatial, self.hparams["r_train"], self.hparams["knn_train"])], axis=-1)
            # e_spatial = torch.cat([e_spatial, 
            #         build_edges(spatial, self.hparams["r_train"], self.hparams["knn"], res)],
            #         axis=-1)
            # e_spatial = torch.cat([e_spatial,
            #                 radius_graph(spatial, r=self.hparams["r_train"], max_num_neighbors=self.hparams["knn"])], axis=-1)

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        # add all truth edges four times
        # in order to balance the number of truth and fake edges in one batch
        e_spatial = torch.cat([
            e_spatial,
            e_bidir.transpose(0,1).repeat(1,self.hparams["weight"]).view(-1, 2).transpose(0,1)
            ], axis=-1)
        y_cluster = np.concatenate([y_cluster.astype(int), np.ones(e_bidir.shape[1]*self.hparams["weight"])])

        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        # euclidean distances in the embedding space between two hits
        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)

        loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=self.hparams["margin"], reduction="mean")

        self.log("train_loss", loss, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):

        if 'ci' in self.hparams["regime"]:
            spatial = self(torch.cat([batch.cell_data, batch.x], axis=-1))
        else:
            spatial = self(batch.x)

        e_bidir = torch.cat([batch.layerless_true_edges,
                               torch.stack([batch.layerless_true_edges[1], batch.layerless_true_edges[0]], axis=1).T], axis=-1)

        # use a clustering algorithm to connect hits based on the embedding information
        # euclidean distance is used. 
        e_spatial = self.clustering(spatial, self.hparams["r_val"], self.hparams["knn_val"])

        e_spatial, y_cluster = graph_intersection(e_spatial, e_bidir)

        hinge = torch.from_numpy(y_cluster).float().to(device)
        hinge[hinge == 0] = -1

        reference = spatial.index_select(0, e_spatial[1])
        neighbors = spatial.index_select(0, e_spatial[0])
        d = torch.sum((reference - neighbors)**2, dim=-1)

        val_loss = torch.nn.functional.hinge_embedding_loss(d, hinge, margin=self.hparams["margin"], reduction="mean")

        self.log("val_loss", val_loss, prog_bar=True)

        cluster_true = 2*len(batch.layerless_true_edges[0])
        cluster_true_positive = y_cluster.sum()
        cluster_positive = len(e_spatial[0])

        self.log_dict({
            'val_eff': torch.tensor(cluster_true_positive/cluster_true),
            'val_pur': torch.tensor(cluster_true_positive/cluster_positive)}, prog_bar=True)


    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_idx,
                    second_order_closure=None, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if (self.hparams["warmup"] is not None) and (self.trainer.global_step < self.hparams["warmup"]):
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams["warmup"])
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams["lr"]

        # update params
        optimizer.step()
        optimizer.zero_grad()