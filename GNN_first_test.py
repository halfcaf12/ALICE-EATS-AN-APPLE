import sys
import keras
import tensorflow as tf
import tensorflow_gnn as tfgnn
from visualizeData import loadFromNPZ
import collections
import functools
import itertools
from typing import Callable, Optional, Mapping, Tuple

# getting familiar with tensorflow pipelines
# store clusters data above directory: change this name for where you 
# store your clusters data from
clusters = loadFromNPZ("../clusters")

inputs = keras.Input(shape=(37,))
x = keras.layers.Dense(32, activation="relu")(inputs)
outputs = keras.layers.Dense(5, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)


# IMPLEMENTING SSSP with GRAPH NETWORKIS

def GraphNetworkGraphUpdate(
    *,
    edges_next_state_factory: Callable[..., tf.keras.layers.Layer],
    nodes_next_state_factory: Callable[..., tf.keras.layers.Layer],
    context_next_state_factory: Optional[Callable[..., tf.keras.layers.Layer]],
    receiver_tag: Optional[tfgnn.IncidentNodeTag] = tfgnn.TARGET,
    reduce_type_to_nodes: str = "sum",
    reduce_type_to_context: str = "sum",
    use_input_context_state: bool = True,
    name: str = "graph_network"):
  """Returns a GraphUpdate to run a GraphNetwork on all node sets and edge sets.

  The returned layer implements a Graph Network, as described by
  Battaglia et al.: ["Relational inductive biases, deep learning, and
  graph networks"](https://arxiv.org/abs/1806.01261), 2018, generalized
  to heterogeneous graphs.

  It expects an input GraphTensor with a `tfgnn.HIDDEN_STATE` feature on all
  node sets and edge sets, and also context if `use_input_context_state=True`.
  It runs edge, node, and context updates, in this order, separately for each
  edge set, node set (regardless whether it has an incoming edge set), and also
  context if `context_next_state_factory` is set. Finally, it returns a
  GraphTensor with updated hidden states, incl. a context state, if
  `context_next_state_factory` is set.

  The model can also behave as an Interaction Network ([Battaglia et al., NIPS
  2016](https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html))
  by setting
    * `use_input_context_state = False`
    * `context_next_state_factory = None`

  Args:
    edges_next_state_factory: Called with keyword argument `edge_set_name=`
      for each edge set to return the NextState layer for use in the respective
      `tfgnn.keras.layers.EdgeSetUpdate`.
    nodes_next_state_factory: Called with keyword argument `node_set_name=`
      for each node set to return the NextState layer for use in the respective
      `tfgnn.keras.layers.NodeSetUpdate`.
    context_next_state_factory: If set, a `tfgnn.keras.layers.ContextUpdate`
      is included with the NextState layer returned by calling this factory.
    receiver_tag: The incident node tag at which each edge set is used to
      update node sets. Defaults to `tfgnn.TARGET`.
    reduce_type_to_nodes: Controls how incident edges at a node are aggregated
      within each EdgeSet. Defaults to `"sum"`. (The aggregates of the various
      incident EdgeSets are then concatenated.)
    reduce_type_to_context: Controls how the nodes of a NodeSet or the edges of
      an EdgeSet are aggregated for the context update. Defaults to `"sum"`.
      (The aggregates of the various NodeSets/EdgeSets are then concatenated.)
    use_input_context_state: If true, the input `GraphTensor.context` must have
      a `tfgnn.HIDDEN_STATE` feature that gets used as input in all edge, node
      and context updates.
    name: A name for the returned layer.
  """
  def deferred_init_callback(graph_spec):
    context_input_feature = (
        tfgnn.HIDDEN_STATE if use_input_context_state else None)

    # To keep track node types that receive each edge type.
    incoming_edge_sets = collections.defaultdict(list)

    # For every edge set, create an EdgeSetUpdate.
    edge_set_updates = {}
    for edge_set_name in sorted(graph_spec.edge_sets_spec.keys()):
      next_state = edges_next_state_factory(edge_set_name=edge_set_name)
      edge_set_updates[edge_set_name] = tfgnn.keras.layers.EdgeSetUpdate(
          next_state=next_state,
          edge_input_feature=tfgnn.HIDDEN_STATE,
          node_input_feature=tfgnn.HIDDEN_STATE,
          context_input_feature=context_input_feature)
      # Keep track of which node set is the receiver for this edge type
      # as we will need it later.
      target_name = graph_spec.edge_sets_spec[
          edge_set_name].adjacency_spec.node_set_name(receiver_tag)
      incoming_edge_sets[target_name].append(edge_set_name)

    # For every node set, create a NodeSetUpdate.
    node_set_updates = {}
    for node_set_name in sorted(graph_spec.node_sets_spec.keys()):
      # Apply a node update, after summing *all* of the received edges
      # for that node set.
      next_state = nodes_next_state_factory(node_set_name=node_set_name)
      node_set_updates[node_set_name] = tfgnn.keras.layers.NodeSetUpdate(
          next_state=next_state,
          edge_set_inputs={
              edge_set_name: tfgnn.keras.layers.Pool(
                  receiver_tag, reduce_type_to_nodes,
                  feature_name=tfgnn.HIDDEN_STATE)
              for edge_set_name in incoming_edge_sets[node_set_name]},
          node_input_feature=tfgnn.HIDDEN_STATE,
          context_input_feature=context_input_feature)

    # Create a ContextUpdate, if requested.
    context_update = None
    if context_next_state_factory is not None:
      next_state = context_next_state_factory()
      context_update = tfgnn.keras.layers.ContextUpdate(
          next_state=next_state,
          edge_set_inputs={
              edge_set_name: tfgnn.keras.layers.Pool(
                  tfgnn.CONTEXT, reduce_type_to_context,
                  feature_name=tfgnn.HIDDEN_STATE)
              for edge_set_name in sorted(graph_spec.edge_sets_spec.keys())},
          node_set_inputs={
              node_set_name: tfgnn.keras.layers.Pool(
                  tfgnn.CONTEXT, reduce_type_to_context,
                  feature_name=tfgnn.HIDDEN_STATE)
              for node_set_name in sorted(graph_spec.node_sets_spec.keys())},
          context_input_feature=context_input_feature)
    return dict(edge_sets=edge_set_updates,
                node_sets=node_set_updates,
                context=context_update)

  return tfgnn.keras.layers.GraphUpdate(
      deferred_init_callback=deferred_init_callback, name=name)