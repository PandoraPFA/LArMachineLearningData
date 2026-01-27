import numpy as np
from collections import defaultdict

import torch

from sklearn.cluster import AgglomerativeClustering, AffinityPropagation
import networkx as nx
import igraph as ig
import leidenalg

from data.event import SuperCluster

def affinity_clustering(t_sim, clusters):
    if len(clusters) == 1:
        return np.array([ 0 for _ in clusters[0].hits ])

    arr_sim = t_sim.detach().cpu().numpy()

    clustering = AffinityPropagation(
        affinity="precomputed", damping=0.9, preference=None, random_state=1
    )
    cluster_labels = clustering.fit_predict(arr_sim)

    return cluster_labels

def leiden_clustering(t_sim, clusters, resolution=1.0, min_weight=0.0):
    arr_sim = t_sim.detach().cpu().numpy()
    N = arr_sim.shape[0]

    sources, targets = np.where(np.triu(arr_sim, 1) > min_weight)
    weights = arr_sim[sources, targets]

    g = ig.Graph(directed=False)
    g.add_vertices(N)
    g.add_edges(list(zip(sources, targets)))
    g.es["weight"] = weights.tolist()

    partition = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"], resolution_parameter=resolution
    )
    cluster_labels = np.array(partition.membership)

    return cluster_labels

def agglomerative_clustering(t_sim, clusters, sim_threshold):
    if len(clusters) == 1: # AgglomerativeClustering throws in this case
        return np.array([ 0 for _ in clusters[0].hits ])

    arr_dist = 1 - t_sim.detach().cpu().numpy()

    clustering = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=sim_threshold,
        n_clusters=None
    )
    cluster_labels = clustering.fit_predict(arr_dist)

    return cluster_labels

def connected_accessory_clustering(t_sim, clusters, sim_threshold, n_hits_fn=None):
    t_sim = t_sim.detach().cpu()
    t_sim_zerodiag = t_sim.clone()
    t_sim_zerodiag.fill_diagonal_(0)

    # no cross similarities are over threshold -> guranteed no merges -> skip computations
    if len(clusters) <= 1 or (torch.max(t_sim_zerodiag) < sim_threshold):
        return [ i for i in range(len(clusters)) ]

    # clusters might be Cluster objects or tensors
    if n_hits_fn is None:
        n_hits_fn = lambda cluster: cluster.get_n_hits()

    # "Core clusters" should be able to bridge superclusters by their adjacency
    core_idxs = [ i for i, cluster in enumerate(clusters) if n_hits_fn(cluster) > 2 ]
    # "Accessory clusters" should not be able to bridge superclusters made from core clusters.
    # They should be mopped up into core clusters, with each other, or left alone
    acc_idxs = [ i for i, cluster in enumerate(clusters) if n_hits_fn(cluster) <= 2 ]

    # Connected clustering for core clusters
    adj = (t_sim > sim_threshold).numpy()
    core_adj = adj[np.ix_(core_idxs, core_idxs)]
    G_core = nx.from_numpy_array(core_adj)
    core_groups = [
        [core_idxs[i] for i in conc_comps] for conc_comps in nx.connected_components(G_core)
    ]

    # Handle accessory clusters separately
    unmerged_acc_idxs = set(acc_idxs)
    candidate_unmerged_acc_idxs = {
        i_acc
        for i_acc in unmerged_acc_idxs if t_sim_zerodiag[i_acc, :].max().item() > sim_threshold
    }
    candidate_core_group_labels = set(range(len(core_groups)))
    planned_merges = []
    # Iteratively merge accessories into cores based on max similarity to core constituents
    while True:
        planned_merges = []
        for i_acc in candidate_unmerged_acc_idxs:
            best_label = None
            best_sim = -1.
            for label in candidate_core_group_labels:
                max_sim = max(
                    t_sim[i_acc, i_core_group].item() for i_core_group in core_groups[label]
                )
                if max_sim > sim_threshold and max_sim > best_sim:
                    best_sim = max_sim
                    best_label = label
            if best_label is not None:
                planned_merges.append((i_acc, best_label))
        if not len(planned_merges):
            break

        # Update core clusters from this iteration
        candidate_core_group_labels = set()
        for i_acc, label in planned_merges:
            core_groups[label].append(i_acc)
            unmerged_acc_idxs.remove(i_acc)
            candidate_unmerged_acc_idxs.remove(i_acc)
            candidate_core_group_labels.add(label)

    # Convert core cluster merge groups to per-cluster label
    cluster_labels = [ None for _ in range(len(clusters)) ]
    for label, group in enumerate(core_groups):
        for i in group:
            cluster_labels[i] = label

    # The remaining accessories have low similarity to the cores and any of the accessories that
    # were merged into the cores. Give them a chance to merge with themselves.
    if unmerged_acc_idxs:
        unmerged_acc_idxs = list(unmerged_acc_idxs)
        acc_adj = adj[np.ix_(unmerged_acc_idxs, unmerged_acc_idxs)]
        G_acc = nx.from_numpy_array(acc_adj)
        acc_groups = [
            [unmerged_acc_idxs[i] for i in conc_comps]
            for conc_comps in nx.connected_components(G_acc)
        ]
        # Add to the per-cluster labels for the new accessory groups
        next_label = len(core_groups)
        for group in acc_groups:
            for i in group:
                cluster_labels[i] = next_label
            next_label += 1

    return cluster_labels

def connected_accessory_clustering_2stage(t_sim, clusters, sim_threshold, sim_threshold_stage2):
    cluster_labels_stage1 = connected_accessory_clustering(t_sim, clusters, sim_threshold)

    if sim_threshold_stage2 >= sim_threshold: # Stage 2 is identity in this case
        return cluster_labels_stage1

    # Construct super clusters
    next_id = max(cluster.id for cluster in clusters) + 1
    label_to_supercluster = {}
    for cluster, label in zip(clusters, cluster_labels_stage1):
        if label not in label_to_supercluster:
            label_to_supercluster[label] = SuperCluster(next_id, cluster.view)
            next_id += 1
        label_to_supercluster[label].add_cluster(cluster)
    superclusters = [ supercluster for supercluster in label_to_supercluster.values() ]

    # Calculate super cluster similarities from minimum of all constituent cluster pairs
    cluster_id_to_idx = { cluster.id : i for i, cluster in enumerate(clusters) }
    t_super_sim = torch.zeros(len(superclusters), len(superclusters), dtype=t_sim.dtype)
    for i_a, supercluster_a in enumerate(superclusters):
        cluster_idxs_a = torch.tensor(
            [ cluster_id_to_idx[id] for id in supercluster_a.constituent_clusters_ids ]
        )
        for i_b, supercluster_b in enumerate(superclusters):
            if (i_b == i_a):
                t_super_sim[i_a, i_b] = 1.
                continue
            cluster_idxs_b = torch.tensor(
                [ cluster_id_to_idx[id] for id in supercluster_b.constituent_clusters_ids ]
            )
            t_super_sim[i_a, i_b] = torch.min(t_sim[cluster_idxs_a][:, cluster_idxs_b])

    # Do clustering again on new super clusters and their similarities with a reduced threshold
    supercluster_labels = connected_accessory_clustering(
        t_super_sim, superclusters, sim_threshold_stage2
    )

    # Convert super cluster labels to original cluster labels
    cluster_labels = [ None for _ in range(len(cluster_labels_stage1)) ]
    for label, supercluster in zip(supercluster_labels, superclusters):
        for cluster in supercluster.constituent_clusters:
            cluster_labels[cluster_id_to_idx[cluster.id]] = label

    return cluster_labels

def connected_clustering(t_sim, clusters, sim_threshold):
    t_adj = (t_sim > sim_threshold)
    G = nx.from_numpy_array(t_adj.detach().cpu().numpy())
    merge_groups = [ list(c) for c in nx.connected_components(G) ]

    cluster_labels = [ None for _ in range(len(clusters)) ]
    for i_label, group in enumerate(merge_groups):
        for i_cluster in group:
            cluster_labels[i_cluster] = i_label

    return cluster_labels
