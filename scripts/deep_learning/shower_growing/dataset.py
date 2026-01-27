import glob, os
from collections import Counter

import logging; logger = logging.getLogger("the_logger")

import torch; import torch.nn as nn

from helpers import scale_cluster_tensor_inplace, add_cardinality_feature, add_aug_tier_feature

""" Start - dataset and loader for cluster similarity prediction """

class ClusterDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, conf, max_events=None):
        self.file_paths = glob.glob(os.path.join(data_path, "*.pt"))[:max_events]

        self.hit_feat_scaling_factors = conf.hit_feat_scaling_factors
        self.hit_feat_log_transforms = conf.hit_feat_log_transforms
        self.hit_feat_vec_dim = conf.hit_feat_vec_dim

        self.hit_feat_add_cardinality = conf.hit_feat_add_cardinality
        self.hit_feat_add_aug_tier = conf.hit_feat_add_aug_tier
        self.hit_feat_add_on = True # Training stable only if toggled off for initial epochs

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        inputs, target = self._read_tensor(idx)
        inputs, target = self._process_tensor(inputs, target)
        return { "input" : inputs, "target" : target }

    def _read_tensor(self, idx):
        t_data = torch.load(self.file_paths[idx])
        return t_data[:-1], t_data[-1]

    def _process_tensor(self, inputs, target):
        if self.hit_feat_add_cardinality:
            inputs = add_cardinality_feature(inputs, active=self.hit_feat_add_on)
        if self.hit_feat_add_aug_tier:
            inputs = add_aug_tier_feature(inputs, 0, active=self.hit_feat_add_on)
        for t in inputs:
            scale_cluster_tensor_inplace(
                t, self.hit_feat_scaling_factors, self.hit_feat_log_transforms
            )
        return inputs, target

class ClusterMCCntsDataset(ClusterDataset):
    def __getitem__(self, idx):
        inputs, target, mc_id_cnts  = self._read_tensor(idx)
        inputs, target = self._process_tensor(inputs, target)
        return { "input" : inputs, "target" : target, "mc_id_cnts" : mc_id_cnts }

    def _read_tensor(self, idx):
        t_data = torch.load(self.file_paths[idx])
        mc_id_cnts = [
            Counter({ id : cnt for id, cnt in zip(ids.tolist(), cnts.tolist()) })
            for ids, cnts in zip(t_data["cluster_mc_ids"], t_data["cluster_mc_cnts"])
        ]
        return t_data["clusters"], t_data["similarity"], mc_id_cnts

class CollateClusters:
    def __init__(self, conf):
        self.bucket_boundaries = torch.tensor(conf.bucket_boundaries)
        self.n_buckets = len(self.bucket_boundaries) + 1

        self.include_cluster_mc_id_cnt = conf.aug_params["iterative_augs"]

    def __call__(self, data_list):
        t_clusters, t_cluster_ev_idxs = [], []
        for i_ev, ev_data in enumerate(data_list):
            for i_cluster, t_cluster in enumerate(ev_data["input"]):
                t_clusters.append(t_cluster)
                t_cluster_ev_idxs.append((i_ev, i_cluster))
        bucket_idxs = torch.bucketize(
            torch.tensor([ t_cluster.size(0) for t_cluster in t_clusters]),
            self.bucket_boundaries
        )
    
        randomiser = torch.randperm(self.n_buckets)
        chunked_t_clusters = [ [] for _ in range(self.n_buckets) ]
        chunked_t_cluster_lens = [ [] for _ in range(self.n_buckets) ]
        chunked_t_cluster_ev_idxs = [ [] for _ in range(self.n_buckets) ]
        for t_cluster, ev_idx, bucket_idx in zip(t_clusters, t_cluster_ev_idxs, bucket_idxs):
            chunked_t_clusters[randomiser[bucket_idx].item()].append(t_cluster)
            chunked_t_cluster_lens[randomiser[bucket_idx].item()].append(t_cluster.size(0))
            chunked_t_cluster_ev_idxs[randomiser[bucket_idx].item()].append(ev_idx)
        chunked_t_clusters = [ lst for lst in chunked_t_clusters if lst ]
        chunked_t_cluster_lens  = [ lst for lst in chunked_t_cluster_lens if lst ]
        chunked_t_cluster_ev_idxs  = [ lst for lst in chunked_t_cluster_ev_idxs if lst ]

        chunked_t_clusters = [
            nn.utils.rnn.pad_sequence(ts, batch_first=True) for ts in chunked_t_clusters
        ]
        chunked_t_clusters_padding_mask = []
        for t_cluster_lens_chunk, t_clusters_chunk in zip(chunked_t_cluster_lens, chunked_t_clusters):
            t_cluster_lens = torch.tensor(t_cluster_lens_chunk)
            if torch.all(t_cluster_lens == t_cluster_lens[0]):
                chunked_t_clusters_padding_mask.append(None)
                continue
            n_clusters, max_cluster_len = t_clusters_chunk.size(0), t_clusters_chunk.size(1)
            chunked_t_clusters_padding_mask.append(
                torch.arange(max_cluster_len).expand(n_clusters, max_cluster_len) >=
                t_cluster_lens.unsqueeze(1)
            )
        
        ev_target = [ event_data["target"].unsqueeze(0) for event_data in data_list ]

        ev_cardinalities = [
            torch.tensor(
                [float(t_cluster.size(0)) for t_cluster in event_data["input"]]
            ).unsqueeze(0)
            for event_data in data_list
        ]

        ret = {
            "chunked_input" : chunked_t_clusters,
            "chunked_input_mask" : chunked_t_clusters_padding_mask,
            "chunked_input_ev_idxs" : chunked_t_cluster_ev_idxs,
            "ev_target" : ev_target,
            "ev_cardinalities" : ev_cardinalities
        }

        if self.include_cluster_mc_id_cnt:
            ret["clusters"] = [ event_data["input"] for event_data in data_list ]
            ret["mc_id_cnts"] = [ event_data["mc_id_cnts"] for event_data in data_list ]

        return ret

""" End - dataset and loader for cluster similarity prediction """
