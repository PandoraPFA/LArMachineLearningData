import itertools
from collections import defaultdict

""" Start - data reading helper classes """

class Event:
    def __init__(
        self,
        cluster_ids, cluster_views,
        mc_ids, mc_pdgs, mc_is_from_beam,
        hit_cluster_ids, hit_mc_ids,
        hit_xs, hit_zs, hit_rs, hit_c_thetas, hit_s_thetas,
        hit_x_widths,
        hit_x_gap_dists,
        hit_energies,
        hit_tpc_vol_ids, hit_daughter_vol_ids
    ):
        self.mcs = {
            id : MCParticle(pdg, is_from_beam)
            for id, pdg, is_from_beam in zip(mc_ids, mc_pdgs, mc_is_from_beam)
        }

        if hit_tpc_vol_ids is None:
            hit_tpc_vol_ids = itertools.repeat(None)
        if hit_daughter_vol_ids is None:
            hit_daughter_vol_ids = itertools.repeat(None)
        hits = defaultdict(list)
        for (
            x, z, r, c_theta, s_theta,
            x_width,
            x_gap_dist,
            energy,
            mc_id, cluster_id,
            tpc_vol_id, daughter_vol_id
        ) in zip(
            hit_xs, hit_zs, hit_rs, hit_c_thetas, hit_s_thetas,
            hit_x_widths,
            hit_x_gap_dists,
            hit_energies,
            hit_mc_ids, hit_cluster_ids,
            hit_tpc_vol_ids, hit_daughter_vol_ids
        ):
            hit = Hit(
                x, z, r, c_theta, s_theta, x_width, x_gap_dist, energy, tpc_vol_id, daughter_vol_id
            )
            hit.add_main_mc(mc_id, self.mcs[mc_id])
            hits[cluster_id].append(hit)

        self.view_clusters = defaultdict(list)
        self.beam_purity = 0
        for id, view in zip(cluster_ids, cluster_views):
            cluster = Cluster(id, view)
            for hit in hits[id]:
                cluster.add_hit(hit)
                self.beam_purity += hit.main_mc_is_from_beam
            self.view_clusters[view].append(cluster)
        self.beam_purity /= sum(
            cluster.get_n_hits(with_mc_id=True)
            for clusters in self.view_clusters.values()
                for cluster in clusters
        )

    def get_n_hits(self, view):
        return sum(len(cluster.hits) for cluster in self.view_clusters[view])

    def get_n_clusters(self, view):
        return len(self.view_clusters[view])

class Cluster:
    def __init__(self, id, view):
        self.id = id
        self.view = view
        self.hits = []
        self.main_mc_id = None
        self.mc_id_cnt = defaultdict(int)
        self.n_hits_missing_mc_id = 0

    def add_hit(self, hit):
        self.hits.append(hit)
        self.main_mc_id = None
        self.mc_id_cnt[hit.main_mc_id] += 1
        if hit.main_mc_id < 0:
            self.n_hits_missing_mc_id += 1

    def calc_main_mc(self):
        if not self.get_n_hits(with_mc_id=True):
            return
        self.main_mc_id = max(
            [ k for k in self.mc_id_cnt.keys() if k >= 0 ], key=lambda k: self.mc_id_cnt[k]
        )

    def get_n_hits(self, with_mc_id=False):
        if not with_mc_id:
            return len(self.hits)
        return len(self.hits) - self.n_hits_missing_mc_id

class SuperCluster(Cluster):
    def __init__(self, id, view, constituent_clusters=None):
        super().__init__(id, view)

        self.constituent_clusters = []
        self.constituent_clusters_ids = set()
        if constituent_clusters is not None:
            for cluster in constituent_clusters:
                self.add_cluster(cluster)
    
    def add_cluster(self, cluster):
        if cluster.view != self.view:
            raise ValueError(
                f"Cluster and SuperCluster views mismatched: {cluster.view} vs. {self.view}"
            )
        self.constituent_clusters_ids.add(cluster.id)
        self.constituent_clusters.append(cluster)
        for hit in cluster.hits:
            self.add_hit(hit)

class Hit:
    def __init__(
        self,
        x, z, r, c_theta, s_theta,
        x_width,
        x_gap_dist,
        energy,
        tpc_vol_id, daughter_vol_id
    ):
        self.x = x
        self.z = z
        self.r = r
        self.c_theta = c_theta
        self.s_theta = s_theta
        self.x_width = x_width
        self.x_gap_dist = x_gap_dist
        self.energy = energy
        self.vol_id = (tpc_vol_id, daughter_vol_id)
        self.main_mc_id = None
        self.main_mc_pdg = None
        self.main_mc_is_from_beam = None

    def add_main_mc(self, id, mcp):
        self.main_mc_id = id
        self.main_mc_pdg = mcp.pdg
        self.main_mc_is_from_beam = mcp.is_from_beam

    def has_vol_id(self):
        return all(id is not None for id in self.vol_id)

class MCParticle:
    def __init__(self, pdg, is_from_beam):
        self.pdg = pdg
        self.is_from_beam = is_from_beam

""" End - data reading helper classes """
