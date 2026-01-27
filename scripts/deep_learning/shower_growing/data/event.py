from collections import defaultdict

""" Start - data reading helper classes """

class Event:
    def __init__(
        self,
        cluster_ids, cluster_views,
        mc_ids, mc_pdgs,
        hit_cluster_ids, hit_mc_ids,
        hit_xs, hit_zs, hit_rs, hit_c_thetas, hit_s_thetas,
        hit_x_widths,
        hit_x_gap_dists,
        hit_energies
    ):
        self.mcs = { id : pdg for id, pdg in zip(mc_ids, mc_pdgs) }

        hits = defaultdict(list)
        for (
            x, z, r, c_theta, s_theta,
            x_width,
            x_gap_dist,
            energy,
            mc_id, cluster_id
        ) in zip(
            hit_xs, hit_zs, hit_rs, hit_c_thetas, hit_s_thetas,
            hit_x_widths,
            hit_x_gap_dists,
            hit_energies,
            hit_mc_ids, hit_cluster_ids
        ):
            hit = Hit(x, z, r, c_theta, s_theta, x_width, x_gap_dist, energy)
            hit.add_main_mc(mc_id, self.mcs[mc_id])
            hits[cluster_id].append(hit)

        self.view_clusters = defaultdict(list)
        for id, view in zip(cluster_ids, cluster_views):
            cluster = Cluster(id, view)
            for hit in hits[id]:
                cluster.add_hit(hit)
            self.view_clusters[view].append(cluster)

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

    def add_hit(self, hit):
        self.hits.append(hit)
        self.main_mc_id = None
        if hit.main_mc_id != -1:
            self.mc_id_cnt[hit.main_mc_id] += 1

    def calc_main_mc(self):
        if not len(self.mc_id_cnt):
            return
        self.main_mc_id = max(self.mc_id_cnt.keys(), key=lambda k: self.mc_id_cnt[k])

    def get_n_hits(self):
        return len(self.hits)

class SuperCluster(Cluster):
    def __init__(self, id, view, constituent_clusters=None):
        super().__init__(id, view)

        self.constituent_clusters = []
        self.constituent_clusters_ids = set()
        if constituent_clusters is not None:
            for cluster in self.constituent_clusters:
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
    def __init__(self, x, z, r, c_theta, s_theta, x_width, x_gap_dist, energy):
        self.x = x
        self.z = z
        self.r = r
        self.c_theta = c_theta
        self.s_theta = s_theta
        self.x_width = x_width
        self.x_gap_dist = x_gap_dist
        self.energy = energy
        self.main_mc_id = None
        self.main_mc_pdg = None

    def add_main_mc(self, id, pdg):
        self.main_mc_id = id
        self.main_mc_pdg = pdg

""" End - data reading helper classes """
