class InputTrack():
    def __init__(self, u_view_x, u_view_wire, u_view_q, u_view_n_hits, 
                       v_view_x, v_view_wire, v_view_q, v_view_n_hits, 
                       w_view_x, w_view_wire, w_view_q, w_view_n_hits, 
                       n_child_trk, n_child_shw, n_descendants, n_descendant_hits, reco_tier, pdg):
        """
            Simple class to store the required information about tracks

            Args:
                u_view_x: the x coordinate of all the hits in the u view of the track
                u_view_wire: the wire coordinate of all the hits in the u view of track
                u_view_q: the charge of all the hits in the u view of the track
                u_view_n_hits: the number of hits in the u view of the track
                v_view_x: the x coordinate of all the hits in the v view of the track
                v_view_wire: the wire coordinate of all the hits in the v view of track
                v_view_q: the charge of all the hits in the v view of the track
                v_view_n_hits: the number of hits in the v view of the track
                w_view_x: the x coordinate of all the hits in the w view of the track
                w_view_wire: the wire coordinate of all the hits in the w view of track
                w_view_q: the charge of all the hits in the w view of the track
                w_view_n_hits: the number of hits in the w view of the track
                n_child_trk: the number of track-like children of this particle
                n_child_shw: the number of shower-like children of this particle
                n_descendants: the total number of descendant particles
                n_descendant_hits: the total number of hits of the descendants
                reco_tier: whether the particle is a primary, secondary...
                pdg: the true PDG code of the particle
        """

        self.u_view_x = u_view_x
        self.u_view_wire = u_view_wire
        self.u_view_q = u_view_q
        self.u_view_n_hits = u_view_n_hits

        self.v_view_x = v_view_x
        self.v_view_wire = v_view_wire
        self.v_view_q = v_view_q
        self.v_view_n_hits = v_view_n_hits

        self.w_view_x = w_view_x
        self.w_view_wire = w_view_wire
        self.w_view_q = w_view_q
        self.w_view_n_hits = w_view_n_hits

        self.n_child_trk = n_child_trk
        self.n_child_shw = n_child_shw
        self.n_descendants = n_descendants
        self.n_descendant_hits = n_descendant_hits
        self.reco_tier = reco_tier

        self.pdg = pdg
