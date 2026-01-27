""" Blocks and models - Adapted (stolen) from juho-lee/set_transformer """

from typing import Optional # torchscript needs this to figure out the function signatures 
from collections import OrderedDict
from abc import ABC, abstractmethod

import torch; import torch.nn as nn

import logging; logger = logging.getLogger("the_logger")

""" Start - blocks """

# Manual implementation
class MAB(nn.Module):
    def __init__(self, dim_q, dim_kv, embd_dim, num_heads, ln=False):
        super().__init__()

        assert embd_dim % num_heads == 0, "Embed dim must be divisible my num heads"
        self.num_heads = num_heads
        self.dim_head = embd_dim // num_heads

        self.fc_q = nn.Linear(dim_q, embd_dim)
        self.fc_k = nn.Linear(dim_kv, embd_dim)
        self.fc_v = nn.Linear(dim_kv, embd_dim)

        self.fc_o = nn.Linear(embd_dim, embd_dim)

        self.ln0 = nn.LayerNorm(embd_dim) if ln else nn.Identity()
        self.ln1 = nn.LayerNorm(embd_dim) if ln else nn.Identity()

    def forward(
        self, Q, K,
        key_padding_mask : Optional[torch.Tensor]=None,
        query_padding_mask : Optional[torch.Tensor]=None
    ):
        B, N_q, _ = Q.shape
        _, N_kv, _ = K.shape

        Q_proj = self.fc_q(Q).view(B, N_q, self.num_heads, self.dim_head).transpose(1, 2)
        K_proj = self.fc_k(K).view(B, N_kv, self.num_heads, self.dim_head).transpose(1, 2)
        V_proj = self.fc_v(K).view(B, N_kv, self.num_heads, self.dim_head).transpose(1, 2)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]
            attn_mask = attn_mask.expand(-1, self.num_heads, N_q, -1)
            attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))
        else:
            attn_mask = None

        # -- Manual SDPA for debugging
        # scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1)) / Q_proj.size(-1)**0.5
        # if attn_mask is not None:
        #     scores = scores + attn_mask  # attention mask must be -inf at padded keys
        # attn_weights = nn.functional.softmax(scores, dim=-1)  # [B, H, L_q, L_k]
        # attn_out = torch.matmul(attn_weights, V_proj)  # [B, H, L_q, D]
        # --
        attn_out = nn.functional.scaled_dot_product_attention(
            Q_proj, K_proj, V_proj, attn_mask=attn_mask, is_causal=False, dropout_p=0.
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(
            B, N_q, self.num_heads * self.dim_head
        )

        O = (
            Q_proj.transpose(1, 2).contiguous().view(B, N_q, self.num_heads * self.dim_head) +
            attn_out
        )
        O = self.ln0(O)
        # O = O + nn.functional.relu(self.fc_o(O))
        O = O + nn.functional.leaky_relu(self.fc_o(O))
        O = self.ln1(O)

        if query_padding_mask is not None:
            O = O.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

        return O

class SAB(nn.Module):
    def __init__(self, dim_in, embd_dim, num_heads, ln=False):
        super().__init__()

        self.mab = MAB(dim_in, dim_in, embd_dim, num_heads, ln=ln)

    def forward(self, X, key_padding_mask : Optional[torch.Tensor]=None):
        return self.mab(X, X, key_padding_mask=key_padding_mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, embd_dim, num_heads, num_inds, ln=False):
        super().__init__()

        self.I = nn.Parameter(torch.Tensor(1, num_inds, embd_dim))
        nn.init.xavier_uniform_(self.I)

        self.mab0 = MAB(embd_dim, dim_in, embd_dim, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, embd_dim, embd_dim, num_heads, ln=ln)

    def forward(self, X, key_padding_mask : Optional[torch.Tensor]=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask)
        return self.mab1(X, H, query_padding_mask=key_padding_mask)

class PMA(nn.Module):
    def __init__(self, embd_dim, num_heads, num_seeds, ln=False):
        super().__init__()

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, embd_dim))
        nn.init.xavier_uniform_(self.S)

        self.mab = MAB(embd_dim, embd_dim, embd_dim, num_heads, ln=ln)

    def forward(self, X, key_padding_mask : Optional[torch.Tensor]=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask)

""" End - blocks """

""" Start - model components """

class SetTransformer(nn.Module):
    def __init__(
        self, dim_in, dim_out,
        num_outputs=None, num_inds=32, dim_hidden=128, num_heads=4, depth=2, ln=False
    ):
        super().__init__()

        self.dim_in, self.dim_out = dim_in, dim_out

        self.enc_first = ISAB(dim_in, dim_hidden, num_heads, num_inds, ln=ln)
        self.encs = nn.ModuleList(
            [ ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln) for _ in range(depth - 1) ]
        )

        if num_outputs is not None:
            self.pma = PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        else:
            self.pma = None

        self.decs = nn.ModuleList(
            [ SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(depth) ]
        )
        self.dec_fc = nn.Linear(dim_hidden, dim_out)

    def print_forward_pass(self, dummy_input):
        modules = []
        def _forward_hook(module, t_input, t_output):
            modules.append(module)
        handle = self.register_forward_hook(_forward_hook)

        with torch.no_grad():
            self(dummy_input)

        print("SetTransformer\n-------------")
        print([module for module in modules])

        handle.remove()

    def forward(self, X, input_key_padding_mask : Optional[torch.Tensor]=None):
        out = self.enc_first(X, key_padding_mask=input_key_padding_mask)
        for enc in self.encs:
            out = enc(out, key_padding_mask=input_key_padding_mask)

        if self.pma is not None:
            out = self.pma(out, key_padding_mask=input_key_padding_mask)
            input_key_padding_mask = None
        
        for dec in self.decs:
            out = dec(out, key_padding_mask=input_key_padding_mask)

        return self.dec_fc(out)

class DotProductWithTemp(nn.Module):
    def __init__(self, init_temp=1.0):
        super().__init__()

        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def forward(self, X):
        sim = torch.bmm(X, X.transpose(1, 2))

        sim = sim * torch.exp(self.temperature)

        sim = torch.sigmoid(sim)

        return sim

class PairwiseMLPSimilarity(nn.Module):
    def __init__(self, embd_dim, dim_hidden=128):
        super().__init__()

        self.mlp_cross_sim = nn.Sequential(
            nn.Linear(2 * embd_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )
        self.mlp_self_sim = nn.Sequential(
            nn.Linear(embd_dim, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, 1),
            nn.Sigmoid()
        )

    def print_forward_pass(self, dummy_input):
        modules = []
        def _forward_hook(module, t_input, t_output):
            modules.append(module)
        handle = self.register_forward_hook(_forward_hook)

        with torch.no_grad():
            self(dummy_input)

        print("PairwiseMLPSimilarity\n-----")
        print([module for module in modules])

        handle.remove()

    def forward(self, X):
        B, N, _ = X.shape

        # idx_i, idx_j = torch.triu_indices(N, N, offset=1) # torch script told me off for this
        triu_indices = torch.triu_indices(N, N, offset=1)
        idx_i = triu_indices[0]; idx_j = triu_indices[1]
        idx_diag = torch.arange(N)

        sim = torch.zeros(B, N, N, device=X.device, dtype=X.dtype)

        X_i = X[:, idx_i, :]
        X_j = X[:, idx_j, :]
        pairwise_input = torch.cat([X_i, X_j], dim=-1) # NOTE Does the ordering matter? choose randomly?
        sim_upper = self.mlp_cross_sim(pairwise_input).squeeze(-1)
        sim[:, idx_i, idx_j] = sim_upper
        sim[:, idx_j, idx_i] = sim_upper

        X_diag = X[:, idx_diag, :]
        sim_diag = self.mlp_self_sim(X_diag).squeeze(-1)
        sim[:, idx_diag, idx_diag] = sim_diag

        return sim

# To prevent N^2 memory usage scaling when running CPU inference
class PairwiseMLPSimilarityLoop(PairwiseMLPSimilarity):
    def forward(self, X):
        B, N, _ = X.shape

        sim = torch.zeros(B, N, N, device=X.device, dtype=X.dtype)

        for b in range(B):
            for i in range(N):
                sim[b, i, i] = self.mlp_self_sim(X[i, i, :]).squeeze(-1)

            for i in range(N):
                for j in range(i + 1, N):
                    pair = torch.cat([X[b, i, :], X[b, j, :]], dim=-1)
                    sim[b, i, j] = self.mlp_cross_sim(pair).squeeze(-1)
                    sim[b, j, i] = sim[b, i, j]

        return sim

# Comprimise between pure loop and pure vectorised approaches
class PairwiseMLPSimilarityChunked(PairwiseMLPSimilarity):
    def __init__(self, embd_dim, dim_hidden=128, chunk_size=1024):
        super().__init__(embd_dim, dim_hidden)
        self.chunk_size = chunk_size

    def forward(self, X):
        B, N, _ = X.shape

        triu_indices = torch.triu_indices(N, N, offset=1)
        idx_i = triu_indices[0]; idx_j = triu_indices[1]
        idx_diag = torch.arange(N)
        num_cross_pairs = idx_i.numel()

        sim = torch.zeros(B, N, N, device=X.device, dtype=X.dtype)

        for b in range(B):
            for start in range(0, N, self.chunk_size):
                end = min(start + self.chunk_size, N)
                idx_diag_chunk = idx_diag[start:end]
                X_diag_chunk = X[b, idx_diag_chunk, :]
                sim_diag_chunk = self.mlp_self_sim(X_diag_chunk).squeeze(-1)
                sim[b, idx_diag_chunk, idx_diag_chunk] = sim_diag_chunk

            for start in range(0, num_cross_pairs, self.chunk_size):
                end = min(start + self.chunk_size, num_cross_pairs)
                idx_i_chunk = idx_i[start:end]
                idx_j_chunk = idx_j[start:end]
                X_i_chunk = X[b, idx_i_chunk, :]
                X_j_chunk = X[b, idx_j_chunk, :]
                pairwise_input_chunk = torch.cat([X_i_chunk, X_j_chunk], dim=-1)
                sim_upper_chunk = self.mlp_cross_sim(pairwise_input_chunk).squeeze(-1)
                sim[b, idx_i_chunk, idx_j_chunk] = sim_upper_chunk
                sim[b, idx_j_chunk, idx_i_chunk] = sim_upper_chunk

        return sim

""" End - model components """

""" Start - models """

class ModelBase(nn.Module, ABC):
    def __init__(self, conf):
        super().__init__()

        self.device = torch.device(conf.device)
        self.gradient_clipping = conf.gradient_clipping
        self.amp_training = conf.amp_training

        self.nets = [ ]
        self.optimizer = None
        self.lr_scheduler = None
        self.lossfunc = None

        self._init_iter()

    def _init_iter(self):
        self.loss = None

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def _calc_loss(self):
        pass

    @abstractmethod
    def set_input(self):
        pass

    @abstractmethod
    def get_current_tensors(self):
        pass

    @abstractmethod
    def save_networks(self, savepath, epoch, val_loss, exp_name):
        pass

    @abstractmethod
    def load_networks(self, loadpath):
        pass

    def print_num_params(self):
        n_params = [ sum(ps.numel() for ps in net.parameters()) / 1e6 for net in self.nets ]
        logger.info(
            f"Model has has {sum(n_params):.2f} "
            f"({'/'.join(f'{n:.2f}' for n in n_params)}) million parameters"
        )

    def eval(self):
        for net in self.nets:
            net.eval()

    def train(self):
        for net in self.nets:
            net.train()

    def get_loss(self):
        return self.loss.item()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        if self.amp_training:
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                self.forward()
                self._calc_loss()
        else:
            self.forward()
            self._calc_loss()
        self.loss.backward()
        if self.gradient_clipping:
            for net in self.nets:
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def step(self, new_epoch=None):
        if self.lr_scheduler is None:
            return

        if new_epoch is not None:
            logger.info(
                f"LR(s) at epoch {new_epoch} start - "
                f"({', '.join([f'{lr:.8f}' for lr in self.lr_scheduler.get_last_lr()])})"
            )
            return

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
            self._scheduler_step()

    def _scheduler_step(self):
        self.lr_scheduler.step()

    def test(self, compute_loss=False):
        with torch.no_grad():
            self.forward()
            if compute_loss:
                self._calc_loss()

class ClusterMergeNet(ModelBase):
    def __init__(self, conf, steps_per_epoch=None):
        super().__init__(conf)

        self.net_intra_cluster_encoder = SetTransformer(
            conf.hit_feat_vec_dim, conf.net_intra_cluster_encoder_params["embd_dim"],
            num_outputs=1,
            num_heads=conf.net_intra_cluster_encoder_params["num_heads"],
            dim_hidden=conf.net_intra_cluster_encoder_params["hidden_dim"],
            num_inds=conf.net_intra_cluster_encoder_params["num_inds"],
            depth=conf.net_intra_cluster_encoder_params["depth"],
            ln=conf.net_intra_cluster_encoder_params["ln"]
        ).to(self.device)

        self.net_inter_cluster_attn = SetTransformer(
            conf.net_intra_cluster_encoder_params["embd_dim"],
            conf.net_inter_cluster_attn_params["embd_dim"],
            num_outputs=None, # Dont apply PMA to have out seq length == in seq length
            num_heads=conf.net_inter_cluster_attn_params["num_heads"],
            dim_hidden=conf.net_inter_cluster_attn_params["hidden_dim"],
            num_inds=conf.net_inter_cluster_attn_params["num_inds"],
            depth=conf.net_inter_cluster_attn_params["depth"],
            ln=conf.net_inter_cluster_attn_params["ln"]
        ).to(self.device)

        if conf.net_inter_cluster_sim_params["use_loop_implementation"]:
            self.net_inter_cluster_sim = PairwiseMLPSimilarityLoop(
                conf.net_inter_cluster_attn_params["embd_dim"],
                dim_hidden=conf.net_inter_cluster_sim_params["hidden_dim"]
            ).to(self.device)
        elif conf.net_inter_cluster_sim_params["use_chunked_implementation"]:
            self.net_inter_cluster_sim = PairwiseMLPSimilarityChunked(
                conf.net_inter_cluster_attn_params["embd_dim"],
                dim_hidden=conf.net_inter_cluster_sim_params["hidden_dim"],
                chunk_size=conf.net_inter_cluster_sim_params["chunk_size"]
            ).to(self.device)
        else:
            self.net_inter_cluster_sim = PairwiseMLPSimilarity(
                conf.net_inter_cluster_attn_params["embd_dim"],
                dim_hidden=conf.net_inter_cluster_sim_params["hidden_dim"]
            ).to(self.device)

        self.nets = [
            self.net_intra_cluster_encoder, self.net_inter_cluster_attn, self.net_inter_cluster_sim
        ]

        if conf.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                [
                    {
                        "params" : self.net_intra_cluster_encoder.parameters(),
                        "lr" : conf.net_intra_cluster_encoder_params["lr"]
                    },
                    {
                        "params" : self.net_inter_cluster_attn.parameters(),
                        "lr" : conf.net_inter_cluster_attn_params["lr"]
                    },
                    {
                        "params" : self.net_inter_cluster_sim.parameters(),
                        "lr" : conf.net_inter_cluster_sim_params["lr"]
                    }
                ]
            )
        else:
            raise ValueError(f"optimizer '{conf.optimizer}' is not valid")

        if conf.lr_scheduler_params["scheduler"] is None:
            self.lr_scheduler = None
        elif conf.lr_scheduler_params["scheduler"] == "OneCycleLR":
            c = conf.lr_scheduler_params["max_lr_factor"]
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=[
                    conf.net_intra_cluster_encoder_params["lr"] * c,
                    conf.net_inter_cluster_attn_params["lr"] * c,
                    conf.net_inter_cluster_sim_params["lr"] * c
                ],
                epochs=conf.epochs,
                steps_per_epoch=steps_per_epoch
            )
        else:
            raise ValueError(f"lr scheduler '{conf.lr_scheduler_params['scheduler']}' is not valid")

        self.lossfunc = nn.MSELoss(reduction="none")
        self.loss_weight_mode = conf.loss_params["weights"]
        self.loss_triplet_lambda = conf.loss_params["triplet_lambda"]
        self.loss_triplet_n_samples = conf.loss_params["triplet_n_samples"]
        self.loss_contrastive_lambda = conf.loss_params["contrastive_lambda"]

        self.training_start_epoch = 0
        if conf.continue_training_from_weights is not None:
            self.load_networks(conf.continue_training_from_weights, continue_training=True)

        self._init_iter()

    def _init_iter(self):
        self.loss = None
        self.chunked_t_clusters = None
        self.chunked_t_clusters_mask = None
        self.chunked_t_cluster_ev_idxs = None
        self.chunked_t_clusters_enc = None
        self.ev_t_clusters_enc = None
        self.ev_t_clusters_attn = None
        self.ev_t_sim = None
        self.ev_t_sim_target = None
        self.ev_t_cardinalities = None

    def print_forward_pass(self):
        logger.info("Printing model:")
        dummy_in = torch.randn(10, 6, self.net_intra_cluster_encoder.dim_in).to(self.device)
        self.net_intra_cluster_encoder.print_forward_pass(dummy_in)
        dummy_in = torch.randn(10, 6, self.net_inter_cluster_attn.dim_in).to(self.device)
        self.net_inter_cluster_attn.print_forward_pass(dummy_in)
        dummy_in = torch.randn(10, 6, self.net_inter_cluster_attn.dim_out).to(self.device)
        self.net_inter_cluster_sim.print_forward_pass(dummy_in)

    def set_input(self, data):
        self._init_iter()
        self.chunked_t_clusters = [ t.to(self.device) for t in data["chunked_input"] ]
        self.chunked_t_clusters_mask = [ t.to(self.device) if t is not None else t for t in data["chunked_input_mask"] ]
        self.chunked_t_cluster_ev_idxs = [ lst for lst in data["chunked_input_ev_idxs"] ]
        self.ev_t_sim_target = [ t.to(self.device) for t in data["ev_target"] ]
        self.ev_t_cardinalities = [ t.to(self.device) for t in data["ev_cardinalities"] ]

    def get_current_tensors(self):
        return {
            "chunked_t_clusters" : self.chunked_t_clusters,
            "chunked_t_clusters_mask" : self.chunked_t_clusters_mask,
            "chunked_t_cluster_ev_idxs" : self.chunked_t_cluster_ev_idxs,
            "ev_t_sim" : self.ev_t_sim,
            "ev_t_sim_target" : self.ev_t_sim_target
        }

    def save_networks(self, savepath, epoch, val_loss, exp_name):
        checkpoint = {
            "net_intra_cluster_encoder_state_dict" : self.net_intra_cluster_encoder.state_dict(),
            "net_inter_cluster_attn_state_dict" : self.net_inter_cluster_attn.state_dict(),
            "net_inter_cluster_sim_state_dict" : self.net_inter_cluster_sim.state_dict(),
            "optimizer_state_dict" : self.optimizer.state_dict(),
            "epoch" : epoch,
            "val_loss" : val_loss,
            "exp_name" : exp_name
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(checkpoint, savepath)
        logger.info(f"Saved weights (epoch {epoch:.2f}, val loss {val_loss:.6f}) to {savepath}")

    def load_networks(self, loadpath, continue_training=False):
        checkpoint = torch.load(loadpath, weights_only=False, map_location=self.device)
        self._load_state_dict_backcompat(
            self.net_intra_cluster_encoder, checkpoint["net_intra_cluster_encoder_state_dict"]
        )
        self._load_state_dict_backcompat(
            self.net_inter_cluster_attn, checkpoint["net_inter_cluster_attn_state_dict"]
        )
        self._load_state_dict_backcompat(
            self.net_inter_cluster_sim, checkpoint["net_inter_cluster_sim_state_dict"]
        )
        logger.info(
            f"Loaded weights (experiment name {checkpoint['exp_name']}, "
            f"epoch {checkpoint['epoch']:.2f}, val loss {checkpoint['val_loss']}) from {loadpath}"
        )
        if continue_training:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info(f"Also loaded optimizer weights")
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
                logger.info(f"Also loaded LR scheduler weights")
            self.training_start_epoch = round(checkpoint["epoch"])
            if abs(checkpoint["epoch"] - self.training_start_epoch) > 1e6:
                logger.warning(
                    "Rounded the continue training checkpoint epoch: "
                    f"{checkpoint['epoch']} -> {self.training_start_epoch}"
                )

    def _load_state_dict_backcompat(self, net, state_dict):
        try:
            net.load_state_dict(state_dict)
        except RuntimeError:
            new_state_dict = OrderedDict()
            for key, val in state_dict.items():
                new_key = key
                if key.startswith("enc_1."):
                    new_key = "enc_first." + key[len("enc_1."):]
                elif key.startswith("enc_2."):
                    new_key = "encs.0." + key[len("enc_2."):]
                elif key.startswith("dec_1."):
                    new_key = "decs.0." + key[len("dec_1."):]
                elif key.startswith("dec_2."):
                    new_key = "decs.1." + key[len("dec_2."):]
                new_state_dict[new_key] = val
            net.load_state_dict(new_state_dict)

    def _forward_intra_cluster_encoder(self):
        self.chunked_t_clusters_enc = []
        for t_clusters_chunk, t_clusters_mask_chunk in zip(
            self.chunked_t_clusters, self.chunked_t_clusters_mask
        ):
            self.chunked_t_clusters_enc.append(
                self.net_intra_cluster_encoder(
                    t_clusters_chunk,
                    input_key_padding_mask=t_clusters_mask_chunk
                )
            )

    def _forward_inter_cluster_attn(self):
        self.ev_t_clusters_attn = []
        for t_clusters_enc in self.ev_t_clusters_enc:
            self.ev_t_clusters_attn.append(self.net_inter_cluster_attn(t_clusters_enc))

    def _forward_inter_cluster_sim(self):
        self.ev_t_sim = []
        for t_clusters_attn in self.ev_t_clusters_attn:
            self.ev_t_sim.append(self.net_inter_cluster_sim(t_clusters_attn))

    def forward(self):
        self._forward_intra_cluster_encoder()

        self._reshape_for_inter_cluster()
    
        self._forward_inter_cluster_attn()

        self._forward_inter_cluster_sim()

    def _reshape_for_inter_cluster(self):
        self.ev_t_clusters_enc = [
            [ None for _ in range(t.size(-1)) ] for t in self.ev_t_sim_target
        ]
        for chunk_idx, ev_idxs in enumerate(self.chunked_t_cluster_ev_idxs):
            for row_idx, ev_idx in enumerate(ev_idxs):
                self.ev_t_clusters_enc[ev_idx[0]][ev_idx[1]] = (
                    self.chunked_t_clusters_enc[chunk_idx][row_idx]
                )
        self.ev_t_clusters_enc = [
            torch.cat(lst).unsqueeze(0).to(self.device) for lst in self.ev_t_clusters_enc
        ]

    def _calc_loss(self):
        ev_losses, ev_weights = [], []
        ev_triplet_losses, ev_triplet_weights = [], []
        ev_contrastive_losses = []
        for pred, target, cards, intra_repr in zip(
            self.ev_t_sim, self.ev_t_sim_target, self.ev_t_cardinalities, self.ev_t_clusters_attn
        ):
            weights = self._make_loss_weights(cards)
            losses = self.lossfunc(pred, target)
            ev_losses.append((losses * weights).sum())
            ev_weights.append(weights.sum())

            if self.loss_triplet_lambda:
                triplet_losses, triplet_weights = self._calc_triplet_loss(pred[0], weights[0])
                ev_triplet_losses.append((triplet_losses * triplet_weights).sum())
                ev_triplet_weights.append((triplet_weights.sum()))

            if self.loss_contrastive_lambda:
                contrastive_losses = self._calc_contrastive_loss(intra_repr, target)
                ev_contrastive_losses.append((contrastive_losses * weights).sum())

        self.loss = sum(ev_losses) / sum(ev_weights)
        if self.loss_triplet_lambda:
            self.loss += (
                self.loss_triplet_lambda * (sum(ev_triplet_losses) / sum(ev_triplet_weights))
            )
        if self.loss_contrastive_lambda:
            self.loss += (
                self.loss_contrastive_lambda * (sum(ev_contrastive_losses) / sum(ev_weights))
            )

    def _calc_triplet_loss(self, t_sim, t_weights):
        N = t_sim.size(0)
        device = t_sim.device

        i = torch.randint(0, N, (self.loss_triplet_n_samples,), device=device)
        j = torch.randint(0, N, (self.loss_triplet_n_samples,), device=device)
        k = torch.randint(0, N, (self.loss_triplet_n_samples,), device=device)

        mask = (i != j) & (j != k) & (i != k)
        i, j, k = i[mask], j[mask], k[mask]

        sim_ij = t_sim[i, j]
        sim_jk = t_sim[j, k]
        sim_ik = t_sim[i, k]

        weights_ij = t_weights[i, j]
        weights_jk = t_weights[j, k]
        weights_ik = t_weights[i, k]
        triplet_weights = (weights_ij + weights_jk + weights_ik) / 3

        triplet_losses = (
            nn.functional.relu(sim_ij + sim_jk - sim_ik - 1.) +
            nn.functional.relu(sim_ij + sim_ik - sim_jk - 1.) +
            nn.functional.relu(sim_ik + sim_jk - sim_ij - 1.)
        )

        return triplet_losses, triplet_weights

    def _calc_contrastive_loss(self, t_clusters_attn, t_sim_target):
        t_embd = nn.functional.normalize(t_clusters_attn, dim=-1)
        t_sim_pred = (torch.matmul(t_embd, t_embd.transpose(-1, -2)) + 1.) / 2.
        contrastive_losses = nn.functional.mse_loss(t_sim_pred, t_sim_target, reduction="none")
        return contrastive_losses

    def _make_loss_weights(self, cardinalities):
        cards_i = cardinalities.unsqueeze(-1)
        cards_j = cardinalities.unsqueeze(-2)

        if self.loss_weight_mode is None:
            return torch.ones_like(cards_i * cards_j)
        elif self.loss_weight_mode == "log":
            return torch.log1p(cards_i) * torch.log1p(cards_j)
        elif self.loss_weight_mode == "log_with_penalty":
            weights = torch.log1p(cards_i) * torch.log1p(cards_j)
            penalty_i = torch.where(cardinalities <= 2, 0.5, 1.0).unsqueeze(-1)
            penalty_j = torch.where(cardinalities <= 2, 0.5, 1.0).unsqueeze(-2)
            penalty = penalty_i * penalty_j
            return weights * penalty
        else:
            raise ValueError(f"loss weighting '{self.loss_weight_mode}' not valid")

""" End - models """
