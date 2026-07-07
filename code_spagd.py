"""
Self-Perturbation Module (Section 3.2).

A Transformer-based autoencoder that reconstructs normal time-series.
The imperfect reconstruction X~ = Decoder(Encoder(X)) serves as a
pseudo-anomalous sample.  As training progresses the reconstruction
improves, giving the downstream classifier a curriculum of deviations
from easy (large) to hard (subtle).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)                                                   # dropout layer
        pe = torch.zeros(max_len, d_model)                                                   # pe  (max_len,d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)                  #[0,1,2,3,...4999]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)             #the frequency scaling factor.
        ) 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                                                 # (1, max_len, d_model)
        self.register_buffer("pe", pe)                                                       #like when model.cuda() - moves to GPU
 
    def forward(self, x):                                                                    # x: (B, T, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)                                                               #some elements are randomly dropped. 


class SelfPerturbationModule(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 32,           # Reduced to act as a bottleneck
        n_heads: int = 8,            # Matched to Appendix B
        n_encoder_layers: int = 3,   # Matched to Appendix B
        n_decoder_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model

        # Input / output projections
        self.input_proj  = nn.Linear(feature_dim, d_model)
        self.output_proj = nn.Linear(d_model, feature_dim)

        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) #keeps gradients in a good range, better than default init for training stability

    def encode(self, x):
        """x: (B, T, d) → memory: (B, T, d_model)"""
        h = self.pos_enc(self.input_proj(x))
        return self.encoder(h)

    def decode(self, memory, tgt):
        """memory, tgt both (B, T, d_model) → (B, T, d)"""
        h = self.decoder(tgt, memory)
        return self.output_proj(h)

    def forward(self, x):
        """
        x: (B, T, d)
        Returns
          x_recon : (B, T, d)   reconstructed time series  X~
        """
        memory = self.encode(x)

        batch_size, seq_len, _ = x.size()
        tgt_zeros = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        tgt = self.pos_enc(tgt_zeros)

        x_recon = self.decode(memory, tgt)
        return x_recon

        

    def reconstruction_loss(self, x):
        """Eq. (2): Frobenius reconstruction loss."""
        x_recon = self.forward(x)
        # loss = F.mse_loss(x_recon, x, reduction="mean")
        diff = x - x_recon
        loss = torch.sum(diff ** 2) / x.size(0)
        return loss, x_recon

import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyAwareGraphConstruction(nn.Module):
    def __init__(self, num_nodes: int, k: int = 15, top_m_pct: float = 0.30):
        super().__init__()
        self.num_nodes  = num_nodes
        self.k          = min(k, num_nodes - 1)
        self.top_m_pct  = top_m_pct

    # ------------------------------------------------------------------ #
    #  Static graph  (Eqs. 3 & 4)
    # ------------------------------------------------------------------ #

    def build_static_graph(self, X: torch.Tensor):
        """
        X : (B, T, d)
        Returns:
          S : (B, d, d) - Dense pairwise cosine similarity matrix
          A : (B, d, d) - Sparsified (top-k) adjacency matrix
        """
        # Eq. (3): Cosine similarity between variables (columns) per sample
        # X is (B, T, d) -> Transpose to (B, d, T) to treat variables as vectors of length T
        X_t = X.transpose(1, 2) 
        X_norm = F.normalize(X_t, dim=2)
        
        # Batch matrix multiplication: (B, d, T) @ (B, T, d) -> (B, d, d)
        sim = torch.bmm(X_norm, X_norm.transpose(1, 2))
        sim = torch.sigmoid(sim)  # map to [0,1]
        
        # Note: Cosine similarity is inherently symmetric, so (sim + sim.t())/2 is mathematically redundant here.
        S = sim  #dense graph
        A = self._sparsify(S)
        return S, A

    # ------------------------------------------------------------------ #
    #  Dynamic graph  (Eqs. 5 & 6)
    # ------------------------------------------------------------------ #

    def build_dynamic_graph(
        self,
        X_normal: torch.Tensor,
        X_perturbed: torch.Tensor,
        S_static: torch.Tensor,
    ) -> torch.Tensor:
        """
        X_normal    : (B, T, d) – original normal time series (X)
        X_perturbed : (B, T, d) – self-perturbed reconstruction (X~) 
        S_static    : (B, d, d) – DENSE similarity matrix from build_static_graph

        Returns A~ : (B, d, d)
        """
        B, _, _ = X_normal.size()
        
        # Eq. (5): Node-level reconstruction residual (Average over time T only) -> (B, d)
        residuals = (X_normal - X_perturbed).abs().mean(dim=1)  

        # Identify top-m% anomalous candidates per sample in the batch
        m = max(1, int(self.top_m_pct * self.num_nodes))
        _, top_idx = torch.topk(residuals, m, dim=1) # (B, m)
        
        # Create boolean mask M for candidates
        M = torch.zeros(B, self.num_nodes, dtype=torch.bool, device=X_normal.device)
        M.scatter_(1, top_idx, True)

        # ϕ(r) = sigmoid(r) normalises residuals to [0, 1]
        phi = torch.sigmoid(residuals)  # (B, d)

        # Eq. (6): boost affinities for anomalous-candidate nodes
        # Expand dimensions to (B, d, d) for matrix addition
        phi_i = phi.unsqueeze(2).expand(-1, -1, self.num_nodes)  # (B, d, d)
        phi_j = phi.unsqueeze(1).expand(-1, self.num_nodes, -1)  # (B, d, d)

        M_i = M.unsqueeze(2).float().expand(-1, -1, self.num_nodes)
        M_j = M.unsqueeze(1).float().expand(-1, self.num_nodes, -1)

        # CORRECTED: Adding to the DENSE similarity matrix S_static
        S_tilde = S_static + (M_i * phi_i) + (M_j * phi_j)

        # Diagonal: add phi only once
        diag_idx = torch.arange(self.num_nodes, device=X_normal.device)
        S_tilde[:, diag_idx, diag_idx] -= M.float() * phi

        # Symmetrise then sparsify (as explicitly requested in the paper)
        S_tilde = (S_tilde + S_tilde.transpose(1, 2)) * 0.5
        A_tilde = self._sparsify(S_tilde)
        return A_tilde

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _sparsify(self, sim: torch.Tensor) -> torch.Tensor:
        """Keep top-k neighbours per row for batched tensors, zero out the rest."""
        _, d, _ = sim.size()
        A = torch.zeros_like(sim)
        
        sim_clone = sim.clone()
        # Exclude self-loop from kNN selection by setting diagonal to a large negative number
        idx = torch.arange(d, device=sim.device)
        sim_clone[:, idx, idx] = -1e9 
        
        # Get top-k values and indices across the last dimension
        topk_vals, topk_idx = torch.topk(sim_clone, self.k, dim=2)
        
        # Scatter the top-k values back into the zeroed matrix
        A.scatter_(2, topk_idx, topk_vals)
        
        # Restore self-loop values
        A[:, idx, idx] = sim[:, idx, idx]
        return A

    def forward(self, X_normal, X_perturbed):
        """
        X_normal    : (B, T, d) – normal samples
        X_perturbed : (B, T, d) – self-perturbed samples X~

        Returns
          A       : (B, d, d) - Static graph
          A_tilde : (B, d, d) - Dynamic graph
        """
        S, A    = self.build_static_graph(X_normal)
        A_tilde = self.build_dynamic_graph(X_normal, X_perturbed, S)
        return A, A_tilde

"""
Spatio-Temporal Anomaly Detection Module (Section 3.4).

Vectorised GAT (B,T,d,latent) — no Python loop over B*T.
Predictor: horizontal concat [Z_T | Z~_T] -> 2*n_chunks*tcn_hidden -> 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised GAT  (Eqs. 7-8)
# ─────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Fully-vectorised single-head GAT.
    Operates directly on (B, T, d, in_dim) — no Python loop.

    Attention factorisation trick:
        e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
             = LeakyReLU( (a1^T Wh_i) + (a2^T Wh_j) )
    This avoids the O(d^2 * 2*out_dim) concatenation tensor.

    Shared weights: same W, a1, a2 used for both A and A_tilde (paper Eq. 7).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.a1    = nn.Parameter(torch.empty(out_dim, 1))
        self.a2    = nn.Parameter(torch.empty(out_dim, 1))
        nn.init.xavier_uniform_(self.a1.data, gain=1.414)
        nn.init.xavier_uniform_(self.a2.data, gain=1.414)
        self.drop  = nn.Dropout(dropout)
        self.lrelu = nn.LeakyReLU(0.2)

    def _attend(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        H : (B, T, d, in_dim)
        A : (B, d, d)
        Returns (B, T, d, out_dim)
        """
        Wh    = self.W(H)                                    # (B, T, d, out_dim)
        # Factorised attention scores
        e_i   = torch.matmul(Wh, self.a1)                   # (B, T, d, 1)
        e_j   = torch.matmul(Wh, self.a2)                   # (B, T, d, 1)
        # broadcast: (B,T,d,1) + (B,T,1,d) → (B,T,d,d)
        e     = self.lrelu(e_i + e_j.transpose(-1, -2))
        # Mask non-edges; A is (B,d,d) → unsqueeze T dim
        e     = e.masked_fill(A.unsqueeze(1) == 0, float("-inf"))
        alpha = F.softmax(e, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0)
        alpha = self.drop(alpha)
        # (B,T,d,d) @ (B,T,d,out_dim) → (B,T,d,out_dim)
        return F.elu(torch.matmul(alpha, Wh))

    def forward(self, H, A, H_tilde, A_tilde):
        """All tensors: H/H_tilde (B,T,d,in_dim), A/A_tilde (B,d,d)."""
        return self._attend(H, A), self._attend(H_tilde, A_tilde)


# ─────────────────────────────────────────────────────────────────────────────
# TCN Block
# ─────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel - 1
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad), Chomp1d(pad),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel, padding=pad), Chomp1d(pad),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.net(x) + self.res(x))


# ─────────────────────────────────────────────────────────────────────────────
# Spatio-Temporal Detector
# ─────────────────────────────────────────────────────────────────────────────

class SpatioTemporalDetector(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int   = 64,
        n_gat_layers: int = 2,
        n_chunks: int     = 5,
        tcn_hidden: int   = 128,
        dropout: float    = 0.1,
    ):
        super().__init__()
        self.n_gat_layers = n_gat_layers
        self.n_chunks     = n_chunks
        self.feature_dim  = feature_dim

        self.node_embed = nn.Linear(1, latent_dim)

        self.gat_layers = nn.ModuleList(
            [GATLayer(latent_dim, latent_dim, dropout) for _ in range(n_gat_layers)]
        )

        tcn_in = feature_dim * latent_dim
        self.tcn = TCNBlock(tcn_in, tcn_hidden, kernel=3, dropout=dropout)

        # Horizontal concat: [Z_T | Z~_T] → n_chunks * tcn_hidden
        pred_in = 2*n_chunks * tcn_hidden
        self.predictor = nn.Sequential(
            nn.Linear(pred_in, pred_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pred_in // 2, 1),
        )

    def _embed(self, X: torch.Tensor) -> torch.Tensor:
        """X: (B, T, d) → (B, T, d, latent_dim)"""
        return self.node_embed(X.unsqueeze(-1))

    def _spatial(self, X_input, X_perturbed, A, A_tilde):
        """
        Same node features go through both graphs.
        The two paths differ only in graph topology (A vs A_tilde).
        This is the correct reading of paper Eq 7-8 and Algorithm 1 line 8.
        """
        H = self._embed(X_input)        # (B, T, d, latent)
        H_tilde = self._embed(X_perturbed)  # same starting features

        for layer in self.gat_layers:
            H, H_tilde = layer(H, A, H_tilde, A_tilde)

        return H, H_tilde


    def _temporal(self, H: torch.Tensor) -> torch.Tensor:
        """(B, T, d, latent) → (B, n_chunks * tcn_hidden)"""
        B, T, d, lat = H.shape
        feat       = H.reshape(B, T, d * lat)
        chunk_size = T // self.n_chunks
        T_eff      = chunk_size * self.n_chunks
        chunks     = torch.split(feat[:, :T_eff], chunk_size, dim=1)
        z_list = []
        for chunk in chunks:
            c = self.tcn(chunk.permute(0, 2, 1))   # (B, tcn_hidden, chunk_sz)
            z_list.append(c.mean(dim=-1))           # (B, tcn_hidden)
        return torch.cat(z_list, dim=-1)            # (B, n_chunks*tcn_hidden)

    def forward(self, X_normal, X_perturbed, A, A_tilde):
        """
        X_normal    : (B, T, d) - Normal sequence
        X_perturbed : (B, T, d) - Pseudo-anomalous sequence
        A, A_tilde  : (B, d, d) - The reference graphs
        Returns p_hat : (B,)
        """
        B = X_normal.size(0)
        H_norm, H_anom = self._spatial(X_normal, X_perturbed, A, A_tilde)
        
        Z_T       = self._temporal(H_norm)       # Features of X under A
        Z_tilde_T = self._temporal(H_anom)       # Features of X~ under A~
        
        # Horizontal concat -> Predictor compares the two representations
        Z_stack = torch.cat([Z_T, Z_tilde_T], dim=-1)   # (B, 2*n_chunks*tcn_hidden)
        
        p_hat = torch.sigmoid(self.predictor(Z_stack)).squeeze(-1)
        return p_hat

import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_perturbation  import SelfPerturbationModule
from .graph_construction import AnomalyAwareGraphConstruction
from .spatio_temporal    import SpatioTemporalDetector


class SPAGD(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        sp_d_model: int        = 128,
        sp_n_heads: int        = 8,
        sp_n_enc_layers: int   = 3,
        sp_n_dec_layers: int   = 3,
        sp_dim_ff: int         = 256,
        sp_dropout: float      = 0.1,
        graph_k: int           = 15,
        graph_top_m_pct: float = 0.30,
        st_latent_dim: int     = 64,
        st_n_gat_layers: int   = 2,
        st_n_chunks: int       = 5,
        st_tcn_hidden: int     = 128,
        st_dropout: float      = 0.1,
        beta: float            = 0.01,
    ):
        super().__init__()
        self.beta = beta

        self.sp_module = SelfPerturbationModule(
            feature_dim      = feature_dim,
            d_model          = sp_d_model,
            n_heads          = sp_n_heads,
            n_encoder_layers = sp_n_enc_layers,
            n_decoder_layers = sp_n_dec_layers,
            dim_feedforward  = sp_dim_ff,
            dropout          = sp_dropout,
        )
        self.aagc = AnomalyAwareGraphConstruction(
            num_nodes = feature_dim,
            k         = graph_k,
            top_m_pct = graph_top_m_pct,
        )
        self.st_detector = SpatioTemporalDetector(
            feature_dim  = feature_dim,
            latent_dim   = st_latent_dim,
            n_gat_layers = st_n_gat_layers,
            n_chunks     = st_n_chunks,
            tcn_hidden   = st_tcn_hidden,
            dropout      = st_dropout,
        )

    # ------------------------------------------------------------------ #
    #  Training (Algorithm 1)
    # ------------------------------------------------------------------ #
    def forward(self, X: torch.Tensor, mode: str = "train"):
        """
        Unified forward pass to support nn.DataParallel.
        """
        if mode == "train":
            return self.forward_train(X)
        elif mode == "score":
            return self.score(X)
        else:
            raise ValueError(f"Unknown mode: {mode}")

# ------------------------------------------------------------------ #
    #  Training (Algorithm 1)
    # ------------------------------------------------------------------ #
    def forward_train(self, X: torch.Tensor):
        """
        Paper Algorithm 1 — all bugs fixed:
        - Single forward pass: one X_tilde used for both loss and graphs
        - p_normal matches score() exactly → trained to output LOW for normal X
        - p_perturbed swaps X and X_tilde → trained to output HIGH
        """
        B = X.size(0)

        # Bug 1 fix: ONE forward pass, reuse X_tilde everywhere
        X_tilde = self.sp_module(X)

        # Paper Eq. 2: Frobenius reconstruction loss (||X - X̃||_F² / B)
        diff    = X - X_tilde
        loss_sp = torch.sum(diff ** 2) / B

        # Graph construction (unchanged)
        A, A_tilde = self.aagc(X, X_tilde)

        # Bug 2+3 fix: correct pairs that align with score() at test time
        # p_normal  : EXACT SAME call as score() → train model to output LOW (0) for normal X
        p_normal    = self.st_detector(X,       X_tilde, A, A_tilde)  # label 0

        # p_perturbed: X_tilde is "anomaly", X is "normal baseline" → output HIGH (1)
        p_perturbed = self.st_detector(X_tilde, X,       A, A_tilde)  # label 1

        p_all  = torch.cat([p_normal, p_perturbed], dim=0)
        labels = torch.cat([
            torch.zeros(B, device=X.device),
            torch.ones(B,  device=X.device),
        ])

        loss_ad    = F.binary_cross_entropy(p_all.clamp(1e-7, 1 - 1e-7), labels)
        loss_total = loss_sp + self.beta * loss_ad

        return loss_total, loss_sp.item(), loss_ad.item(), p_normal.detach()

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def score(self, X: torch.Tensor) -> torch.Tensor:
        """
        At test time, X_recon acts as the "normal" anchor, and X acts as the 
        potentially anomalous target sequence to be evaluated.
        """
        # Step 1 & 2: Generate baseline and build graphs
        X_tilde    = self.sp_module(X)
        A, A_tilde = self.aagc(X, X_tilde)          # same argument order as training
        return self.st_detector(X, X_tilde, A, A_tilde) 

"""
src/data/anomaly_dataset.py

Loader for MSL, SMAP, SWaT using pre-processed .npy files
from https://github.com/thuml/Anomaly-Transformer
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class TimeSeriesDataset(Dataset):
    def __init__(self, windows: np.ndarray, labels=None):
        self.X = torch.from_numpy(windows).float()
        if labels is None:
            self.y = torch.zeros(len(windows), dtype=torch.long)
        else:
            self.y = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_anomaly_dataset(
    data_dir: str,
    dataset: str,
    window_size: int = 100,
    stride: int = 100,
    val_ratio: float = 0.1,
    batch_size: int = 64,
):
    train_path = os.path.join(data_dir, dataset, f"{dataset}_train.npy")
    test_path  = os.path.join(data_dir, dataset, f"{dataset}_test.npy")
    label_path = os.path.join(data_dir, dataset, f"{dataset}_test_label.npy")

    for p in [train_path, test_path, label_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    X_train_raw = np.load(train_path).astype(np.float32)
    X_test_raw  = np.load(test_path).astype(np.float32)
    y_test_raw  = np.load(label_path).astype(np.int64)

    if y_test_raw.ndim == 2:
        y_test_raw = y_test_raw[:, 0]

    scaler = StandardScaler().fit(X_train_raw)
    X_train_raw = scaler.transform(X_train_raw)
    X_test_raw  = scaler.transform(X_test_raw)

    def make_windows(X, y=None):
        wins, labs = [], []
        for s in range(0, len(X) - window_size + 1, stride):
            wins.append(X[s : s + window_size])
            if y is not None:
                labs.append(int(y[s : s + window_size].max()))
        wins_arr = np.stack(wins) if wins else np.empty((0, window_size, X.shape[1]), dtype=np.float32)
        labs_arr = np.array(labs, dtype=np.int64) if y is not None else None
        return wins_arr, labs_arr

    # ── Bug 4 fix: create windows FIRST (temporal order intact), THEN randomly split windows ──
    all_train_wins, _ = make_windows(X_train_raw)   # (N_windows, T, d) — temporally ordered

    N         = len(all_train_wins)
    n_val_w   = max(1, int(N * val_ratio))
    perm      = np.random.RandomState(42).permutation(N)

    X_tr_w  = all_train_wins[perm[n_val_w:]]   # 90% of windows, random
    X_val_w = all_train_wins[perm[:n_val_w]]   # 10% of windows, random

    # Test: stride=1 to not miss any anomalous window
    X_te_w, y_te_w = make_windows(X_test_raw, y_test_raw)
    # Override stride for test regardless of config
    test_wins, test_labs = [], []
    for s in range(0, len(X_test_raw) - window_size + 1, 1):
        test_wins.append(X_test_raw[s : s + window_size])
        test_labs.append(int(y_test_raw[s : s + window_size].max()))
    X_te_w  = np.stack(test_wins).astype(np.float32)
    y_te_w  = np.array(test_labs, dtype=np.int64)

    feature_dim = X_tr_w.shape[2]

    expected = {
        "MSL":  dict(train_rows=58317,  test_rows=73729,  channels=55, ts_anom_pct=10.72),
        "SMAP": dict(train_rows=135183, test_rows=427617, channels=25, ts_anom_pct=13.13),
        "SWaT": dict(train_rows=496800, test_rows=449919, channels=51, ts_anom_pct=11.98),
    }
    if dataset in expected:
        e = expected[dataset]
        print(f"\n{dataset} data check:")
        print(f"  Train rows : {len(X_train_raw):>7}  expected {e['train_rows']:>7}")
        print(f"  Test rows  : {len(X_test_raw):>7}  expected {e['test_rows']:>7}")
        print(f"  Channels   : {feature_dim:>7}  expected {e['channels']:>7}")
        print(f"  Anom %     : {y_te_w.mean()*100:>7.2f}%  (paper ts-level: {e['ts_anom_pct']:.2f}%)")
        print(f"  Train wins : {X_tr_w.shape[0]:>7} | Val wins: {X_val_w.shape[0]:>7} | "
              f"Test wins: {X_te_w.shape[0]:>7}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(TimeSeriesDataset(X_tr_w),
                              batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(TimeSeriesDataset(X_val_w),
                              batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(TimeSeriesDataset(X_te_w, y_te_w),
                              batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    return train_loader, val_loader, test_loader, feature_dim

"""
scripts/train.py  —  SPAGD training script

Key fixes vs. original:
  1. validate() now mirrors train_one_epoch() during warmup (SP-only loss).
     Original used full model loss (including untrained AD module), which caused
     early-stopping to fire prematurely and the "best" checkpoint to be saved
     before the AD module ever trained → AUC ≈ 50% (near-random).
  2. Patience counter and best_val_loss are reset when warmup ends so early
     stopping starts fresh in the post-warmup phase.
  3. Patience counter does not increment during warmup epochs.
  4. --resume flag is now honoured.
  5. stride defaults to window_size (non-overlapping, per Appendix B) instead of
     hardcoded 100.
  6. lsp / lad are safely converted to Python float before accumulation.
  7. Checkpoint directory is created automatically.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.anomaly_dataset import load_anomaly_dataset
from src.models.spagd import SPAGD
from src.utils.metrics import compute_metrics
from src.utils.logger import get_logger, save_checkpoint, load_checkpoint, save_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/swat.yaml")
    p.add_argument("--gpu",    default="0")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint to resume from.")
    p.add_argument("--seed",   type=int, default=42)
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _scalar(x) -> float:
    """Return a plain Python float whether x is a Tensor or already a scalar."""
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, grad_clip,
                    epoch: int, warmup_epochs: int = 5):
    model.train()
    total_loss = total_sp = total_ad = 0.0
    n_batches  = 0
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    for X, _ in loader:
        X = X.to(device)
        optimizer.zero_grad()

        if epoch <= warmup_epochs:
            # SP-only warmup: train the self-perturbation module alone.
            X_tilde = base_model.sp_module(X)
            loss    = (X - X_tilde).pow(2).mean()
            lsp     = loss.item()
            lad     = 0.0
        else:
            loss, lsp, lad, _ = model(X, mode="train")
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_sp   += _scalar(lsp)
        total_ad   += _scalar(lad)
        n_batches  += 1

    n = max(n_batches, 1)
    return total_loss / n, total_sp / n, total_ad / n


@torch.no_grad()
def validate(model, loader, device, epoch: int = 1, warmup_epochs: int = 5):
    """
    Validation loss is computed with the SAME objective used during training:
      - SP reconstruction loss only  (during warmup)
      - Full SP + AD loss            (after warmup)

    BUG IN ORIGINAL: validate() always used the full model loss.  During warmup
    the AD module is uninitialised, so val_loss was dominated by random AD output,
    the best checkpoint was saved early, and early-stopping fired before the model
    actually converged.
    """
    model.eval()
    total_loss = total_sp = total_ad = 0.0
    n_batches  = 0
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    for X, _ in loader:
        X = X.to(device)

        if epoch <= warmup_epochs:
            X_tilde = base_model.sp_module(X)
            loss    = (X - X_tilde).pow(2).mean()
            lsp     = loss.item()
            lad     = 0.0
        else:
            loss, lsp, lad, _ = model(X, mode="train")
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

        total_loss += loss.item()
        total_sp   += _scalar(lsp)
        total_ad   += _scalar(lad)
        n_batches  += 1

    n = max(n_batches, 1)
    return total_loss / n, total_sp / n, total_ad / n


@torch.no_grad()
def evaluate(model, loader, device, logger):
    model.eval()
    all_scores, all_labels = [], []

    for X, y in loader:
        scores = model(X.to(device), mode="score")
        all_scores.append(scores.cpu().numpy())
        all_labels.append(y.numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    metrics = compute_metrics(all_scores, all_labels)
    return metrics, all_scores, all_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    set_seed(args.seed)

    gpu_ids = [int(g) for g in args.gpu.split(",") if g.strip()]
    device  = torch.device(
        f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    )

    os.makedirs(os.path.dirname(cfg["output"]["log_file"]), exist_ok=True)
    logger = get_logger("spagd", log_file=cfg["output"]["log_file"])
    logger.info(f"Device : {device}")

    ds_cfg = cfg["dataset"]

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader, feature_dim = load_anomaly_dataset(
        data_dir    = ds_cfg["data_dir"],
        dataset     = ds_cfg["name"],
        window_size = ds_cfg["window_size"],
        # FIX: default stride = window_size (non-overlapping per Appendix B)
        stride      = ds_cfg.get("stride", ds_cfg["window_size"]),
        val_ratio   = ds_cfg["val_ratio"],
        batch_size  = ds_cfg.get("batch_size", 64),
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    m_cfg = cfg["model"]
    model = SPAGD(
        feature_dim      = feature_dim,
        sp_d_model       = m_cfg["sp_d_model"],
        sp_n_heads       = m_cfg["sp_n_heads"],
        sp_n_enc_layers  = m_cfg["sp_n_enc_layers"],
        sp_n_dec_layers  = m_cfg["sp_n_dec_layers"],
        sp_dim_ff        = m_cfg["sp_dim_ff"],
        sp_dropout       = m_cfg["sp_dropout"],
        graph_k          = m_cfg["graph_k"],
        graph_top_m_pct  = m_cfg["graph_top_m_pct"],
        st_latent_dim    = m_cfg["st_latent_dim"],
        st_n_gat_layers  = m_cfg["st_n_gat_layers"],
        st_n_chunks      = m_cfg["st_n_chunks"],
        st_tcn_hidden    = m_cfg["st_tcn_hidden"],
        st_dropout       = m_cfg["st_dropout"],
        beta             = m_cfg["beta"],
    ).to(device)

    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    tr_cfg        = cfg["training"]
    warmup_epochs = tr_cfg.get("warmup_epochs", 5)
    base_model    = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Differential learning rate: ST detector trains 10× faster
    # optimizer = optim.Adam([
    #     {"params": base_model.sp_module.parameters(),   "lr": tr_cfg["learning_rate"]},
    #     {"params": base_model.st_detector.parameters(), "lr": tr_cfg["learning_rate"] * 10},
    # ], lr=tr_cfg["learning_rate"])
    
    optimizer = optim.Adam(model.parameters(), lr=tr_cfg["learning_rate"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    ckpt_dir  = cfg["output"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)          # FIX: ensure dir exists
    best_ckpt = os.path.join(ckpt_dir, "best_model.pt")

    start_epoch    = 0
    best_val_loss  = float("inf")
    patience_ctr   = 0

    # ------------------------------------------------------------------
    # FIX: honour --resume
    # ------------------------------------------------------------------
    if args.resume and os.path.isfile(args.resume):
        meta = load_checkpoint(model, optimizer, args.resume, device)
        if isinstance(meta, dict):
            start_epoch   = meta.get("epoch", 0)
            best_val_loss = meta.get("val_loss", float("inf"))
        logger.info(f"Resumed from {args.resume} (epoch {start_epoch})")

    logger.info(
        f"Starting training — epochs={tr_cfg['epochs']}, "
        f"warmup={warmup_epochs}, patience={tr_cfg['early_stop_patience']}"
    )

    for epoch in range(start_epoch + 1, tr_cfg["epochs"] + 1):

        train_loss, train_sp, train_ad = train_one_epoch(
            model, train_loader, optimizer, device,
            tr_cfg["grad_clip"], epoch, warmup_epochs=warmup_epochs,
        )
        # FIX: pass epoch so validate uses the same objective as training
        val_loss, val_sp, val_ad = validate(
            model, val_loader, device,
            epoch=epoch, warmup_epochs=warmup_epochs,
        )
        scheduler.step(val_loss)

        phase = "warmup" if epoch <= warmup_epochs else "train "
        logger.info(
            f"Epoch {epoch:03d} [{phase}] | "
            f"train={train_loss:.4f} (sp={train_sp:.4f} ad={train_ad:.4f}) | "
            f"val={val_loss:.4f} (sp={val_sp:.4f} ad={val_ad:.4f})"
        )

        # ------------------------------------------------------------------
        # FIX: reset patience and best-loss when warmup ends so early
        # stopping starts fresh in the post-warmup training phase.
        # ------------------------------------------------------------------
        if epoch == warmup_epochs:
            best_val_loss = float("inf")
            patience_ctr  = 0
            logger.info("Warmup complete — early-stop counter reset.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            save_checkpoint(model, optimizer, epoch,
                            {"epoch": epoch, "val_loss": val_loss}, best_ckpt)
            logger.info(f"  ✓ best val_loss={best_val_loss:.4f} saved.")
        else:
            # FIX: do not count warmup epochs toward patience
            if epoch > warmup_epochs:
                patience_ctr += 1
                if patience_ctr >= tr_cfg["early_stop_patience"]:
                    logger.info(f"Early stopping at epoch {epoch} "
                                f"(no improvement for {patience_ctr} epochs).")
                    break

    # ------------------------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------------------------
    logger.info("\n=== Loading best model for final evaluation ===")
    load_checkpoint(model, None, best_ckpt, device)
    metrics, scores, labels = evaluate(model, test_loader, device, logger)

    logger.info(f"\n=== Final Test Results — {ds_cfg['name']} ===")
    logger.info(f"  AUC   : {metrics['auc']:.2f}%")
    logger.info(f"  AUPRC : {metrics['auprc']:.2f}%")
    logger.info(f"  F1    : {metrics['f1']:.2f}%")
    logger.info(f"  Threshold : {metrics['threshold']:.4f}")

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    save_json(metrics, os.path.join(results_dir, "train_eval_metrics.json"))
    np.save(os.path.join(results_dir, "test_scores.npy"), scores)
    np.save(os.path.join(results_dir, "test_labels.npy"), labels)


if __name__ == "__main__":
    main()

"""
scripts/evaluate.py  —  Standalone evaluation of a trained SPAGD checkpoint.

Usage
-----
  python scripts/evaluate.py --config configs/msl.yaml  \
      --checkpoint outputs/checkpoints/msl/best_model.pt --gpu 0 --plot

  python scripts/evaluate.py --config configs/smap.yaml \
      --checkpoint outputs/checkpoints/smap/best_model.pt --gpu 0 --plot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, precision_recall_curve

from src.data.anomaly_dataset import load_anomaly_dataset
from src.models.spagd         import SPAGD
from src.utils.metrics        import compute_metrics
from src.utils.logger         import get_logger, load_checkpoint, save_json


# ---------------------------------------------------------------------------
# Args / config
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/msl.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--gpu",        default="0")
    p.add_argument("--plot",       action="store_true")
    p.add_argument("--batch_size", type=int, default=64)
    return p.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for X, y in loader:
        scores = model(X.to(device), mode="score")
        all_scores.append(scores.cpu().numpy())
        all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(scores, labels, dataset_name, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {dataset_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # PR
    prec, rec, _ = precision_recall_curve(labels, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color="darkorange", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {dataset_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pr_curve.png"), dpi=150)
    plt.close(fig)

    # Score distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[labels == 0], bins=60, alpha=0.6, label="Normal",    color="steelblue")
    ax.hist(scores[labels == 1], bins=60, alpha=0.6, label="Anomalous", color="tomato")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Score Distribution — {dataset_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "score_dist.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    ds_cfg = cfg["dataset"]
    m_cfg  = cfg["model"]

    gpu_ids = [int(g) for g in args.gpu.split(",") if g.strip()]
    device  = torch.device(
        f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and gpu_ids else "cpu"
    )

    logger = get_logger("spagd_eval")
    logger.info(f"Dataset    : {ds_cfg['name']}")
    logger.info(f"Checkpoint : {args.checkpoint}")
    logger.info(f"Device     : {device}")

    # FIX: stride defaults to window_size (non-overlapping, Appendix B)
    _, _, test_loader, feature_dim = load_anomaly_dataset(
        data_dir    = ds_cfg["data_dir"],
        dataset     = ds_cfg["name"],
        window_size = ds_cfg["window_size"],
        stride      = ds_cfg.get("stride", ds_cfg["window_size"]),
        val_ratio   = ds_cfg["val_ratio"],
        batch_size  = args.batch_size,
    )

    model = SPAGD(
        feature_dim      = feature_dim,
        sp_d_model       = m_cfg["sp_d_model"],
        sp_n_heads       = m_cfg["sp_n_heads"],
        sp_n_enc_layers  = m_cfg["sp_n_enc_layers"],
        sp_n_dec_layers  = m_cfg["sp_n_dec_layers"],
        sp_dim_ff        = m_cfg["sp_dim_ff"],
        sp_dropout       = m_cfg["sp_dropout"],
        graph_k          = m_cfg["graph_k"],
        graph_top_m_pct  = m_cfg["graph_top_m_pct"],
        st_latent_dim    = m_cfg["st_latent_dim"],
        st_n_gat_layers  = m_cfg["st_n_gat_layers"],
        st_n_chunks      = m_cfg["st_n_chunks"],
        st_tcn_hidden    = m_cfg["st_tcn_hidden"],
        st_dropout       = m_cfg["st_dropout"],
        beta             = m_cfg["beta"],
    ).to(device)

    load_checkpoint(model, None, args.checkpoint, device)
    logger.info("Model loaded.")

    logger.info("Running inference on test set ...")
    scores, labels = infer(model, test_loader, device)

    metrics = compute_metrics(scores, labels, verbose=False)

    logger.info(f"\n=== Results on {ds_cfg['name']} ===")
    logger.info(f"  AUC      : {metrics['auc']:.2f}%")
    logger.info(f"  AUPRC    : {metrics['auprc']:.2f}%")
    logger.info(f"  F1       : {metrics['f1']:.2f}%")
    logger.info(f"  Threshold: {metrics['threshold']:.4f}")

    # Paper targets (non-adjusted metrics from main-text Table 2)
    targets = {
        "MSL":  {"auc": 66.50, "auprc": 21.45, "f1": 30.89},
        "SMAP": {"auc": 62.38, "auprc": 18.15, "f1": 27.32},
        "SWaT": {"auc": 86.30, "auprc": 77.20, "f1": 78.77},
    }
    if ds_cfg["name"] in targets:
        t = targets[ds_cfg["name"]]
        logger.info(
            f"\n  Paper target — AUC: {t['auc']:.2f}%  "
            f"AUPRC: {t['auprc']:.2f}%  F1: {t['f1']:.2f}%"
        )

    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    save_json(metrics, os.path.join(results_dir, "eval_metrics.json"))
    np.save(os.path.join(results_dir, "test_scores.npy"), scores)
    np.save(os.path.join(results_dir, "test_labels.npy"), labels)

    if args.plot:
        plot_results(scores, labels, ds_cfg["name"],
                     os.path.join(results_dir, "plots"))

    return metrics


if __name__ == "__main__":
    main()