"""M0 — standard Variational Graph Autoencoder (Kipf & Welling, 2016).

2-layer GCN encoder producing node-level latents z_i ~ N(mu_i, sigma_i),
inner-product decoder A_hat = sigmoid(Z Z^T). Built on PyG's VGAE wrapper,
which handles the reparameterization trick and KL term.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.transforms import RandomLinkSplit


class DegreeCorrectedDecoder(torch.nn.Module):
    """Inner-product decoder with learned per-node degree biases:
    p(A_ij = 1) = sigmoid(z_i . z_j + b_i + b_j).

    Motivation (E1/E2 finding): the plain inner-product decoder forces latent
    geometry to encode degree, which manufactures transitivity (over-clustering).
    Per-node biases absorb degree so the geometry only has to encode affinity.
    Same role as degree correction in the degree-corrected SBM.
    """

    def __init__(self, num_nodes: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(num_nodes))

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, sigmoid: bool = True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        value = value + self.bias[edge_index[0]] + self.bias[edge_index[1]]
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: torch.Tensor, sigmoid: bool = True):
        adj = z @ z.t() + self.bias.unsqueeze(0) + self.bias.unsqueeze(1)
        return torch.sigmoid(adj) if sigmoid else adj


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, latent_dim)
        self.conv_logstd = GCNConv(hidden_channels, latent_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        h = self.conv1(x, edge_index).relu()
        return self.conv_mu(h, edge_index), self.conv_logstd(h, edge_index)


def build_vgae(
    in_channels: int,
    hidden_channels: int = 32,
    latent_dim: int = 16,
    num_nodes: int | None = None,
    degree_corrected: bool = False,
) -> VGAE:
    encoder = GCNEncoder(in_channels, hidden_channels, latent_dim)
    if degree_corrected:
        assert num_nodes is not None, "degree-corrected decoder needs num_nodes"
        return VGAE(encoder, decoder=DegreeCorrectedDecoder(num_nodes))
    return VGAE(encoder)


@dataclass
class TrainResult:
    model: VGAE
    losses: list[float]
    auc: float
    ap: float


def train_vgae(
    data: Data,
    hidden_channels: int = 32,
    latent_dim: int = 16,
    epochs: int = 300,
    lr: float = 0.01,
    val_ratio: float = 0.05,
    test_ratio: float = 0.10,
    seed: int = 0,
    verbose: bool = False,
    degree_corrected: bool = False,
) -> TrainResult:
    """Train M0 on a single graph; evaluate link-prediction AUC/AP on held-out edges."""
    torch.manual_seed(seed)
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        split_labels=True,
        add_negative_train_samples=False,
    )
    train_data, _, test_data = transform(data)

    model = build_vgae(
        data.num_node_features,
        hidden_channels,
        latent_dim,
        num_nodes=data.num_nodes,
        degree_corrected=degree_corrected,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(z, train_data.pos_edge_label_index)
        loss = loss + model.kl_loss() / data.num_nodes
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 50 == 0:
            print(f"epoch {epoch + 1:4d}  loss {loss:.4f}")

    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x, train_data.edge_index)
        auc, ap = model.test(
            z, test_data.pos_edge_label_index, test_data.neg_edge_label_index
        )
    return TrainResult(model=model, losses=losses, auc=float(auc), ap=float(ap))


@torch.no_grad()
def encode_latents(model: VGAE, data: Data) -> torch.Tensor:
    """Posterior-mean latents for all nodes — input to the M1 subsampling decoder."""
    model.eval()
    return model.encode(data.x, data.edge_index)


def node_biases(model: VGAE) -> torch.Tensor | None:
    """Learned per-node degree biases, or None for the plain inner-product decoder."""
    dec = model.decoder
    return dec.bias.detach() if isinstance(dec, DegreeCorrectedDecoder) else None
