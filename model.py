"""SBA-GOAD model described in the manuscript.

The module contains model definitions only. The expected input is a batch of
differential-entropy features with shape [batch, electrodes, frequency_bands].
"""

from dataclasses import dataclass
from typing import Dict, Iterator, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv


@dataclass(frozen=True)
class SBAGOADConfig:
    num_electrodes: int = 62
    num_bands: int = 5
    num_classes: int = 3
    num_students: int = 3
    band_attention_hidden: int = 16
    hidden_dim: int = 128
    association_dim: int = 64
    cbam_reduction: int = 8
    attention_heads: int = 4
    graph_layers: int = 3
    dropout: float = 0.3
    discriminator_dropout: float = 0.2
    knn_ratio: float = 0.2

    @property
    def knn_k(self) -> int:
        return round(self.knn_ratio * self.num_electrodes)


def build_spatial_knn_edge_index(
    coordinates: Tensor,
    k: int | None = None,
    knn_ratio: float = 0.2,
) -> Tensor:
    """Build the symmetric spatial KNN topology used by the model.

    The directed KNN graph is symmetrized by set union, and self-loops are
    appended. ``coordinates`` must contain one 2-D or 3-D coordinate per
    electrode and have shape [N, C].
    """
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32)
    if coordinates.ndim != 2:
        raise ValueError("coordinates must have shape [num_electrodes, dimensions]")

    num_nodes = coordinates.size(0)
    neighbors = round(knn_ratio * num_nodes) if k is None else k
    if not 0 < neighbors < num_nodes:
        raise ValueError("k must satisfy 0 < k < num_electrodes")

    distances = torch.cdist(coordinates, coordinates)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.topk(neighbors, dim=1, largest=False).indices

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    sources = torch.arange(num_nodes).unsqueeze(1).expand_as(nearest)
    adjacency[sources, nearest] = True
    adjacency = adjacency | adjacency.T
    adjacency.fill_diagonal_(True)
    return adjacency.nonzero(as_tuple=False).T.contiguous().long()


def _batch_edge_index(edge_index: Tensor, batch_size: int, num_nodes: int) -> Tensor:
    edge_index = edge_index.long()
    edge_count = edge_index.size(1)
    offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
    offsets = offsets.repeat_interleave(edge_count).unsqueeze(0)
    return edge_index.repeat(1, batch_size) + offsets


class BandAttention(nn.Module):
    def __init__(self, num_bands: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bands),
        )
        self.projection = nn.Linear(num_bands, output_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        band_weights = F.softmax(self.attention(x), dim=-1)
        return self.projection(x * band_weights), band_weights


class CBAMFeatureRefinement(nn.Module):
    """Feature-wise and electrode-wise attention for EEG node features."""

    def __init__(self, feature_dim: int, reduction: int) -> None:
        super().__init__()
        reduced_dim = max(1, feature_dim // reduction)
        self.shared_feature_mlp = nn.Sequential(
            nn.Linear(feature_dim, reduced_dim),
            nn.ReLU(),
            nn.Linear(reduced_dim, feature_dim),
        )
        self.electrode_attention = nn.Sequential(
            nn.Linear(2, 16),
            nn.ELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        average_descriptor = x.mean(dim=1)
        maximum_descriptor = x.max(dim=1).values
        feature_weights = torch.sigmoid(
            self.shared_feature_mlp(average_descriptor)
            + self.shared_feature_mlp(maximum_descriptor)
        )
        x = x * feature_weights.unsqueeze(1)

        electrode_descriptor = torch.cat(
            (x.mean(dim=-1, keepdim=True), x.max(dim=-1, keepdim=True).values),
            dim=-1,
        )
        electrode_weights = torch.sigmoid(self.electrode_attention(electrode_descriptor))
        return x * electrode_weights, feature_weights, electrode_weights


class SpearmanFeatureAssociation(nn.Module):
    """Project sample-wise Spearman feature-association profiles into node features."""

    def __init__(self, num_electrodes: int, association_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(num_electrodes, association_dim),
            nn.ELU(),
        )

    @staticmethod
    def association_matrix(x: Tensor) -> Tensor:
        with torch.no_grad():
            ranks = x.argsort(dim=-1).argsort(dim=-1).to(x.dtype)
            centered_ranks = ranks - ranks.mean(dim=-1, keepdim=True)
            normalized_ranks = F.normalize(centered_ranks, p=2, dim=-1, eps=1e-8)
            association = normalized_ranks @ normalized_ranks.transpose(1, 2)
        return association

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        association = self.association_matrix(x)
        descriptor = self.projection(association.detach())
        return torch.cat((x, descriptor), dim=-1), association


class GATv2ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        heads: int,
        concatenate_heads: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if concatenate_heads and output_dim % heads != 0:
            raise ValueError("output_dim must be divisible by heads when heads are concatenated")

        head_dim = output_dim // heads if concatenate_heads else output_dim
        self.gat = GATv2Conv(
            input_dim,
            head_dim,
            heads=heads,
            concat=concatenate_heads,
            dropout=dropout,
            add_self_loops=False,
            share_weights=False,
        )
        self.residual = (
            nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
        )
        self.normalization = nn.LayerNorm(output_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.gat(x, edge_index) + self.residual(x)
        return self.dropout(self.activation(self.normalization(x)))


class GATv2Encoder(nn.Module):
    def __init__(self, config: SBAGOADConfig) -> None:
        super().__init__()
        if config.graph_layers != 3:
            raise ValueError("the manuscript model uses exactly three GATv2 layers")

        graph_input_dim = config.hidden_dim + config.association_dim
        self.blocks = nn.ModuleList(
            [
                GATv2ResidualBlock(
                    graph_input_dim,
                    config.hidden_dim,
                    config.attention_heads,
                    concatenate_heads=True,
                    dropout=config.dropout,
                ),
                GATv2ResidualBlock(
                    config.hidden_dim,
                    config.hidden_dim,
                    config.attention_heads,
                    concatenate_heads=True,
                    dropout=config.dropout,
                ),
                GATv2ResidualBlock(
                    config.hidden_dim,
                    config.hidden_dim,
                    config.attention_heads,
                    concatenate_heads=False,
                    dropout=config.dropout,
                ),
            ]
        )
        self.skip_fusion = nn.Sequential(
            nn.Linear(3 * config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
        )
        self.refinement = nn.Sequential(
            nn.Linear(config.hidden_dim, 2 * config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
        )
        self.refinement_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        layer_outputs: List[Tensor] = []
        for block in self.blocks:
            x = block(x, edge_index)
            layer_outputs.append(x)
        skip = self.skip_fusion(torch.cat(layer_outputs, dim=-1))
        return self.refinement_norm(skip + self.refinement(skip))


class SBAStudent(nn.Module):
    def __init__(self, config: SBAGOADConfig) -> None:
        super().__init__()
        self.config = config
        self.band_attention = BandAttention(
            config.num_bands,
            config.band_attention_hidden,
            config.hidden_dim,
        )
        self.cbam = CBAMFeatureRefinement(config.hidden_dim, config.cbam_reduction)
        self.feature_association = SpearmanFeatureAssociation(
            config.num_electrodes,
            config.association_dim,
        )
        self.graph_encoder = GATv2Encoder(config)
        self.classifier = nn.Sequential(
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.num_classes),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, electrodes, frequency_bands]")
        batch_size, num_nodes, num_bands = x.shape
        if num_nodes != self.config.num_electrodes or num_bands != self.config.num_bands:
            raise ValueError(
                f"expected [B, {self.config.num_electrodes}, {self.config.num_bands}], "
                f"received {tuple(x.shape)}"
            )

        x, band_weights = self.band_attention(x)
        x, feature_weights, electrode_weights = self.cbam(x)
        x, association = self.feature_association(x)

        batched_edges = _batch_edge_index(
            edge_index.to(x.device),
            batch_size=batch_size,
            num_nodes=num_nodes,
        )
        node_embeddings = self.graph_encoder(
            x.reshape(batch_size * num_nodes, -1),
            batched_edges,
        ).reshape(batch_size, num_nodes, self.config.hidden_dim)

        graph_embedding = torch.cat(
            (node_embeddings.mean(dim=1), node_embeddings.max(dim=1).values),
            dim=-1,
        )
        logits = self.classifier(graph_embedding)
        return {
            "logits": logits,
            "node_embeddings": node_embeddings,
            "graph_embedding": graph_embedding,
            "band_weights": band_weights,
            "feature_weights": feature_weights,
            "electrode_weights": electrode_weights,
            "feature_association": association,
        }


class NodeDiscriminator(nn.Module):
    """Node-wise discriminator D -> 64 -> 32 -> 1 used by local distillation."""

    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.batch_norm = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 1)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_embeddings: Tensor) -> Tensor:
        if node_embeddings.ndim != 3:
            raise ValueError("node_embeddings must have shape [batch, electrodes, hidden_dim]")
        batch_size, num_nodes, hidden_dim = node_embeddings.shape
        x = node_embeddings.reshape(batch_size * num_nodes, hidden_dim)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.batch_norm(self.fc2(x))))
        return self.fc3(x).reshape(batch_size, num_nodes)


class SBAGOAD(nn.Module):
    """Three-student SBA-GOAD model with cyclic node discriminators."""

    def __init__(self, config: SBAGOADConfig | None = None) -> None:
        super().__init__()
        self.config = config or SBAGOADConfig()
        self.students = nn.ModuleList(
            [SBAStudent(self.config) for _ in range(self.config.num_students)]
        )
        self.discriminators = nn.ModuleList(
            [
                NodeDiscriminator(
                    self.config.hidden_dim,
                    self.config.discriminator_dropout,
                )
                for _ in range(self.config.num_students)
            ]
        )
        self.ensemble_weight_logits = nn.Parameter(torch.zeros(self.config.num_students))
        self.apply(self._initialize_module)

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def ensemble_weights(self) -> Tensor:
        return F.softmax(self.ensemble_weight_logits, dim=0)

    def student_parameters(self) -> Iterator[nn.Parameter]:
        return self.students.parameters()

    def discriminator_parameters(self) -> Iterator[nn.Parameter]:
        return self.discriminators.parameters()

    def ensemble_parameters(self) -> Iterator[nn.Parameter]:
        yield self.ensemble_weight_logits

    def forward(self, x: Tensor, edge_index: Tensor) -> Dict[str, Tensor]:
        outputs = [student(x, edge_index) for student in self.students]
        student_logits = torch.stack([output["logits"] for output in outputs], dim=0)
        node_embeddings = torch.stack(
            [output["node_embeddings"] for output in outputs],
            dim=0,
        )
        ensemble_logits = torch.einsum(
            "m,mbc->bc",
            self.ensemble_weights,
            student_logits,
        )
        return {
            "ensemble_logits": ensemble_logits,
            "student_logits": student_logits,
            "node_embeddings": node_embeddings,
            "ensemble_weights": self.ensemble_weights,
            "student_outputs": outputs,
        }

    def discriminator_logits(self, node_embeddings: Tensor) -> Tensor:
        """Apply each D_n to H_n; cyclic peer selection is handled by the loss."""
        expected = (
            self.config.num_students,
            self.config.hidden_dim,
        )
        if node_embeddings.ndim != 4:
            raise ValueError(
                "node_embeddings must have shape [students, batch, electrodes, hidden_dim]"
            )
        if node_embeddings.size(0) != expected[0] or node_embeddings.size(-1) != expected[1]:
            raise ValueError("node_embeddings do not match the configured model dimensions")
        return torch.stack(
            [
                discriminator(node_embeddings[index])
                for index, discriminator in enumerate(self.discriminators)
            ],
            dim=0,
        )


__all__ = [
    "SBAGOADConfig",
    "build_spatial_knn_edge_index",
    "BandAttention",
    "CBAMFeatureRefinement",
    "SpearmanFeatureAssociation",
    "GATv2ResidualBlock",
    "GATv2Encoder",
    "SBAStudent",
    "NodeDiscriminator",
    "SBAGOAD",
]
