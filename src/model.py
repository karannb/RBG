"""
Custom SchNet-like model that accepts arbitrary node embeddings instead of atomic numbers.

Based on PyG's SchNet implementation but replaces the atomic number embedding
with a linear projection for pre-computed embeddings (e.g., from ESM).
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.nn.models.schnet import (
    InteractionBlock,
    GaussianSmearing,
    ShiftedSoftplus,
    RadiusInteractionGraph,
)


class EmbeddingSchNet(nn.Module):
    """
    A SchNet-like model that accepts pre-computed embeddings (e.g., from ESM)
    instead of atomic numbers.

    The key difference from standard SchNet:
    - Instead of an Embedding layer for atomic numbers, we use a linear projection
      to map input embeddings to the hidden dimension.

    Args:
        input_dim: Dimension of input node embeddings (e.g., 1280 for ESM-2 large)
        hidden_channels: Hidden embedding size
        num_filters: Number of filters in continuous-filter convolution
        num_interactions: Number of interaction blocks
        num_gaussians: Number of Gaussian functions for distance expansion
        cutoff: Cutoff distance for interatomic interactions
        interaction_graph: Optional custom function to compute edges
        max_num_neighbors: Maximum number of neighbors per node
        readout: Global aggregation method ('add' or 'mean')
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        max_num_neighbors: int = 32,
        num_gaussians: int = 50,
        cutoff: float = 20.0,
        readout: str = 'add',
    ):
        super().__init__()


        self.readout = aggr_resolver(readout)

        # Project input embeddings to hidden dimension (replaces atomic number embedding)
        self.input_proj = nn.Linear(input_dim, hidden_channels)

        # Interaction graph (same as SchNet)
        self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        # Distance expansion using Gaussian basis functions
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        # Interaction blocks (same as SchNet)
        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(
                hidden_channels,
                num_gaussians,
                num_filters,
                cutoff
            )
            self.interactions.append(block)

        # Output network
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = nn.Linear(hidden_channels // 2, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x: Node embeddings with shape [num_nodes, input_dim]
            pos: Node positions with shape [num_nodes, 3]
            batch: Batch indices with shape [num_nodes] (optional)

        Returns:
            Predicted property with shape [num_graphs, 1] or [1] if single graph
        """
        # Handle single graph case
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device) if batch is None else batch

        # Project input embeddings to hidden dimension
        h = self.input_proj(x)

        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        # Per-node output
        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        out = self.readout(h, batch, dim=0)

        return out


class Model(nn.Module):
    def __init__(self, input_dim: int = 1280, hidden_dim: int = 128, output_dim: int = 1, cutoff: float = 20.0):
        """
        Wrapper around SchNet model for specific input/output dimensions.

        Args:
            input_dim: Dimension of input node embeddings (1280 for ESM-2 large)
            hidden_dim: Hidden embedding size (e.g., 128)
            output_dim: Dimension of output property (1 for regression)
            cutoff: Cutoff distance for interatomic interactions (20.0 for this task)
        """
        super().__init__()
        self.schnet = EmbeddingSchNet(
            input_dim=input_dim,
            hidden_channels=hidden_dim,
            cutoff=cutoff
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor, pos: Tensor, batch: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through SchNet and output layer.

        Args:
            x: Node embeddings with shape [num_nodes, input_dim]
            pos: Node positions with shape [num_nodes, 3]
            batch: Batch indices with shape [num_nodes] (optional)

        Returns:
            Predicted property with shape [num_graphs, 1] or [1] if single graph
        """
        schnet_out = self.schnet(x, pos, batch) # (num_graphs, hidden_dim)
        out = self.output_layer(schnet_out) # (num_graphs, output_dim)
        return out
