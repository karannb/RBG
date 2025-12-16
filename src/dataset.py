"""
Dataset for loading proteins from PDB files for ESM + SchNet models.

Two modes:
- "embedding": Returns sequences for ESM embedding generation
- "training": Returns PyG Data objects with pre-computed embeddings for SchNet
"""

import os
import glob
import torch
import numpy as np
import Bio.PDB as bpdb
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional, Literal, Union


# Standard 3-letter to 1-letter amino acid mapping
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}


def parse_pdb(pdb_file: str) -> Tuple[str, np.ndarray]:
    """
    Parse a PDB file to extract amino acid sequence and C-alpha coordinates.

    Args:
        pdb_file: Path to the PDB file

    Returns:
        sequence: Amino acid sequence as a string of one-letter codes
        ca_coords: C-alpha coordinates as numpy array of shape (num_residues, 3)
    """
    parser = bpdb.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    sequence = []
    ca_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname()
                # Skip non-standard residues (water, ligands, etc.)
                if res_name not in THREE_TO_ONE:
                    continue

                # Check if C-alpha atom exists
                if 'CA' not in residue:
                    continue

                sequence.append(THREE_TO_ONE[res_name])
                ca_coords.append(residue['CA'].get_coord())

    return "".join(sequence), np.array(ca_coords, dtype=np.float32)


class ProteinDataset(Dataset):
    TOTAL_LENGTH = 42
    TOTAL_LENGTH_MINUS_LONG = 39

    def __init__(
        self,
        data_path: str = "./data/",
        mode: Literal["embedding", "training"] = "embedding",
        embeddings_path: Optional[str] = None,
        indices: Optional[List[int]] = None,
    ):
        """
        Dataset for loading proteins from PDB files.

        Args:
            data_path: Path to directory containing PDB files and labels.txt
            mode:
                - "embedding": Returns sequences for ESM embedding generation
                - "training": Returns pre-computed embeddings + structure for SchNet
            embeddings_path: Path to directory with pre-computed embeddings (required for training mode)
        """
        self.mode = mode
        self.data_path = data_path
        self.embeddings_path = embeddings_path

        # Validate mode
        if mode not in ("embedding", "training"):
            raise ValueError(f"mode must be 'embedding' or 'training', got '{mode}'")

        if mode == "training" and embeddings_path is None:
            raise ValueError("embeddings_path is required for training mode")

        # Load pdb files
        if mode == "embedding":
            self.pdb_files = sorted(glob.glob(os.path.join(data_path, "*.pdb")))
            if indices is not None:
                self.pdb_files = [self.pdb_files[i] for i in indices]
        else:
            self.pdb_files = sorted(glob.glob(os.path.join(embeddings_path, "*.pt")))
            if indices is not None:
                self.pdb_files = [self.pdb_files[i] for i in indices]

        # Load labels
        labels_file = os.path.join(data_path, "labels.txt")
        with open(labels_file, "r") as f:
            lines = f.readlines()

        all_labels = {line.split()[0]: line.split()[1] for line in lines}

        # Filter labels to only include PDB files we're using
        pdb_names = {os.path.basename(f).replace(".pdb", "").replace(".pt", "") for f in self.pdb_files}
        self.labels = {k: v for k, v in all_labels.items() if k in pdb_names}

        assert len(self.pdb_files) == len(self.labels), \
            f"Mismatch between PDB files ({len(self.pdb_files)}) and labels ({len(self.labels)})"

    def __len__(self) -> int:
        return len(self.pdb_files)

    def __getitem__(self, idx: int) -> Union[Dict, Data]:
        pdb_file = self.pdb_files[idx]
        pdb_name = os.path.basename(pdb_file).replace(".pdb", "").replace(".pt", "")

        if self.mode == "embedding":
            return self._get_embedding_item(pdb_file, pdb_name)
        else:  # training mode
            return self._get_training_item(pdb_file, pdb_name)

    def _get_embedding_item(self, pdb_file: str, pdb_name: str) -> Dict:
        """
        Returns data needed for ESM embedding generation.
        """
        sequence, ca_coords = parse_pdb(pdb_file)

        return {
            "sequence": sequence,
            "coords": torch.from_numpy(ca_coords),  # (N, 3) - needed to save alongside embeddings
            "pdb_name": pdb_name,
        }

    def _get_training_item(self, pdb_file: str, pdb_name: str) -> Data:
        """
        Returns PyG Data object for SchNet training with pre-computed embeddings.
        """
        # Load pre-computed embeddings
        embedding_file = os.path.join(self.embeddings_path, f"{pdb_name}.pt")
        embedding_data = torch.load(embedding_file, weights_only=True)

        # Embeddings should contain: embeddings (N, D), coords (N, 3)
        embeddings = embedding_data["embeddings"]  # (N, hidden_dim)
        ca_coords = embedding_data["coords"]  # (N, 3)

        # Compute graph structure
        if isinstance(ca_coords, torch.Tensor):
            ca_coords_np = ca_coords.numpy()
        else:
            ca_coords_np = ca_coords

        # Get label
        label = self.labels.get(pdb_name)

        # Return PyG Data object - batching handled by PyG DataLoader
        return Data(
            x=embeddings,  # (N, hidden_dim)
            pos=torch.from_numpy(ca_coords_np).float() if isinstance(ca_coords_np, np.ndarray) else ca_coords.float(),
            y=torch.tensor([float(label)]) if label is not None else None,
            pdb_name=pdb_name,
        )


class EmbeddingCollator:
    """
    Collate function for embedding mode.
    """
    def __init__(self, tokenizer=None, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate function for embedding mode.

        Args:
            batch: List of samples from ProteinDataset in embedding mode
            tokenizer: HuggingFace ESM tokenizer
            max_length: Maximum sequence length for tokenization

        Returns:
            Batched dictionary ready for ESM model
        """
        sequences = [item["sequence"] for item in batch]
        pdb_names = [item["pdb_name"] for item in batch]
        coords_list = [item["coords"] for item in batch]

        # Tokenize sequences with padding
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
        else:
            raise ValueError("tokenizer is required for embedding mode")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "coords": coords_list,  # List of variable-length tensors
            "pdb_names": pdb_names
        }


if __name__ == "__main__":
    # Print stats of the dataset
    dataset = ProteinDataset(mode="embedding")

    print(f"Number of samples: {len(dataset)}")
    print(f"Sample PDB names: {[os.path.basename(f).replace('.pdb', '') for f in dataset.pdb_files[:5]]}")

    # Compute sequence length statistics
    seq_lengths = []
    for i in range(len(dataset)):
        item = dataset[i]
        seq_lengths.append(len(item["sequence"]))

    seq_lengths = np.array(seq_lengths)
    print(f"\nSequence length statistics:")
    print(f"  Min: {seq_lengths.min()}")
    print(f"  Max: {seq_lengths.max()}")
    print(f"  Mean: {seq_lengths.mean():.1f}")
    print(f"  Median: {np.median(seq_lengths):.1f}")
    print(f"  Std: {seq_lengths.std():.1f}")

    # Plot sequence length distribution
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(seq_lengths, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(seq_lengths.mean(), color='red', linestyle='--', label=f'Mean: {seq_lengths.mean():.1f}')
    ax.axvline(np.median(seq_lengths), color='orange', linestyle='--', label=f'Median: {np.median(seq_lengths):.1f}')
    ax.set_xlabel('Sequence Length (residues)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Protein Sequence Lengths')
    ax.legend()
    plt.tight_layout()
    plt.savefig('sequence_length_distribution.png', dpi=300)
    print("\nPlot saved to 'sequence_length_distribution.png'")

