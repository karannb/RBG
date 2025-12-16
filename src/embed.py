"""
Create protein embeddings using ESM2.
"""

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.dataset import ProteinDataset, EmbeddingCollator


MODEL = "facebook/esm2_t33_650M_UR50D"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def embed(model, dataloader, save_path: str):
    """
    Generate and save embeddings for proteins in the dataloader using the provided ESM2 model.

    Args:
        model: Pre-trained ESM2 model
        dataloader: DataLoader providing batches of protein sequences
        save_path: Directory to save the embeddings
    """
    model.eval()
    os.makedirs(save_path, exist_ok=True)

    for batch in tqdm(dataloader):
        pdb_names = batch["pdb_names"]

        # Move input tensors to device
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # Get embeddings from the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, L, D)

        for i, pdb_name in enumerate(pdb_names):
            seq_len = (attention_mask[i] == 1).sum().item()  # Actual length of the sequence
            if seq_len == 1024:
                print(f"Warning: Sequence {pdb_name} truncated to max length 1024, and thus not saved.")
                continue

            embeddings = token_embeddings[i, 1:seq_len-1, :].cpu()  # Exclude CLS and EOS tokens

            # Save embeddings and coordinates
            embedding_data = {
                "embeddings": embeddings.to("cpu"),  # (N, D)
                "coords": batch["coords"][i],  # (N, 3)
            }
            embedding_file = os.path.join(save_path, f"{pdb_name}.pt")
            torch.save(embedding_data, embedding_file)
            print(f"Saved embeddings for {pdb_name} to {embedding_file}")


def main():

    # Data Loading routines
    dataset = ProteinDataset(mode="embedding")
    collator = EmbeddingCollator(
        tokenizer=AutoTokenizer.from_pretrained(MODEL),
        max_length=1024,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collator,
    )

    # Load pre-trained ESM model
    model = AutoModel.from_pretrained(MODEL)
    model.to(DEVICE)

    # Generate and save embeddings
    embed(model, dataloader, save_path="./embeddings")


if __name__ == "__main__":
    main()
