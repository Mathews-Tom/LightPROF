from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# Type alias for reasoning subgraphs
Subgraph = List[List[Tuple[str, str, str]]]

# Assuming utils is in the same parent directory or accessible
from .utils import get_gemini_tokenizer  # Using the tokenizer from utils



class KnowledgeAdapter(nn.Module):
    """
    Converts a reasoning subgraph (list of paths) into compact, fused embeddings
    that match the LLM input space.

    This involves tokenizing and embedding triples, encoding structural information,
    and fusing/projecting these embeddings.
    """

    def __init__(
        self,
        bert_model_name: str = "bert-base-uncased",
        structural_mlp_hidden_dim: int = 64,
        llm_embedding_dim: int = 768,  # Assuming LLM embedding dim is same as BERT for now
    ):
        """
        Initializes the KnowledgeAdapter.

        Args:
            bert_model_name (str): Name of the BERT model to use for text embedding.
                                    Defaults to 'bert-base-uncased'.
            structural_mlp_hidden_dim (int): Hidden dimension size for the structural encoding MLP.
                                                Defaults to 64.
            llm_embedding_dim (int): The target embedding dimension for the LLM input space.
                                        Defaults to 768 (BERT base output size).
        """
        super().__init__()

        # 1. BERT Encoder for Text Embedding (Frozen)
        self.tokenizer = get_gemini_tokenizer()  # Use the tokenizer from utils
        self.text_encoder = AutoModel.from_pretrained(bert_model_name)
        # Freeze the BERT model parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.bert_embedding_dim = self.text_encoder.config.hidden_size

        # 2. MLP for Structural Information Encoding
        # We will encode the position within the triple (head, relation, tail)
        # Using an embedding layer for the 3 positions (0, 1, 2)
        self.structural_embedding = nn.Embedding(3, structural_mlp_hidden_dim)
        self.structural_mlp = nn.Sequential(
            nn.Linear(structural_mlp_hidden_dim, structural_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                structural_mlp_hidden_dim, structural_mlp_hidden_dim
            ),  # Output dim for structural info
        )
        self.structural_embedding_dim = structural_mlp_hidden_dim

        # 3. Fusion and Projection Layer
        # Concatenate text embedding and structural embedding
        fused_input_dim = self.bert_embedding_dim + self.structural_embedding_dim
        self.fusion_projection = nn.Linear(fused_input_dim, llm_embedding_dim)

        self.llm_embedding_dim = llm_embedding_dim

    def forward(
        self, reasoning_subgraph: List[List[Tuple[str, str, str]]]
    ) -> torch.Tensor:
        """
        Processes a reasoning subgraph (list of paths) and converts it into fused embeddings.

        Args:
            reasoning_subgraph (List[List[Tuple[str, str, str]]]): A list of paths,
                                                                    where each path is a list of triples
                                                                    (head, relation, tail).

        Returns:
            torch.Tensor: A tensor of shape [N x E], where N is the total number of
                            elements (entities/relations) across all paths, and E is
                            the LLM embedding dimension.
        """
        all_element_embeddings: List[torch.Tensor] = []

        # Iterate through each path in the subgraph
        for path in reasoning_subgraph:
            # Iterate through each triple in the path
            for triple_idx, triple in enumerate(path):
                head, relation, tail = triple

                # Process each element (head, relation, tail) in the triple
                for element_idx, element_text in enumerate([head, relation, tail]):
                    # Tokenize and get text embedding
                    # Add batch dimension for BERT input
                    encoded_input = self.tokenizer(
                        element_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.tokenizer.model_max_length,
                    )
                    # Move tensors to the same device as the model
                    encoded_input = {
                        k: v.to(self.text_encoder.device)
                        for k, v in encoded_input.items()
                    }

                    with torch.no_grad():  # Ensure BERT is frozen
                        text_output = self.text_encoder(**encoded_input)
                    # Use the [CLS] token embedding as the representation
                    text_embedding = text_output.last_hidden_state[
                        :, 0, :
                    ]  # Shape: [1, bert_embedding_dim]

                    # Encode structural information (position within the triple)
                    structural_input = torch.tensor([element_idx], dtype=torch.long).to(
                        self.structural_embedding.weight.device
                    )
                    structural_emb = self.structural_embedding(
                        structural_input
                    )  # Shape: [1, structural_mlp_hidden_dim]
                    structural_encoding = self.structural_mlp(
                        structural_emb
                    )  # Shape: [1, structural_mlp_hidden_dim]

                    # Fuse structural and textual embeddings
                    fused_embedding = torch.cat(
                        (text_embedding, structural_encoding), dim=1
                    )  # Shape: [1, fused_input_dim]

                    # Project to LLM embedding space
                    projected_embedding = self.fusion_projection(
                        fused_embedding
                    )  # Shape: [1, llm_embedding_dim]

                    # Append to the list of all element embeddings
                    all_element_embeddings.append(projected_embedding)

        # Stack all collected embeddings into a single tensor
        if not all_element_embeddings:
            # Return an empty tensor with the correct dimension if no paths/elements were processed
            return torch.empty(0, self.llm_embedding_dim)

        fused_embeddings_tensor = torch.cat(
            all_element_embeddings, dim=0
        )  # Shape: [N, llm_embedding_dim]

        return fused_embeddings_tensor


# Example Usage (for testing the module in isolation)
if __name__ == "__main__":
    # Create a dummy reasoning subgraph (list of paths)
    # Path 1: entityA -> relationX -> entityB
    # Path 2: entityA -> relationZ -> entityC
    dummy_subgraph = [
        [("entityA", "relationX", "entityB")],
        [("entityA", "relationZ", "entityC")],
    ]

    # Initialize the KnowledgeAdapter
    # Using default parameters, assuming BERT base and output dim 768
    adapter = KnowledgeAdapter()

    # Process the dummy subgraph
    fused_embeddings = adapter(dummy_subgraph)

    print(f"Input subgraph: {dummy_subgraph}")
    print(f"Output fused embeddings shape: {fused_embeddings.shape}")
    # Expected shape: [Number of elements (3+3=6), LLM embedding dim (768)]
    assert fused_embeddings.shape == (6, adapter.llm_embedding_dim)

    print("\nKnowledgeAdapter initialized and forward pass completed successfully.")
