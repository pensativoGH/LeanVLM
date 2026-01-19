"""Language model wrapper for SmolLM2 variants."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LanguageModel(nn.Module):
    """
    Wrapper around SmolLM2 models for causal language modeling.

    Supports SmolLM2-135M, SmolLM2-360M-Instruct, etc.
    Hidden size is dynamically read from model config.

    Provides access to:
    - Embedding layer for getting text embeddings
    - Forward pass for computing logits/loss
    - Generate method for inference
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-135M",
        freeze: bool = False,
    ):
        super().__init__()

        # Load the causal LM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Get config values
        self.hidden_size = self.model.config.hidden_size  # 576
        self.vocab_size = self.model.config.vocab_size    # 49152

        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all parameters in the language model."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters in the language model."""
        for param in self.model.parameters():
            param.requires_grad = True

    def freeze_partial(self, unfreeze_layers: list = None):
        """
        Partially freeze the language model.

        Freezes all layers except those specified in unfreeze_layers.
        Useful for Stage 1 training where we want to train embeddings + first block.

        Args:
            unfreeze_layers: List of layers to keep trainable.
                            - "embed": embedding layer
                            - 0, 1, 2, ...: transformer layer indices
                            Default: ["embed", 0] (embeddings + first block)
        """
        if unfreeze_layers is None:
            unfreeze_layers = ["embed", 0]

        # First freeze everything
        self.freeze()

        # SmolLM2 structure: model.model.embed_tokens, model.model.layers[i]
        inner_model = self.model.model

        # Unfreeze specified layers
        for layer_spec in unfreeze_layers:
            if layer_spec == "embed":
                # Unfreeze embedding layer
                for param in inner_model.embed_tokens.parameters():
                    param.requires_grad = True
                print("  Unfroze: embed_tokens")
            elif isinstance(layer_spec, int):
                # Unfreeze transformer layer by index
                if layer_spec < len(inner_model.layers):
                    for param in inner_model.layers[layer_spec].parameters():
                        param.requires_grad = True
                    print(f"  Unfroze: layer {layer_spec}")
                else:
                    print(f"  Warning: layer {layer_spec} doesn't exist (max: {len(inner_model.layers)-1})")

        # Count trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  LM trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def get_input_embeddings(self) -> nn.Module:
        """Get the input embedding layer."""
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        """Get the output embedding layer (lm_head)."""
        return self.model.get_output_embeddings()

    def resize_token_embeddings(self, new_num_tokens: int) -> None:
        """
        Resize token embeddings to accommodate new tokens.

        This resizes both the input embeddings (embed_tokens) and
        output embeddings (lm_head) to the new vocabulary size.

        Args:
            new_num_tokens: New vocabulary size
        """
        self.model.resize_token_embeddings(new_num_tokens)
        self.vocab_size = new_num_tokens
        print(f"  Resized LM embeddings to {new_num_tokens}")

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for input token IDs.

        Args:
            input_ids: (B, seq_len) tensor of token IDs

        Returns:
            (B, seq_len, 576) tensor of embeddings
        """
        return self.get_input_embeddings()(input_ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass through the language model.

        Args:
            inputs_embeds: (B, seq_len, 576) tensor of embeddings
            attention_mask: (B, seq_len) attention mask
            labels: (B, seq_len) token IDs for loss computation (-100 for ignored)

        Returns:
            CausalLMOutput with loss and logits
        """
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
    ):
        """
        Generate text given input embeddings.

        Args:
            inputs_embeds: (B, seq_len, 576) tensor of embeddings
            attention_mask: (B, seq_len) attention mask
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )


def get_tokenizer(model_name: str = "HuggingFaceTB/SmolLM2-135M"):
    """
    Get the tokenizer for SmolLM2.

    Args:
        model_name: HuggingFace model name

    Returns:
        AutoTokenizer with padding configured
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer
