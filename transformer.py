import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from model_utils import _generate_square_subsequent_mask, Embedding, PositionalEncoding
from tokenizer import PAD_ID

class Transformer(nn.Module):
    def __init__(self, src_vocab_size=128, tgt_vocab_size=128,
                 embedding_dim=128, fcn_hidden_dim=128,
                 num_heads=4, num_layers=2, dropout=0.2):
        super(Transformer, self).__init__()

        self.embedding_dim = embedding_dim
        # Source and Encoder layers
        self.src_embed = Embedding(src_vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.src_pos_encoder = PositionalEncoding(embedding_dim)
        encoder_layer = TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                   dim_feedforward=fcn_hidden_dim, dropout=dropout)
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        # Target and Decoder layers
        self.tgt_embed = Embedding(tgt_vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.tgt_pos_encoder = PositionalEncoding(embedding_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads,
                                                   dim_feedforward=fcn_hidden_dim, dropout=dropout)
        decoder_norm = nn.LayerNorm(embedding_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)
        # Final linear layer
        self.final_out = nn.Linear(embedding_dim, tgt_vocab_size)

        # Initialize masks
        self.src_mask = None
        self.tgt_mask = None
        self.mem_mask = None
        # Initialize weights of model
        self._reset_parameters()

    def _reset_parameters(self):
        """ Initiate parameters in the transformer model. """
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def encode(self, src, src_key_padding_mask=None):
        """
        Applies embedding, positional encoding and then runs the transformer encoder on the source
        :param src: source tokens batch
        :param src_key_padding_mask: source padding mask
        :return: memory- the encoder hidden states
        """
        # Source embedding and positional encoding, changes dimension (N, S) -> (N, S, E) -> (S, N, E)
        src_embed = self.src_embed(src).transpose(0, 1)
        src_embed = self.src_pos_encoder(src_embed)
        # Pass the source to the encoder
        memory = self.encoder(src_embed, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, memory, tgt, tgt_key_padding_mask=None, memory_key_padding_mask=None, has_mask=True):
        """
        Applies embedding, positional encoding on target  and then runs the transformer encoder on the memory and target.
        Also creates square subsequent mask for teacher learning.
        :param memory: The encoder hidden states
        :param tgt: Target tokens batch
        :param tgt_key_padding_mask: target padding mask
        :param memory_key_padding_mask: memory padding mask
        :param has_mask: Whether to use square subsequent mask for teacher learning
        :return: decoder output
        """
        # Create target mask for transformer if no appropriate one was created yet, created of size (T, T)
        if has_mask:
            if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.size(1):
                self.tgt_mask = _generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        else:
            self.tgt_mask = None
        # Target embedding and positional encoding, changes dimension (N, T) -> (N, T, E) -> (T, N, E)
        tgt_embed = self.tgt_embed(tgt).transpose(0, 1)
        tgt_embed = self.tgt_pos_encoder(tgt_embed)
        # Get output of decoder. Dimensions stay the same
        decoder_output = self.decoder(tgt_embed, memory, tgt_mask=self.tgt_mask, memory_mask=self.mem_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
        # Add linear layer & log softmax, (T, N, E) -> (T, N, tgt_vocab_size)
        output = F.log_softmax(self.final_out(decoder_output), dim=-1)
        # Change back batch and sequence dimensions, from (T, N, tgt_vocab_size) -> (N, T, tgt_vocab_size)
        return output.transpose(0, 1)


    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, has_mask=True):
        """Take in and process masked source/target sequences.

		Args:
			src: the sequence to the encoder (required).
			tgt: the sequence to the decoder (required).
			src_mask: the additive mask for the src sequence (optional).
			tgt_mask: the additive mask for the tgt sequence (optional).
			memory_mask: the additive mask for the encoder output (optional).
			src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).

		Shape:
			- src: :math:`(S, N, E)`. Starts as (N, S) and changed after embedding
			- tgt: :math:`(T, N, E)`. Starts as (N, T) and changed after embedding
			- src_mask: :math:`(S, S)`.
			- tgt_mask: :math:`(T, T)`.
			- memory_mask: :math:`(T, S)`.
			- src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.

			Note: [src/tgt/memory]_mask should be filled with
			float('-inf') for the masked positions and float(0.0) else. These masks
			ensure that predictions for position i depend only on the unmasked positions
			j and are applied identically for each sequence in a batch.
			[src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
			that should be masked with float('-inf') and False values will be unchanged.
			This mask ensures that no information will be taken from position i if
			it is masked, and has a separate mask for each sequence in a batch.

			- output: :math:`(T, N, E)`.

			Note: Due to the multi-head attention architecture in the transformer model,
			the output sequence length of a transformer is same as the input sequence
			(i.e. target) length of the decode.

			where S is the source sequence length, T is the target sequence length, N is the
			batch size, E is the feature number

		Examples:
			output = transformer_model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
		"""
        # Applies embedding, positional encoding and the transformer encoder on the source
        memory = self.encode(src, src_key_padding_mask)
        # Applies embedding, positional encoding on target  and then runs the transformer encoder on the memory and target.
        output = self.decode(memory, tgt, tgt_key_padding_mask, memory_key_padding_mask, has_mask)
        return output
