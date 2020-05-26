import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

from model_utils import PositionalEncoding, _generate_square_subsequent_mask, Embedding
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer
from decoder import TransformerDecoder, TransformerDecoderFinalLayer


class PointerGeneratorTransformer(nn.Module):
    def __init__(self, src_vocab_size=128, tgt_vocab_size=128,
                 embedding_dim=128, fcn_hidden_dim=128,
                 num_heads=4, num_layers=2, dropout=0.2,
                 src_to_tgt_vocab_conversion_matrix=None):
        super(PointerGeneratorTransformer, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_dim = embedding_dim
        self.src_to_tgt_vocab_conversion_matrix = src_to_tgt_vocab_conversion_matrix
        self.pos_encoder = PositionalEncoding(embedding_dim)
        # Source and target embeddings
        self.src_embed = Embedding(self.src_vocab_size, embedding_dim, padding_idx=2)
        self.tgt_embed = Embedding(self.tgt_vocab_size, embedding_dim, padding_idx=2)

        # Encoder layers
        self.encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, fcn_hidden_dim, dropout)
        self.encoder = TransformerEncoder(self.encoder_layer, num_layers)

        # Decoder layers
        self.decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, fcn_hidden_dim, dropout)
        self.decoder_final_layer = TransformerDecoderFinalLayer(embedding_dim, num_heads, fcn_hidden_dim, dropout)
        self.decoder = TransformerDecoder(self.decoder_layer, self.decoder_final_layer, num_layers)

        # Final linear layer + softmax. for probability over target vocabulary
        self.p_vocab = nn.Sequential(
            nn.Linear(self.embedding_dim, self.tgt_vocab_size),
            nn.Softmax(dim=-1))

        # P_gen, probability of generating output
        self.p_gen = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 1),
            nn.Sigmoid())
        # Context vector
        self.c_t = None

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
        src_embed = self.pos_encoder(src_embed)
        # Pass the source to the encoder
        memory = self.encoder(src_embed, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, memory, tgt, src, tgt_key_padding_mask=None, memory_key_padding_mask=None, has_mask=True):
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
        tgt_embed_pos = self.pos_encoder(tgt_embed)
        # Get output of decoder and attention weights. decoder Dimensions stay the same
        decoder_output, attention = self.decoder(tgt_embed_pos, memory, tgt_mask=self.tgt_mask,
                                                 memory_mask=self.mem_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask)
        # Get probability over target vocabulary, (T, N, E) -> (T, N, tgt_vocab_size)
        p_vocab = self.p_vocab(decoder_output)

        # ---Compute Pointer Generator probability---
        # Get hidden states of source (easier/more understandable computation). (S, N, E) -> (N, S, E)
        hidden_states = memory.transpose(0, 1)
        # compute context vectors. (N, T, S) x (N, S, E) -> (N, T, E)
        context_vectors = torch.matmul(attention, hidden_states).transpose(0, 1)
        total_states = torch.cat((context_vectors, decoder_output, tgt_embed), dim=-1)
        # Get probability of generating output. (N, T, 3*E) -> (N, T, 1)
        p_gen = self.p_gen(total_states)
        # Get probability of copying from input. (N, T, 1)
        p_copy = 1 - p_gen

        # Get representation of src tokens as one hot encoding
        one_hot = torch.zeros(src.size(0), src.size(1), self.src_vocab_size, device=src.device)
        one_hot = one_hot.scatter_(dim=-1, index=src.unsqueeze(-1), value=1)
        # p_copy from source is sum over all attention weights for each token in source
        p_copy_src_vocab = torch.matmul(attention, one_hot)
        # convert representation of token from src vocab to tgt vocab
        p_copy_tgt_vocab = torch.matmul(p_copy_src_vocab, self.src_to_tgt_vocab_conversion_matrix).transpose(0,
                                                                                                                  1)
        # Compute final probability
        p = torch.add(p_vocab * p_gen, p_copy_tgt_vocab * p_copy)

        # Change back batch and sequence dimensions, from (T, N, tgt_vocab_size) -> (N, T, tgt_vocab_size)
        return torch.log(p.transpose(0, 1))

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, has_mask=True):
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
        output = self.decode(memory, tgt, src, tgt_key_padding_mask, memory_key_padding_mask, has_mask)
        return output




# def get_context_vectors_1(self, hidden_states, attention, N, T):
#     """ compute context vectors using hidden states and attention over the source """
#     # Replace source and embedding dimension
#     hidden_states = hidden_states.transpose(1, 2)
#     context_vectors = torch.zeros(N, T, self.embedding_dim).type(torch.float32)
#     # Get context vector
#     for i in range(N):  # go over each data sample i in batch
#         h_t_i = hidden_states[i]  # all hidden_states - E x S
#         for j in range(T):  # go over each target token j
#             attn_i_j = attention[i][j]  # attention over source for target token j - S
#             context_vectors[i][j] = torch.mv(h_t_i, attn_i_j)
#     context_vectors = context_vectors.transpose(0, 1)
#     return context_vectors.type(torch.float32)

# one_hot_cat = torch.cat(T * [one_hot]).view(N, T, S, self.src_vocab_size)
        # one_hot_cat = one_hot.unsqueeze(1).repeat(1, T, 1, 1)

# def one_hot_encoding(src, src_vocab_size):
#     one_hot = torch.zeros(src.size(0), src.size(1), src_vocab_size)
#     src_ext = src.unsqueeze_(-1)
#     one_hot = one_hot.scatter_(dim=-1, index=src_ext, value=1)
#     return one_hot
