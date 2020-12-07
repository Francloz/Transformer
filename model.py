import math

from torch.nn.modules.transformer import *
from torch.nn.modules.transformer import _get_activation_fn

BASE_MODEL = {
    'd_model': 512,
    'n_heads': 8,
    'n_duplicates': 6,
    'd_feedforward': 2048,
    'dropout': 0.1,
}

BIG_MODEL = {
    'd_model': 1024,
    'n_heads': 16,
    'n_duplicates': 6,
    'd_feedforward': 4096,
    'dropout': 0.3,
}


class AIAYNTransformerEncoderLayer(torch.nn.Module):
    r"""
    Attempt to recreate the exact copy of the transformer encoder layer shown in the AIAYN paper.

    For this, this code is practically the same as found in torch.nn.modules.transformer, with some dropout layers
    removed.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(AIAYNTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = _get_activation_fn('relu')

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(AIAYNTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Multi-head attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # Add & Norm
        src = self.norm1(src + self.dropout1(src2))
        # FF
        src2 = self.linear2(self.activation(self.linear1(src)))
        # Add & Norm
        src = self.norm2(src + self.dropout2(src2))
        return src


class AIAYNTransformerDecoderLayer(torch.nn.Module):
    r"""
    Attempt to recreate the exact copy of the transformer decoder layer shown in the AIAYN paper.

    For this, this code is practically the same as found in torch.nn.modules.transformer, with some dropout layers
    removed.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(AIAYNTransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.multihead_attn = MultiheadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn('relu')

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(AIAYNTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # Target attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        # Middle attention (mixed)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        # Output attention
        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class AIAYNTransformer(torch.nn.Module):
    """
    Complete Transformer model exactly as in the paper Attention is all you need (AIAYN).

    It contains the embeddings, positional encoding and the final softmax function.
    For more details check https://arxiv.org/pdf/1706.03762.pdf.
    """

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def __init__(self, vocab_size, d_model, n_heads, n_duplicates, d_feedforward, p_dropout):
        super(AIAYNTransformer, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)

        encoder_layer = AIAYNTransformerEncoderLayer(d_model, n_heads, d_feedforward, p_dropout)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, n_duplicates, encoder_norm)

        decoder_layer = AIAYNTransformerDecoderLayer(d_model, n_heads, d_feedforward, p_dropout)
        decoder_norm = LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, n_duplicates, decoder_norm)

        self.dropout1 = Dropout(p_dropout)
        self.dropout2 = Dropout(p_dropout)

        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=n_duplicates,
            num_decoder_layers=n_duplicates,
            dim_feedforward=d_feedforward,
            dropout=p_dropout,
            activation='relu',
            custom_encoder=self.encoder,
            custom_decoder=self.decoder)

        self.linear = Linear(d_model, vocab_size)
        self.softmax = F.softmax

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        # src_mask = get_pad_mask(src)

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.shape[0]).to(tgt.device)
        # tgt_mask2 = get_subsequent_mask(tgt)  # get_pad_mask(tgt) & get_subsequent_mask(tgt)

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        src = self.dropout1(src)
        tgt = self.dropout2(tgt)

        out = self.transformer.forward(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask,
                                       tgt_key_padding_mask, memory_key_padding_mask)

        out = self.linear(out)
        out = self.softmax(out, dim=2)

        return out


class ScheduledOptim:
    '''A simple wrapper class for learning rate scheduling
     Exact copy from https://github.com/jadore801120/attention-is-all-you-need-pytorch'''

    def __init__(self, optimizer, init_lr, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return (self.d_model ** -0.5) * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        ''' Returns the state dict to save and load '''
        return {'state': self._optimizer.state_dict(),
                'schedule': [self.init_lr, self.d_model, self.n_warmup_steps, self.n_steps]}

    def load_state_dict(self, state):
        val = state['schedule']
        self._optimizer.load_state_dict(state['state'])
        self.init_lr = val[0]
        self.d_model = val[1]
        self.n_warmup_steps = val[2]
        self.n_steps = val[3]
