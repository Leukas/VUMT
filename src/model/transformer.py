# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Adapted from fairseq-py

from logging import getLogger
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.layer_norm import LayerNorm
from ..modules.multihead_attention import MultiheadAttention
from ..modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from ..sequence_generator import SequenceGenerator

from . import LatentState


logger = getLogger()


def embedding_noise(emb, emb_mask, alpha=0.5):
    """ Embedding noise based on batch standard deviation """
    if alpha == 0:
        return emb
    real_word_embs = emb[1-emb_mask.t()]
    stds = real_word_embs.std(dim=0).to(emb.device)
    mags = torch.randn(real_word_embs.size()).to(emb.device)
    noise = (stds * mags * alpha)
    emb[1-emb_mask.t()] += noise
    return emb



class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    ENC_ATTR = ['n_langs', 'n_words', 'dropout', 'padding_idx']

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout

        self.embed_noise_alpha = args.embed_noise_alpha
        self.sample_dist = args.sample_dist
        self.word_neighbor = args.word_neighbor
        self.word_neighbor_dist = args.word_neighbor_dist

        self.n_langs = args.n_langs
        self.n_words = args.n_words
        embed_dim = args.encoder_embed_dim
        if args.share_lang_emb:
            assert len(set(args.n_words)) == 1
            logger.info("Sharing encoder input embeddings")
            layer_0 = Embedding(args.n_words[0], embed_dim, args.pad_index)
            embeddings = [layer_0 for _ in range(self.n_langs)]
        else:
            embeddings = [Embedding(n_words, embed_dim, padding_idx=args.pad_index) for n_words in self.n_words]
        self.embeddings = nn.ModuleList(embeddings)
        self.freeze_enc_emb = args.freeze_enc_emb

        self.padding_idx = args.pad_index
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=args.left_pad_source,
        )

        self.layers = nn.ModuleList([None for i in range(args.encoder_layers)])
        for k in range(args.encoder_layers):
            # share top share_enc layers
            layer_is_shared = (k >= (args.encoder_layers - args.share_enc))
            if layer_is_shared:
                logger.info("Sharing encoder transformer parameters for layer %i" % k)

            self.layers[k] = nn.ModuleList([
                # layer for first lang
                TransformerEncoderLayer(args)
            ])
            for i in range(1, self.n_langs):
                # layer for lang i
                if layer_is_shared:
                    # share layer from lang 0
                    self.layers[k].append(self.layers[k][0])
                else:
                    self.layers[k].append(TransformerEncoderLayer(args))

        self.variational = args.variational
        if self.variational:
            self.var_ff = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim*2)

    def neighbor_noise(self, src_tokens, lang_id):
        # src tokens: len x batch_size

        sen_len = src_tokens.size(0)
        batch_size = src_tokens.size(1)

        embed_tokens = self.embeddings[lang_id] # emb.weight: vocab_size x emb_dim
        src_emb = embed_tokens(src_tokens) # len x batch x emb_dim
        src_emb = src_emb.view(-1, src_emb.size(2)) # (len x batch) x emb_dim
        sims = src_emb @ embed_tokens.weight.t() # (len x batch) x vocab_size

        # normalize similarities
        emb_sums = torch.clamp(torch.sqrt(torch.sum(embed_tokens.weight**2, dim=1)), min=1e-8) # vocab_size
        sims = sims / emb_sums
        src_sums = torch.clamp(torch.sqrt(torch.sum(src_emb**2, dim=1)), min=1e-8) # (len x batch)
        sims = (sims.t() / src_sums).t()

        sims = sims.view(sen_len, batch_size, -1) # len x batch x vocab_size

        neighbors = torch.topk(sims, dim=2, k=self.word_neighbor_dist+1)[1] # len x batch x k-neighbors (+1 to ignore same word)
        choices = torch.randint(1, self.word_neighbor_dist+1, (sen_len, batch_size))
        choices += (torch.arange(sen_len*batch_size)*(self.word_neighbor_dist+1)).view(sen_len, batch_size)
        chosen_neighbors = torch.take(neighbors, choices.cuda())

        p_mask = torch.rand(sen_len, batch_size).cuda() < self.word_neighbor
        pad_mask = 1-src_tokens.eq(self.padding_idx)
        p_mask *= pad_mask # dont add noise to padding
        p_mask[0,:] = 0 # dont change language token
        
        out = src_tokens.masked_scatter(p_mask, chosen_neighbors)
        return out


    def forward(self, src_tokens, src_lengths, lang_id, noise=None):
        assert type(lang_id) is int

        embed_tokens = self.embeddings[lang_id]

        if noise is 0: # only add neighbor noise to reconstruction
            neighbored_tokens = self.neighbor_noise(src_tokens, lang_id)
            src_tokens = neighbored_tokens
        # compute padding mask
        encoder_padding_mask = src_tokens.t().eq(self.padding_idx)
        # embed tokens
        x = self.embed_scale * embed_tokens(src_tokens)
        x = x.detach() if self.freeze_enc_emb else x

        # add embedding noise
        if noise is 0: # only add embedding noise to reconstruction
            x = embedding_noise(x, encoder_padding_mask, self.embed_noise_alpha)

        # embed positions
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)



        # encoder layers
        for layer in self.layers:
            x = layer[lang_id](x, encoder_padding_mask)

        if self.variational:
            x = self.var_ff(x)
            x_mean, x_logvar = torch.split(x, int(x.size(2)/2), dim=2)
            # print(x_out)
            x_std = torch.exp(0.5*x_logvar)
            if noise is None: # random noise
                # noise = torch.randn_like(x_std)
                noise = torch.randn(x_std.size()[1:])
                #noise = torch.randn(1)*torch.ones(x_std.size()[1:])
                noise *= self.sample_dist
            elif noise is 0: # no noise (i.e. magnitude 0)
                # noise = torch.zeros_like(x_std)
                noise = torch.zeros(x_std.size()[1:])
            elif isinstance(noise, float): # noise with specified magnitude
                #noise_mag = noise/20.0
                noise_mag = noise
                noise = torch.randn(x_std.size()[1:])
                # noise = torch.randn(1)*torch.ones(x_std.size()[1:])*noise_mag
                cur_mag = torch.norm(noise, dim=1)
                noise = (noise.t() * (noise_mag / cur_mag)).t()

            noise = noise.type_as(x_std)
            # noise /= x_std.size()[0]
            x = noise.mul(x_std).add_(x_mean)

            return LatentState(
                input_len=src_lengths,
                dec_input={
                    'encoder_out': x,  # T x B x C
                    'encoder_padding_mask': encoder_padding_mask,  # B x T
                },
                dis_input=x,
                vae_vars={
                    'mean': x_mean,
                    'logvar': x_logvar,
                    'std': x_std,
                    'noise': noise
                }
            )#, x_mean, x_logvar, noise

            # return x, x_mean, x_logvar, noise

        return LatentState(
            input_len=src_lengths,
            dec_input={
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': encoder_padding_mask,  # B x T
            },
            dis_input=x,
            vae_vars={}
        )

    def forward_encode(self, src_tokens, src_lengths, lang_id):
        """ Forward pass without sampling (for VAE otf generation)"""
        assert type(lang_id) is int

        embed_tokens = self.embeddings[lang_id]

        # compute padding mask
        encoder_padding_mask = src_tokens.t().eq(self.padding_idx)
        # embed tokens
        x = self.embed_scale * embed_tokens(src_tokens)
        x = x.detach() if self.freeze_enc_emb else x

        # add embedding noise
        x = embedding_noise(x, encoder_padding_mask, self.embed_noise_alpha)

        # embed positions
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)


        # encoder layers
        for layer in self.layers:
            x = layer[lang_id](x, encoder_padding_mask)

        x = self.var_ff(x)
        x_mean, x_logvar = torch.split(x, int(x.size(2)/2), dim=2)
        # print(x_out)
        x_std = torch.exp(0.5*x_logvar)

        return LatentState(
            input_len=src_lengths,
            dec_input={
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': encoder_padding_mask,  # B x T
            },
            dis_input=x,
            vae_vars={
                'mean': x_mean,
                'logvar': x_logvar,
                'std': x_std,
                'noise': None
            }
        )


    def forward_sample(self, latent_state, noise=None):
        """ Forward sampling (for VAE otf generation)"""

        x_std = latent_state.vae_vars['std']
        x_logvar = latent_state.vae_vars['logvar']
        x_mean = latent_state.vae_vars['mean']

        if noise is None: # random noise
            noise = torch.randn(x_std.size()[1:])
            noise *= self.sample_dist
        elif noise is 0: # no noise (i.e. magnitude 0)
            noise = torch.zeros(x_std.size()[1:])
        elif isinstance(noise, float): # noise with specified magnitude
            noise_mag = noise
            noise = torch.randn(x_std.size()[1:])
            cur_mag = torch.norm(noise, dim=1)
            noise = (noise.t() * (noise_mag / cur_mag)).t()

        noise = noise.type_as(x_std)
        x = noise.mul(x_std).add_(x_mean)

        return LatentState(
            input_len=latent_state.input_len,
            dec_input={
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': latent_state.dec_input['encoder_padding_mask'],  # B x T
            },
            dis_input=x,
            vae_vars={
                'mean': x_mean,
                'logvar': x_logvar,
                'std': x_std,
                'noise': noise
            }
        )



    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    @staticmethod
    def expand_encoder_out_(encoder_out, beam_size):
        T, B, C = encoder_out['encoder_out'].size()
        assert encoder_out['encoder_padding_mask'].size() == (B, T)
        encoder_out['encoder_out'] = encoder_out['encoder_out'].repeat(1, 1, beam_size).view(T, -1, C)
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].repeat(1, beam_size).view(-1, T)


class TransformerDecoder(nn.Module):
    """Transformer decoder."""

    DEC_ATTR = ['n_langs', 'n_words', ('share_lang_emb', False), ('share_encdec_emb', False), ('share_decpro_emb', False), ('share_dec', False), 'dropout', 'eos_index', 'pad_index', 'bos_index']

    def __init__(self, args, encoder):
        super().__init__()
        self.dropout = args.dropout
        self.n_langs = args.n_langs
        self.n_words = args.n_words
        self.share_lang_emb = args.share_lang_emb
        self.share_encdec_emb = args.share_encdec_emb
        self.share_decpro_emb = args.share_decpro_emb
        self.share_output_emb = args.share_output_emb
        self.share_dec = args.share_dec
        self.freeze_dec_emb = args.freeze_dec_emb
        self.encoder_class = encoder.__class__
        self.beam_size = args.beam_size
        self.length_penalty = args.length_penalty

        # indexes
        self.eos_index = args.eos_index
        self.pad_index = args.pad_index
        self.bos_index = args.bos_index

        # words allowed for generation
        self.vocab_mask_neg = args.vocab_mask_neg if len(args.vocab) > 0 else None  # TODO: implement

        # embedding layers
        self.emb_dim = args.decoder_embed_dim
        if self.share_encdec_emb:
            logger.info("Sharing encoder and decoder input embeddings")
            embeddings = encoder.embeddings
        else:
            if self.share_lang_emb:
                logger.info("Sharing decoder input embeddings")
                layer_0 = Embedding(self.n_words[0], self.emb_dim, padding_idx=self.pad_index)
                embeddings = [layer_0 for _ in range(self.n_langs)]
            else:
                embeddings = [Embedding(n_words, self.emb_dim, padding_idx=self.pad_index) for n_words in self.n_words]
            embeddings = nn.ModuleList(embeddings)
        self.embeddings = embeddings
        self.embed_scale = math.sqrt(self.emb_dim)
        self.embed_positions = PositionalEmbedding(
            1024, self.emb_dim, self.pad_index,
            left_pad=args.left_pad_target,
        )

        self.layers = nn.ModuleList([None for i in range(args.decoder_layers)])
        for k in range(args.decoder_layers):
            # share bottom share_dec layers
            layer_is_shared = (k < args.share_dec)
            if layer_is_shared:
                logger.info("Sharing decoder transformer parameters for layer %i" % k)

            self.layers[k] = nn.ModuleList([
                # layer for first lang
                TransformerDecoderLayer(args)
            ])
            for i in range(1, self.n_langs):
                # layer for lang i
                if layer_is_shared:
                    # share layer from lang 0
                    self.layers[k].append(self.layers[k][0])
                else:
                    self.layers[k].append(TransformerDecoderLayer(args))

        # projection layers
        proj = [nn.Linear(self.emb_dim, n_words) for n_words in self.n_words]
        if self.share_decpro_emb:
            logger.info("Sharing input embeddings and projection matrix in the decoder")
            for i in range(self.n_langs):
                proj[i].weight = self.embeddings[i].weight
            if self.share_lang_emb:
                assert self.share_output_emb
                logger.info("Sharing decoder projection matrices")
                for i in range(1, self.n_langs):
                    proj[i].bias = proj[0].bias
        elif self.share_output_emb:
            assert self.share_lang_emb
            logger.info("Sharing decoder projection matrices")
            for i in range(1, self.n_langs):
                proj[i].weight = proj[0].weight
                proj[i].bias = proj[0].bias
        self.proj = nn.ModuleList(proj)

    def forward(self, encoded, y, lang_id, one_hot=False, incremental_state=None):
        assert not one_hot, 'one_hot=True has not been implemented for transformer'
        assert type(lang_id) is int

        prev_output_tokens = y  # T x B
        encoder_out = encoded.dec_input
        embed_tokens = self.embeddings[lang_id]
        proj_layer = self.proj[lang_id]

        # embed positions
        positions = self.embed_positions(prev_output_tokens, incremental_state)

        # embed tokens and positions
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[-1:, :]  # only keep last time step
        x = self.embed_scale * embed_tokens(prev_output_tokens)
        x = x.detach() if self.freeze_dec_emb else x
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for layer in self.layers:
            x, attn = layer[lang_id](
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                incremental_state=incremental_state,
            )

        # project back to size of vocabulary
        x = proj_layer(x)

        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def reorder_incremental_state_(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )
        self.apply(apply_reorder_incremental_state)

    def reorder_encoder_out_(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)

    def generate(self, encoded, lang_id, max_len=200, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        if self.beam_size > 0:
            return self.generate_beam(encoded, lang_id, self.beam_size, max_len, sample, temperature)

        encoder_out = encoded.dec_input
        latent = encoder_out['encoder_out']

        x_len = encoded.input_len
        is_cuda = latent.is_cuda
        one_hot = None

        # check inputs
        assert type(lang_id) is int
        assert latent.size() == (x_len.max(), x_len.size(0), self.emb_dim)
        assert (sample is True) ^ (temperature is None)

        # initialize generated sentences batch
        slen, bs = latent.size(0), latent.size(1)
        assert x_len.max() == slen and x_len.size(0) == bs
        cur_len = 1
        decoded = torch.LongTensor(max_len, bs).fill_(self.pad_index)
        unfinished_sents = torch.LongTensor(bs).fill_(1)
        lengths = torch.LongTensor(bs).fill_(1)
        if is_cuda:
            decoded = decoded.cuda()
            unfinished_sents = unfinished_sents.cuda()
            lengths = lengths.cuda()
        decoded[0] = self.bos_index[lang_id]

        incremental_state = {}
        while cur_len < max_len:

            # previous word embeddings
            scores = self.forward(encoded, decoded[:cur_len], lang_id, one_hot, incremental_state)
            scores = scores.data[-1, :, :]  # T x B x V -> B x V

            # select next words: sample or one-hot
            if sample:
                next_words = torch.multinomial((scores / temperature).exp(), 1).squeeze(1)
            else:
                next_words = torch.topk(scores, 1)[1].squeeze(1)
            assert next_words.size() == (bs,)
            decoded[cur_len] = next_words * unfinished_sents + self.pad_index * (1 - unfinished_sents)
            lengths.add_(unfinished_sents)
            unfinished_sents.mul_(next_words.ne(self.eos_index).long())
            cur_len += 1

            # stop when there is a </s> in each sentence
            if unfinished_sents.max() == 0:
                break

        if cur_len == max_len:
            decoded[max_len - 1].masked_fill_(unfinished_sents.byte(), self.eos_index)
        assert (decoded == self.eos_index).sum() == bs

        return decoded[:cur_len], lengths, one_hot

    def generate_beam(self, encoded, lang_id, beam_size=20, max_len=175, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        self.encoder_class.expand_encoder_out_(encoded.dec_input, beam_size)

        x_len = encoded.input_len
        is_cuda = encoded.dec_input['encoder_out'].is_cuda
        one_hot = None

        # check inputs
        assert type(lang_id) is int
        # assert latent.size() == (x_len.max(), x_len.size(0) * beam_size, self.emb_dim)
        assert (sample is True) ^ (temperature is None)
        assert temperature is None, 'not supported'

        generator = SequenceGenerator(
            self, self.bos_index[lang_id], self.pad_index, self.eos_index,
            self.n_words[lang_id], beam_size=beam_size, maxlen=max_len, sampling=sample,
            len_penalty=self.length_penalty,
        )
        if is_cuda:
            x_len = x_len.cuda()
        results = generator.generate(x_len, encoded, lang_id)

        lengths = torch.LongTensor([sent[0]['tokens'].numel() for sent in results])
        lengths.add_(1)  # for BOS
        max_len = lengths.max()
        bsz = len(results)
        decoded = results[0][0]['tokens'].new(max_len, bsz).fill_(0)
        decoded[0, :] = self.bos_index[lang_id]
        for i, sent in enumerate(results):
            ntoks = sent[0]['tokens'].numel()  # pick the top beam result
            decoded[1:ntoks + 1, i] = sent[0]['tokens']

        return decoded, lengths, one_hot


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x, key=x, value=x, mask_future_timesteps=True,
            incremental_state=incremental_state, need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x, attn = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state, static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, init_size=num_embeddings)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m
