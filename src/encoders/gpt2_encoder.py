from typing import Dict, Any

import tensorflow as tf

from .utils.bert_self_attention import BertConfig, BertModel
from .masked_seq_encoder import MaskedSeqEncoder
from utils.tfutils import pool_sequence_embedding
from transformers import GPT2Model, GPT2Config
import torch

class Gpt2Encoder(MaskedSeqEncoder):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        encoder_hypers = {                 
        'input_ids':'None',
        'past_key_values':'None',
        'attention_mask':'None',
        'token_type_ids':'None',
        'position_ids':'None',
        'head_mask':'None',
        'inputs_embeds':'None',
        'encoder_hidden_states':'None',
        'encoder_attention_mask':'None',
        'use_cache':'None',
        'output_attentions':'None',
        'output_hidden_states':'None',
        'return_dict':'None',
        'vocab_size':'50257',
        'n_positions':'1024',
        'n_ctx':'1024',
        'n_embd':'768',
        'n_layer':'12',
        'n_head':'12',
        'n_inner':'None',
        'activation_function':'gelu_new',
        'resid_pdrop':'0.1',
        'embd_pdrop':'0.1',
        'attn_pdrop':'0.1',
        'layer_norm_epsilon':'1e-5',
        'initializer_range':'0.02',
        'summary_type':'cls_index',
        'summary_use_proj':'True',
        'summary_activation':'None',
        'summary_proj_to_labels':'True',
        'summary_first_dropout':'0.1',
        'bos_token_id':'50256',
        'eos_token_id':'50256',
        }
        hypers = super().get_default_hyperparameters()
        hypers.update(encoder_hypers)
        return hypers

    def __init__(self, label: str, hyperparameters: Dict[str, Any], metadata: Dict[str, Any]):
        super().__init__(label, hyperparameters, metadata)

    @property
    def output_representation_size(self):
        return self.get_hyper('self_attention_hidden_size')

    def make_model(self, is_train: bool = False) -> tf.Tensor:
        with tf.variable_scope("self_attention_encoder"):
            self._make_placeholders()

            config = GPT2Config(self.vocab_size = self.get_hyper('vocab_size'),
                                self.n_ctx = self.get_hyper('n_ctx'),
                                self.n_positions = self.get_hyper('n_positions'),
                                self.n_embd = self.get_hyper('n_embd'),
                                self.n_layer = self.get_hyper('n_layer'),
                                self.n_head = self.get_hyper('n_head'),
                                self.n_inner = self.get_hyper('n_inner'),
                                self.activation_function = self.get_hyper('activation_function'),
                                self.resid_pdrop = self.get_hyper('resid_pdrop'),
                                self.embd_pdrop = self.get_hyper('embd_pdrop'),
                                self.attn_pdrop = self.get_hyper('attn_pdrop'),
                                self.layer_norm_epsilon = self.get_hyper('layer_norm_epsilon'),
                                self.initializer_range = self.get_hyper('initializer_range'),
                                self.summary_type = self.get_hyper('summary_type'),
                                self.summary_use_proj = self.get_hyper('summary_use_proj'),
                                self.summary_activation = self.get_hyper('summary_activation'),
                                self.summary_first_dropout = self.get_hyper('summary_first_dropout'),
                                self.summary_proj_to_labels = self.get_hyper('summary_proj_to_labels'),
                                self.bos_token_id = self.get_hyper('bos_token_id'),
                                self.eos_token_id = self.get_hyper('eos_token_id'))

            model = GPT2Config(config=config)
            configuration = model.config             

            output_pool_mode = self.get_hyper('self_attention_pool_mode').lower()
            if output_pool_mode == 'bert':
                return model.get_pooled_output()
            else:
                seq_token_embeddings = model.get_sequence_output()
                seq_token_masks = self.placeholders['tokens_mask']
                seq_token_lengths = tf.reduce_sum(seq_token_masks, axis=1)  # B
                return pool_sequence_embedding(output_pool_mode,
                                               sequence_token_embeddings=seq_token_embeddings,
                                               sequence_lengths=seq_token_lengths,
                                               sequence_token_masks=seq_token_masks)
