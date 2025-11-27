import torch
from transformers import BertConfig,BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertModel,EarlyStoppingCallback
from transformers.models.bert.modeling_bert import BertLayer, BertSelfAttention, BertIntermediate, BertOutput
import torch
import torch.nn as nn
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score
import math
from typing import Optional, Tuple
import pytorch_spiking
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math
from transformers import Trainer, TrainingArguments
import torch.nn.functional as F
from spikingjelly.clock_driven.surrogate import ATan as atan
from spikingjelly.clock_driven import neuron, functional, surrogate
from spiking_all_in_one import *
from neuron import *
from config import TRAINING_CONFIG, SPIKING_CONFIG, FILE_PATHS, DATA_CONFIG, EVAL_CONFIG, DEVICE_CONFIG
from transformers import BertForSequenceClassification, BertConfig
output_dir = TRAINING_CONFIG['output_dir']
num_train_epochs = TRAINING_CONFIG['num_train_epochs']
per_device_train_batch_size = TRAINING_CONFIG['per_device_train_batch_size']
per_device_eval_batch_size = TRAINING_CONFIG['per_device_eval_batch_size']
warmup_steps = TRAINING_CONFIG['warmup_steps']
weight_decay = TRAINING_CONFIG['weight_decay']
logging_dir = TRAINING_CONFIG['logging_dir']
logging_steps = TRAINING_CONFIG['logging_steps']
metric_for_best_model = TRAINING_CONFIG['metric_for_best_model']

alpha_spiking_q = SPIKING_CONFIG['alpha_spiking_q']
alpha_origin_q = SPIKING_CONFIG['alpha_origin_q']

train_file = FILE_PATHS['train_file']
validation_file = FILE_PATHS['validation_file']
test_file = FILE_PATHS['test_file']
local_model_path = FILE_PATHS['local_model_path']
class DynamicHybridModulation(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder
        self.spiking_activation_q = DualThresholdSelfregulatingIntegrate(
            torch.nn.Tanh(), dt=1, spiking_aware_training=True,T=1
        )
        self.spiking_activation_k = DualThresholdSelfregulatingIntegrate(
            torch.nn.Tanh(), dt=1, spiking_aware_training=True,T=1
        )
        self.spiking_activation_v = DualThresholdSelfregulatingIntegrate(
            torch.nn.Tanh(), dt=1, spiking_aware_training=True,T=1
        )
        self.alpha_spiking = nn.Parameter(torch.tensor(alpha_spiking_q))
        self.alpha_orig = nn.Parameter(torch.tensor(alpha_origin_q))
        self.alpha_spiking_q = nn.Parameter(torch.tensor(0.52))
        self.alpha_orig_q = nn.Parameter(torch.tensor(1.07)) 
        self.lif1 = neuron.MultiStepLIFNode(tau=2., surrogate_function=surrogate.ATan(alpha=2.0), backend='torch',
                                            v_threshold=1.)
        self.IF = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(alpha=2.0), backend='torch',
                                            v_threshold=1.)
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    def coefficient_loss(self):
        loss = torch.abs(self.alpha_spiking + self.alpha_orig - 1)
        loss += (self.alpha_spiking ** 2 + self.alpha_orig ** 2)

        return loss
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.spiking_activation_q(self.query(hidden_states))
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.spiking_activation_k(self.key(encoder_hidden_states)) * self.alpha_spiking + self.alpha_orig*self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.spiking_activation_k(self.key(hidden_states)) * self.alpha_spiking + self.alpha_orig*self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.spiking_activation_k(self.key(hidden_states)) * self.alpha_spiking + self.alpha_orig*self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask + 0.0001*self.select_linesnn(attention_scores,auto_balance_snn=True)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    def linesnn_mcz(self, x, channel, reduction=4):
        device = x.device  
        _, _, h, w = x.size()
        x = x.flatten(start_dim=2)
        x = self.spiking_activation_v(x)
        x = self.transpose_for_line_mcz(x)
        x_h = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2)
        x_w = torch.mean(x, dim=2, keepdim=True)
        x_cat_conv_relu = nn.ReLU()(nn.BatchNorm2d(channel // reduction,device=device)(nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, bias=False,device=device)(torch.cat((x_h, x_w), 3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h, w], 3)
        s_h = nn.Sigmoid()(nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,bias=False,device=device)(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = nn.Sigmoid()(nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,bias=False,device=device)(x_cat_conv_split_w))
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
        return out
    def transpose_for_line_mcz(self, x):
        _, _, hw = x.size()
        h = int(math.sqrt(hw))
        new_x_shape = x.size()[:-1] + (h, h) 
        x = x.view(*new_x_shape)
        return x
    def linesnn5(self, x):
        device = x.device
        b, c, h, w = x.size()
        x = x.view(x.size(0), x.size(1), 1, -1)
        x = x.view(-1, 1, h * w)
        x = nn.Conv1d(
            in_channels=1,  
            out_channels=1,  
            kernel_size=w,  
            stride=w,  
            padding=0, 
            bias=False, 
            device=device
        )(x)
        x = nn.BatchNorm1d(num_features=1,device=device)(x)
        x = nn.ReLU()(x)
        x = x.view(b, c, 1, -1)
        return x
    def select_linesnn(self,x,auto_balance_snn=False):
        if auto_balance_snn:
            x = self.linesnn_mcz(x, self.num_attention_heads)
        else:
            x = self.linesnn5(x)
        return x
    def calculate_nonzero_ratio(self,matrix):
   
        total_elements = matrix.numel()  
        nonzero_elements = torch.count_nonzero(matrix)   
        ratio = nonzero_elements.float() / total_elements  
        return ratio.item()  

    def save_ratios_to_txt(self,matrices, filename):
        with open(filename, 'w') as f:
            for i, matrix in enumerate(matrices):
                ratio = self.calculate_nonzero_ratio(matrix)
                f.write(f"Matrix {i + 1}: Nonzero Ratio = {ratio:.4f}\n")  # 写入文件
class CustomBertModel(BertForSequenceClassification):
    def __init__(self, config):
        super(CustomBertModel, self).__init__(config)
        for i in range(config.num_hidden_layers):
            self.bert.encoder.layer[i].attention.self = DynamicHybridModulation(config)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )  
        logits = outputs.logits
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            total_coefficient_loss = 0.0
            for layer in self.bert.encoder.layer:
                total_coefficient_loss += layer.attention.self.coefficient_loss()
            total_loss = loss + 0.01 * total_coefficient_loss
            return (total_loss, logits) 
        return outputs
if __name__ == "__main__":
    print("The current module contains model definitions. Please run train.py to start the training process.")





