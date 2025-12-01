
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import copy
import json
import math
import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from config import hbert_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.neuron import DualThresholdSelfregulatingIntegrate, DualThresholdSelfregulatingIntegrateAction
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon        # 一个很小的常数，防止除0

    def forward(self, x):
        u = x.mean(-1, keepdim=True)                    # LN是对最后一个维度做Norm
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CustomLIFNeuron(nn.Module):
    def __init__(self, tau=2.0, v_threshold=1.0, dt=1.0):
        super(CustomLIFNeuron, self).__init__()
        self.tau = tau
        self.v_threshold = v_threshold
        self.dt = dt

    def forward(self, input_current, v_prev=None):
        if v_prev is None:
            v_prev = torch.zeros_like(input_current)
        dv = self.dt / self.tau * (-v_prev + input_current)
        v_curr = v_prev + dv
        spike = (v_curr >= self.v_threshold).float()
        # 重置膜电位，将spike的位置设为0
        v_curr = torch.where(spike == 1, torch.zeros_like(v_curr), v_curr)
        return spike, v_curr
import torch
import torch.nn as nn

class CustomLIFNeuron1(nn.Module):
    def __init__(self, tau=2.0, v_threshold=1.0, dt=1.0, temperature=1.0):
        super(CustomLIFNeuron1, self).__init__()
        self.tau = tau
        self.v_Threshold = v_threshold
        self.dt = dt
        self.temperature = temperature 
    
    def forward(self, input_current, v_prev=None):
        if v_prev is None:
            v_prev = torch.zeros_like(input_current)
        dv = self.dt / self.tau * (-v_prev + input_current)
        v_curr = v_prev + dv
        spike = torch.where(v_curr >= self.v_Threshold, 1.0, 0.0)
        v_curr = torch.where(spike == 1, torch.zeros_like(v_curr), v_curr)
        return spike, v_curr
    
    def backward_hook(self, grad_output):
        input_current = self.input_data
        v_prev = self.v_prev_data
        dv = self.dt / self.tau * (-v_prev + input_current)
        v_curr = v_prev + dv
        sigmoid_grad = torch.sigmoid((v_curr - self.v_Threshold) / self.temperature) * (1 - torch.sigmoid((v_curr - self.v_Threshold) / self.temperature))
        grad_input_current = grad_output * sigmoid_grad
        return grad_input_current,

class CustomIFNeuron1(nn.Module):
    def __init__(self, v_threshold=1.0, dt=0.1):
        super(CustomIFNeuron, self).__init__()
        self.v_threshold = v_threshold 
        self.dt = dt                   
        self.reset()                  

    def reset(self):
        self.membrane_potential = None 

    def forward(self, input_current):
        if self.membrane_potential is None:
            batch_size = input_current.size(0)
            self.membrane_potential = torch.zeros_like(input_current).to(input_current.device)
        self.membrane_potential += input_current * self.dt
        spike = (self.membrane_potential >= self.v_threshold).float()
        self.membrane_potential = torch.where(spike > 0, torch.zeros_like(self.membrane_potential), self.membrane_potential)
        return spike
    def reset_(self):
        if self.membrane_potential is not None:
            batch_size = self.membrane_potential.size(0)
            self.membrane_potential = torch.zeros_like(self.membrane_potential).to(self.membrane_potential.device)
class CustomIFNeuron(nn.Module):
    def __init__(self, v_threshold=1.0, dt=1.0):
        super(CustomIFNeuron, self).__init__()
        self.v_Threshold = v_threshold
        self.dt = dt
    def forward(self, input_current, v_prev=None):
        if v_prev is None:
            v_prev = torch.zeros_like(input_current)
        dv = self.dt * input_current
        v_curr = v_prev + dv
        v_curr = torch.where(spike == 1, torch.zeros_like(v_curr), v_curr)
        return spike, v_curr

class DynamicHybridModulation(nn.Module):
    def __init__(self, config):
        super(DynamicHybridModulation, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size) #config.hidden_size=512
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.spiking_activation_q = DualThresholdSelfregulatingIntegrateAction(
            torch.nn.Tanh(), dt=1, spiking_aware_training=True,T=1
        )
        self.spiking_activation_k = DualThresholdSelfregulatingIntegrateAction(
            torch.nn.Tanh(), dt=1, spiking_aware_training=True,T=1
        )
        self.spiking_activation_v = DualThresholdSelfregulatingIntegrateAction(
            torch.nn.Tanh(), dt=1, spiking_aware_training=True,T=1
        )
        self.alpha_spiking = hbert_config.alpha_spiking
        self.alpha_orig = hbert_config.alpha_orig  
        self.k = hbert_config.k
        self.iflayer = CustomLIFNeuron(v_threshold=1.0)
        self.IF = CustomIFNeuron()
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) 
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask): 
        mixed_query_layer = self.spiking_activation_q(self.query(hidden_states))
        mixed_key_layer = self.key(hidden_states)
        mixed_key_layer = self.alpha_spiking*self.spiking_activation_k(mixed_key_layer)+self.alpha_orig*mixed_key_layer
        mixed_value_layer = self.value(hidden_states).detach()  
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores /= math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask + self.k*self.select_linesnn(attention_scores,auto_balance_snn=True)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)     #防止过拟合
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
    def alpha(self):
        a = self.alpha_spiking
        b = self.alpha_orig
        return a,b
    def coefficient_loss(self):
        loss = torch.abs(self.alpha_spiking + self.alpha_orig - 1)
        loss += (self.alpha_spiking ** 2 + self.alpha_orig ** 2)
        return loss
    def linesnn_mcz(self, x, channel, reduction=4):
        device = x.device
        _, _, h, w = x.size()
        x = x.flatten(start_dim=2)
        x = self.spiking_activation_v(x)
        x = self.IF(x)[0]
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
                f.write(f"Matrix {i + 1}: Nonzero Ratio = {ratio:.4f}\n")  

class BERTSelfOutput(nn.Module):
    """BERTSelfAttention 之后还有一个 feed forward,dropout,add and norm """
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)   
        return hidden_states


class BERTAttention(nn.Module):
    """一个BERT block中的前面部分"""
    def __init__(self, config):
        super(BERTAttention, self).__init__()
        self.self = DynamicHybridModulation(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BERTIntermediate(nn.Module):
    """BERT模型中唯一用到了激活函数的地方, BERTIntermediate只是在中间扩充了一下维度，在BERTOutput中又转回去了"""
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BERTOutput(nn.Module):
    """一个BERT block中的后面部分，和前面的BERTSelfOutput几乎相同"""
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTLayer(nn.Module):
    """一个BERT block包括三个部分：BERTAttention， BERTIntermediate， BERTOutput"""
    def __init__(self, config):
        super(BERTLayer, self).__init__()
        self.attention = BERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BERTEncoder(nn.Module):
    """12 个BERT block， 中间一定要用copy.deepcopy，否则指代的会是同一个block"""
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        layer = BERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers   # 记录了第一层到最后一层，所有time_step的输出


class BERTPooler(nn.Module):
    """取的[CLS]位输出做分类"""
    def __init__(self, config):
        super(BERTPooler, self).__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class BertModel(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output


class BertForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class BertForQuestionAnswering(nn.Module):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__()
        self.bert = BertModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
            if isinstance(module, nn.Linear):
                module.bias.data.zero_()
        self.apply(init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, start_positions=None, end_positions=None):
        all_encoder_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = all_encoder_layers[-1]
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

if __name__=="__main__":
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])
    # from torchsummary import summary
    # from thop import profile
    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=8, intermediate_size=1024)

    model = BertModel(config=config)

  
