import math
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import copy
import pickle




class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()

        self.dense_1 = nn.Conv1d(hidden_size, inner_size, kernel_size=1)
        self.intermediate_act_fn = torch.nn.GELU()

        self.dense_2 = nn.Conv1d(inner_size, hidden_size, kernel_size=1)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(hidden_dropout_prob)
        self.dropout2 = torch.nn.Dropout(hidden_dropout_prob)

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor.transpose(-1, -2))
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = self.LayerNorm(hidden_states.transpose(-1, -2) + input_tensor)

        return hidden_states

class MyModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.opti = []

        self.args = args

        self.dircetor_emb = nn.Embedding(args.item_num + 1, int(args.latent_factor), padding_idx=0)

        # self.item_emb = nn.Embedding.from_pretrained(weight,freeze=False)
        self.item_emb = nn.Embedding(args.item_num + 1, int(args.latent_factor / 2), padding_idx=0)



        self.pos_emb1 = nn.Embedding(args.batch, args.latent_factor)
        self.pos_emb2 = nn.Embedding(args.batch, args.latent_factor)


        self.emb_dropout = nn.Dropout(p=args.dropout_rate)

        self.trm_encoder = TransformerEncoder(
            n_layers=args.num_blocks,
            n_heads=args.num_heads,
            hidden_size=args.latent_factor*4,
            inner_size=args.latent_factor*4,
            hidden_dropout_prob=args.dropout_rate,
            attn_dropout_prob=args.dropout_rate,
            layer_norm_eps=1e-8
        )

        self.layernorm = nn.LayerNorm(args.latent_factor*4, eps=1e-8)
        
    def forward(self, datas):
        p_out = []
        d_out = []
        position = np.tile(np.array(range(datas['p_rec'].shape[0])), [datas['p_rec'].shape[1], 1]).T
        dircetor_seqs = self.dircetor_emb(torch.LongTensor(datas['p_rec']))
        dircetor_seqs += self.pos_emb2(torch.LongTensor(position))
        path_emb = [dircetor_seqs]
        for path in self.args.meta_path:
            states = []
            for i in datas['p_rec'].T:
                nei = []
                for ii in i:

                    item = self.item_emb(torch.LongTensor(np.array([ii])))
                    if ii != 0:
                        if ii in datas['p_' + path + '_nei']:
                            nei_item = self.item_emb(torch.LongTensor(datas['p_' + path + '_nei'][ii]))
                        else:
                            nei_item = self.item_emb(torch.LongTensor(np.array([ii])))
                        item = torch.cat([item, nei_item.mean(0).unsqueeze(0)], 1)


                    else:
                        item = torch.cat([item, item], 1)

                    nei.append(item)

                states.append(torch.stack(nei))
            item_seqs = torch.cat(states,1)




            
            


            item_seqs += self.pos_emb1(torch.LongTensor(position))
            path_emb.append(item_seqs)

        seqs = torch.cat(path_emb,-1)
        seqs = self.layernorm(seqs)
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(datas['p_rec'] == 0)
        seqs *= ~timeline_mask.unsqueeze(-1)

        t1 = item_seqs.shape[1] 
        attention_mask1 = ~torch.tril(torch.ones((t1, t1), dtype=torch.bool))

        trm_output = self.trm_encoder(seqs, attention_mask1, output_all_encoded_layers=True)
        seqs = trm_output[-1]

        output1 = seqs[:, :, 100:400]
        output2 = seqs[:, :, 0:100]

        item_seq = []
        dire_seq = []

        for jj in datas['p_t']:
            item = output1[jj[0],jj[1],:]
            item_seq.append(item)
            director = output2[jj[0], jj[1], :]
            dire_seq.append(director)

        p_out = torch.stack(item_seq)
        d_out = torch.stack(dire_seq)

        context = p_out

        return context,torch.cat([d_out,d_out,d_out],1)






class QNet(nn.Module):
    def __init__(self,args):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(args.latent_factor*3, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, args.item_num+1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self,context):
        x = self.fc1(context)
        x = F.leaky_relu(x)
        actions_value = self.out(x).squeeze()
        return actions_value

class RecModel(nn.Module):
    def __init__(self, args):
        super(RecModel, self).__init__()
        self.args = args
        self.Model = MyModel(self.args)
        self.QNet = QNet(self.args)
    def forward(self, datas, nce):
        output1,output2 = self.Model(datas)
        if(nce == 1):
            return output1,output2
        else:
            action = self.QNet(output1)
        return action


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output



class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class BiTransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.

        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(BiTransformerEncoder, self).__init__()
        layer = BiTransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states1, attention_mask1, hidden_states2, attention_mask2, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers1 = []
        all_encoder_layers2 = []
        for layer_module in self.layer:
            hidden_states1, hidden_states2 = layer_module(hidden_states1, attention_mask1, hidden_states2, attention_mask2)
            if output_all_encoded_layers:
                all_encoder_layers1.append(hidden_states1)
                all_encoder_layers2.append(hidden_states2)
        if not output_all_encoded_layers:
            all_encoder_layers1.append(hidden_states1)
            all_encoder_layers2.append(hidden_states2)
        return all_encoder_layers1, all_encoder_layers2

class BiTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(BiTransformerLayer, self).__init__()
        self.bi_attention = BertBiAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.bi_feed_forward = BiFeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states1, attention_mask1, hidden_states2, attention_mask2):
        attention_output1, attention_output2 = self.bi_attention(hidden_states1, attention_mask1, hidden_states2, attention_mask2)
        feedforward_output1, feedforward_output2 = self.bi_feed_forward(attention_output1, attention_output2)
        return feedforward_output1, feedforward_output2

class BertBiAttention(nn.Module):
    def __init__(self, bi_num_attention_heads, bi_hidden_size, hidden_dropout_prob, attention_probs_dropout_prob, layer_norm_eps):
        super(BertBiAttention, self).__init__()
        if bi_hidden_size % bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (bi_hidden_size, bi_num_attention_heads)
            )

        self.num_attention_heads = bi_num_attention_heads
        self.attention_head_size = int(
            bi_hidden_size / bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(bi_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(bi_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(bi_hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(attention_probs_dropout_prob)

        self.query2 = nn.Linear(bi_hidden_size, self.all_head_size)
        self.key2 = nn.Linear(bi_hidden_size, self.all_head_size)
        self.value2 = nn.Linear(bi_hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(attention_probs_dropout_prob)

        self.dense1 = nn.Linear(bi_hidden_size, bi_hidden_size)
        self.LayerNorm1 = nn.LayerNorm(bi_hidden_size, eps=layer_norm_eps)
        self.out_dropout1 = nn.Dropout(hidden_dropout_prob)

        self.dense2 = nn.Linear(bi_hidden_size, bi_hidden_size)
        self.LayerNorm2 = nn.LayerNorm(bi_hidden_size, eps=layer_norm_eps)
        self.out_dropout2 = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor1, attention_mask1, input_tensor2, attention_mask2, co_attention_mask=None, use_co_attention_mask=False):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1

        if use_co_attention_mask:
            attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        if use_co_attention_mask:
            attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        hidden_states1 = self.dense1(context_layer1)
        hidden_states1 = self.out_dropout1(hidden_states1)
        hidden_states1 = self.LayerNorm1(hidden_states1 + input_tensor1)

        hidden_states2 = self.dense2(context_layer2)
        hidden_states2 = self.out_dropout2(hidden_states2)
        hidden_states2 = self.LayerNorm2(hidden_states2 + input_tensor2)

        return hidden_states1, hidden_states2

class BiFeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(BiFeedForward, self).__init__()
        self.i_dense_1 = nn.Linear(hidden_size, inner_size)
        self.i_intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.i_dense_2 = nn.Linear(inner_size, hidden_size)
        self.i_LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.i_dropout = nn.Dropout(hidden_dropout_prob)

        self.s_dense_1 = nn.Linear(hidden_size, inner_size)
        self.s_intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.s_dense_2 = nn.Linear(inner_size, hidden_size)
        self.s_LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.s_dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": F.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor1, input_tensor2):
        i_hidden_states = self.i_dense_1(input_tensor1)
        i_hidden_states = self.i_intermediate_act_fn(i_hidden_states)

        i_hidden_states = self.i_dense_2(i_hidden_states)
        i_hidden_states = self.i_dropout(i_hidden_states)
        i_hidden_states = self.i_LayerNorm(i_hidden_states + input_tensor1)

        s_hidden_states = self.s_dense_1(input_tensor2)
        s_hidden_states = self.s_intermediate_act_fn(s_hidden_states)

        s_hidden_states = self.s_dense_2(s_hidden_states)
        s_hidden_states = self.s_dropout(s_hidden_states)
        s_hidden_states = self.s_LayerNorm(s_hidden_states + input_tensor2)

        return i_hidden_states, s_hidden_states