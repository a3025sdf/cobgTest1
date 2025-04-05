# from IPython.core.debugger import set_trace
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math

class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim = 0, center = True, scale = True, epsilon = None, conditional = False,
                 hidden_units = None, hidden_activation = 'linear', hidden_initializer = 'xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features = self.cond_dim, out_features = self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features = self.cond_dim, out_features = input_dim, bias=False)

        self.initialize_weights()


    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            # 下面这两个为什么都初始化为0呢? 
            # 为了防止扰乱原来的预训练权重，两个变换矩阵可以全零初始化（单层神经网络可以用全零初始化，连续的多层神经网络才不应当用全零初始化），这样在初始状态，模型依然保持跟原来的预训练模型一致。
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)


    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            # for _ in range(K.ndim(inputs) - K.ndim(cond)): # K.ndim: 以整数形式返回张量中的轴数。
            # TODO: 这两个为什么有轴数差呢？ 为什么在 dim=1 上增加维度??
            # 为了保持维度一致，cond可以是（batch_size, cond_dim）
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)
            
            # cond在加入beta和gamma之前做一次线性变换，以保证与input维度一致
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs**2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) **2
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs
    
class HandshakingKernel(nn.Module):
    def __init__(self, hidden_size, shaking_type, inner_enc_type,only_look_after=True):
        super().__init__()
        self.shaking_type = shaking_type
        self.only_look_after = only_look_after

        if shaking_type == "cat":
            self.combine_fc = nn.Linear(hidden_size * 2, hidden_size)
        elif shaking_type == "cat_plus":
            self.combine_fc = nn.Linear(hidden_size * 3, hidden_size)
        elif shaking_type == "cln":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
        elif shaking_type == "cln_plus":
            self.tp_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            self.inner_context_cln = LayerNorm(hidden_size, hidden_size, conditional = True)
            
        self.inner_enc_type = inner_enc_type
        if inner_enc_type == "mix_pooling":
            self.lamtha = Parameter(torch.rand(hidden_size))
        elif inner_enc_type == "lstm":
            # self.inner_context_lstm = nn.LSTM(hidden_size,
            #                hidden_size,
            #                num_layers = 1,
            #                bidirectional = False,
            #                batch_first = True)
            assert only_look_after is True
            self.lstm4span = nn.LSTM(hidden_size,
                                    hidden_size,
                                    num_layers=1,
                                    bidirectional=False,
                                    batch_first=True)

    def enc_inner_hiddens(self, seq_hiddens, inner_enc_type = "lstm"):
        # seq_hiddens: (batch_size, seq_len, hidden_size)
        def pool(seqence, pooling_type):
            if pooling_type == "mean_pooling":
                pooling = torch.mean(seqence, dim = -2)
            elif pooling_type == "max_pooling":
                pooling, _ = torch.max(seqence, dim = -2)
            elif pooling_type == "mix_pooling":
                pooling = self.lamtha * torch.mean(seqence, dim = -2) + (1 - self.lamtha) * torch.max(seqence, dim = -2)[0]
            return pooling
        if "pooling" in inner_enc_type:
            inner_context = torch.stack([pool(seq_hiddens[:, :i+1, :], inner_enc_type) for i in range(seq_hiddens.size()[1])], dim = 1)
        elif inner_enc_type == "lstm":
            inner_context, _ = self.inner_context_lstm(seq_hiddens)
            
        return inner_context
    
    def forward(self, seq_hiddens):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        # seq_len = seq_hiddens.size()[-2]
        # shaking_hiddens_list = []
        # for ind in range(seq_len):
        #     hidden_each_step = seq_hiddens[:, ind, :]
        #     visible_hiddens = seq_hiddens[:, ind:, :] # ind: only look back
        #     repeat_hiddens = hidden_each_step[:, None, :].repeat(1, seq_len - ind, 1)
        #
        #     if self.shaking_type == "cat":
        #         shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens], dim = -1)
        #         shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
        #     elif self.shaking_type == "cat_plus":
        #         inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
        #         shaking_hiddens = torch.cat([repeat_hiddens, visible_hiddens, inner_context], dim = -1)
        #         shaking_hiddens = torch.tanh(self.combine_fc(shaking_hiddens))
        #     elif self.shaking_type == "cln":
        #         shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
        #     elif self.shaking_type == "cln_plus":
        #         inner_context = self.enc_inner_hiddens(visible_hiddens, self.inner_enc_type)
        #         shaking_hiddens = self.tp_cln(visible_hiddens, repeat_hiddens)
        #         shaking_hiddens = self.inner_context_cln(shaking_hiddens, inner_context)
        #
        #     shaking_hiddens_list.append(shaking_hiddens)
        # long_shaking_hiddens = torch.cat(shaking_hiddens_list, dim = 1)
        # return long_shaking_hiddens

        seq_len = seq_hiddens.size()[1]

        guide = seq_hiddens[:, :, None, :].repeat(1, 1, seq_len, 1)
        visible = guide.permute(0, 2, 1, 3)


        shaking_pre = None

        # pre_num = 0

        def add_presentation(all_prst, prst):
            if all_prst is None:
                all_prst = prst
            else:
                all_prst += prst
            return all_prst

        if self.only_look_after:
            if "lstm" in self.shaking_type:
                batch_size, _, matrix_size, vis_hidden_size = visible.size()
                # mask lower triangle
                upper_visible = visible.permute(0, 3, 1, 2).triu().permute(0, 2, 3, 1).contiguous()

                # visible4lstm: (batch_size * matrix_size, matrix_size, hidden_size)
                visible4lstm = upper_visible.view(-1, matrix_size, vis_hidden_size)
                span_pre, _ = self.lstm4span(visible4lstm)
                span_pre = span_pre.view(batch_size, matrix_size, matrix_size, vis_hidden_size)

                # drop lower triangle and convert matrix to sequence
                # span_pre: (batch_size, shaking_seq_len, hidden_size)
                span_pre = MyMatrix.upper_reg2seq(span_pre)
                shaking_pre = add_presentation(shaking_pre, span_pre)


            # guide, visible: (batch_size, shaking_seq_len, hidden_size)
            guide = MyMatrix.upper_reg2seq(guide)
            visible = MyMatrix.upper_reg2seq(visible)

        if "cat" in self.shaking_type:
            tp_cat_pre = torch.cat([guide, visible], dim=-1)
            tp_cat_pre = torch.relu(self.combine_fc(tp_cat_pre))
            shaking_pre = add_presentation(shaking_pre, tp_cat_pre)

        if "cln" in self.shaking_type:
            tp_cln_pre = self.tp_cln(visible, guide)
            shaking_pre = add_presentation(shaking_pre, tp_cln_pre)

        return shaking_pre

class MyMatrix:
    @staticmethod
    def upper_reg2seq(ori_tensor):
        '''
        drop lower region and flat upper region to sequence
        :param ori_tensor: (batch_size, matrix_size, matrix_size, hidden_size)
        :return: (batch_size, matrix_size + ... + 1, hidden_size)
        '''
        tensor = ori_tensor.permute(0, 3, 1, 2).contiguous()
        uppder_ones = torch.ones([tensor.size()[-1], tensor.size()[-1]]).long().triu().to(ori_tensor.device)
        upper_diag_ids = torch.nonzero(uppder_ones.view(-1), as_tuple=False).view(-1)
        # flat_tensor: (batch_size, matrix_size * matrix_size, hidden_size)
        flat_tensor = tensor.view(tensor.size()[0], tensor.size()[1], -1).permute(0, 2, 1)
        tensor_upper = torch.index_select(flat_tensor, dim=1, index=upper_diag_ids)
        return tensor_upper