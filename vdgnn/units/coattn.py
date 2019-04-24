import torch
import torch.nn as nn
import torch.nn.functional as F

class Attn(nn.Module):
    r"""
    Unit attention operation for alternating co-attention.
    ``https://arxiv.org/pdf/1606.00061.pdf``
    
    .. math::
        \begin{array}{ll}
        H = \tanh(W_x * X + (W_g * g)) \\
        a = softmax(w_{hx}^T * H}) \\
        output = sum a_i * x_i
        \end{array}

    Args:
        num_hidden: Number of output hidden size
        input_feat_size: Feature size of input image
        guidance_size:  Feature size of attention guidance [default: 0]
        dropout: Dropout rate of attention operation [default: 0.5]

    Inputs:
        - **X** (batch, input_seq_size, input_feat_size): Input image feature
        - **g** (batch, guidance_size): Attention guidance
    """
    def __init__(self, num_hidden, input_feat_size, guidance_size=0, dropout=0.5):
        super(Attn, self).__init__()

        self.num_hidden = num_hidden
        self.input_feat_size = input_feat_size
        self.guidance_size = guidance_size
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.W_x = nn.Linear(input_feat_size, num_hidden)
        if guidance_size > 0:
            self.W_g = nn.Linear(guidance_size, num_hidden)
        self.W_hx = nn.Linear(num_hidden, 1)

    def forward(self, X, g=None):
        batch_size, input_seq_size, input_feat_size = X.size()
        feat = self.W_x(X)

        if g is not None:
            g_emb = self.W_g(g).view(-1, 1, self.num_hidden)
            feat = feat + g_emb.expand_as(feat)

        hidden_feat = torch.tanh(feat)
        if self.dropout is not None:
            hidden_feat = self.dropout(hidden_feat)

        attn_weight = F.softmax(self.W_hx(hidden_feat), dim=1)
        attn_X = torch.bmm(attn_weight.view(-1, 1, input_seq_size),
                        X.view(-1, input_seq_size, input_feat_size))
        return attn_X.view(-1, input_feat_size)

class AlterCoAttn(nn.Module):
    r"""
    Alternatiing Co-Attention mechanism according to ``https://arxiv.org/pdf/1606.00061.pdf``

    .. math::
        \begin{array}{ll}
            \hat{s} = Attn(Q, 0) \\
            \hat{v} = Attn(\hat{s}, V) \\
            \hat{q} = Attn(Q, \hat{v})
        \end{array}
    """
    def __init__(self, num_hidden, img_feat_size, ques_feat_size, dropout=0.5):
        super(AlterCoAttn, self).__init__()
        self.num_hidden = num_hidden
        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size

        self.attn1 = Attn(num_hidden, ques_feat_size, guidance_size=0, dropout=dropout)
        self.attn2 = Attn(num_hidden, img_feat_size, ques_feat_size, dropout=dropout)
        self.attn3 = Attn(num_hidden, ques_feat_size, img_feat_size, dropout=dropout)


    def forward(self, ques_feat, img_feat):
        """
        : param ques_feat: [batch, ques_feat_size]
        : param img_feat: [batch, img_seq_size, ques_feat_size]
        """
        ques_self_attn =  self.attn1(ques_feat.unsqueeze(1), None)
        img_attn_feat = self.attn2(img_feat, ques_self_attn)
        ques_attn_feat = self.attn3(ques_feat.unsqueeze(1), img_attn_feat)

        return ques_attn_feat, img_attn_feat

class ParallelCoAttn(nn.Module):
    def __init__(self, num_hidden, img_feat_size, ques_feat_size, img_seq_size, ques_seq_size=1, dropout=0.5):
        super(ParallelCoAttn, self).__init__()
        self.num_hidden = num_hidden
        self.img_feat_size = img_feat_size
        self.ques_feat_size = ques_feat_size
        self.img_seq_size = img_seq_size
        self.ques_seq_size = ques_seq_size

        self.W_b = nn.Linear(img_feat_size, ques_feat_size, bias=False)
        self.W_v = nn.Linear(img_feat_size, num_hidden, bias=False)
        self.W_q = nn.Linear(ques_feat_size, num_hidden, bias=False)
        self.W_hq = nn.Linear(num_hidden, 1, bias=False)
        self.W_hv = nn.Linear(num_hidden, 1, bias=False)

        self.dropout_q = nn.Dropout(dropout) if dropout > 0  else None
        self.dropout_i = nn.Dropout(dropout) if dropout > 0 else None


    def forward(self, ques_feat, img_feat):
        img_corr = self.W_b(img_feat.view(-1, self.img_seq_size, self.img_feat_size)).view(-1, self.img_seq_size, self.ques_feat_size)

        affinity_matrix = torch.tanh(torch.bmm(ques_feat.view(-1, self.ques_seq_size, self.ques_feat_size), img_corr.transpose(1, 2)))

        ques_embed = self.W_q(ques_feat.view(-1, self.ques_seq_size, self.ques_feat_size)).view(-1, self.ques_seq_size, self.num_hidden)

        img_embed = self.W_v(img_feat.view(-1, self.img_seq_size, self.img_feat_size)).view(-1, self.img_seq_size, self.num_hidden)

        img_transformed = torch.bmm(affinity_matrix, img_embed)
        ques_attn_hidden = torch.tanh(img_transformed + ques_embed)
        if self.dropout_q is not None:
            ques_attn_hidden = self.dropout_q(ques_attn_hidden)
        ques_attn_weight = F.softmax(self.W_hq(ques_attn_hidden).view(-1, self.ques_seq_size), dim=1)
        ques_attn_feat = torch.bmm(ques_attn_weight.view(-1, 1, self.ques_seq_size), ques_feat.view(-1, self.ques_seq_size, self.ques_feat_size))
        ques_attn_feat = ques_attn_feat.view(-1, self.ques_feat_size)

        ques_transformed = torch.bmm(affinity_matrix.transpose(1, 2), ques_embed)
        img_attn_hidden = torch.tanh(ques_transformed + img_embed)
        if self.dropout_i is not None:
            img_attn_hidden = self.dropout_i(img_attn_hidden)
        img_attn_weight = F.softmax(self.W_hv(img_attn_hidden).view(-1, self.img_seq_size), dim=1)
        img_attn_feat = torch.bmm(img_attn_weight.view(-1, 1, self.img_seq_size), img_feat.view(-1, self.img_seq_size, self.img_feat_size))
        img_attn_feat = img_attn_feat.view(-1, self.img_feat_size)

        return ques_attn_feat, img_attn_feat
