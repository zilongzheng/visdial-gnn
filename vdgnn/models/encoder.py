import torch
import torch.nn as nn
import torch.nn.functional as F
import vdgnn.units as units

class GCNNEncoder(nn.Module):
    def __init__(self, model_args):
        super(GCNNEncoder, self).__init__()
        self.img_feat_size = model_args.img_feat_size
        self.embed_size = model_args.embed_size
        self.rnn_hidden_size = model_args.rnn_hidden_size
        self.num_layers = model_args.num_layers
        self.dropout = model_args.dropout
        self.message_size = model_args.message_size
        self.m_step = model_args.m_step
        self.e_step = model_args.e_step
        self.use_cuda = model_args.use_cuda

        # modules
        self.word_embed = nn.Embedding(model_args.vocab_size, model_args.embed_size, padding_idx=0)

        self.node_rnn = units.DynamicRNN(nn.LSTM(self.embed_size, self.rnn_hidden_size, self.num_layers,
                            batch_first=True, dropout=self.dropout))
        self.node_coattn = units.AlterCoAttn(num_hidden=512, img_feat_size=self.img_feat_size,
                            ques_feat_size=self.rnn_hidden_size, dropout=self.dropout)

        funsion_size = self.img_feat_size + self.rnn_hidden_size
        self.fusion = nn.Linear(funsion_size, self.message_size)

        self.node_feat_size = self.message_size

        self.link_fun = units.LinkFunction('graphconvattn', input_size=self.message_size, link_hidden_size=512, link_hidden_layers=2, link_relu=False)
        self.update_fun = units.UpdateFunction('gru', self.message_size, self.node_feat_size, update_hidden_layers=1, update_bias=False, update_dropout=0)

    def get_node_feat_vec(self, batch):
        img_feat = batch['img_feat']
        img_feat = img_feat.view(img_feat.size(0), -1, self.img_feat_size)

        ques = batch['ques']
        ques_len = batch['ques_len']
        hist = batch['hist']
        hist_len = batch['hist_len']

        round_id = batch['round']

        num_convs, num_rounds, max_hist_len = hist.size()
        hist = hist.contiguous().view(-1, num_rounds * max_hist_len)
        hist = self.word_embed(hist)
        hist = hist.view(num_convs, num_rounds, max_hist_len, -1)

        num_nodes = round_id + 2
        node_vec = torch.zeros(num_convs, num_nodes, self.message_size)

        if self.use_cuda:
            node_vec = node_vec.cuda()
        
        # encode history to nodes
        # cap, (q_0, a_0), ..., (q_{rnd-1}, a_{rnd-1})
        for rnd in range(num_rounds):
            h_i = hist[:, rnd, :, :]
            h_l = hist_len[:, rnd]

            hist_emb = self.node_rnn(h_i, h_l, initial_state=None)
            hist_attn, img_attn = self.node_coattn(hist_emb, img_feat)

            fused_vec = torch.cat((img_attn, hist_attn), dim=1)
            fused_emb = self.fusion(fused_vec)

            node_vec[:, rnd, :] = fused_emb

        # encode q_rnd
        q_emb = self.word_embed(ques)
        q_emb = self.node_rnn(q_emb, ques_len, initial_state=None)
        q_attn, img_attn = self.node_coattn(q_emb, img_feat)
        q_node = torch.cat((img_attn, q_attn), dim=1)
        q_node = self.fusion(q_node)

        node_vec[:, round_id+1, :] = q_node

        return node_vec

    def forward(self, batch, args):
        node_features = self.get_node_feat_vec(batch)

        batch_size, num_nodes, node_feat_size = node_features.size()
        # bn x node_feat_size x num_nodes
        node_features = node_features.permute(0, 2, 1)

        pred_node_feats = [node_features.clone() for _ in range(self.m_step+1)]

        for t in range(self.m_step):
            # M-step: compute edge weights
            pred_adj_mat = self.link_fun(pred_node_feats[t])
            softmax_pred_adj_mat = F.softmax(pred_adj_mat, dim=2)

            hidden_node_states = [pred_node_feats[t].clone() for _ in range(self.e_step+1)]
            # E-step: inner-loop for message passing
            for s in range(self.e_step):
                for z in range(num_nodes-1, num_nodes):
                    h_z = hidden_node_states[s][:, :, z]
                    h_v = hidden_node_states[s]

                    # Sum up messages from different nosdes according to weights
                    m_z = softmax_pred_adj_mat[:, z, :].unsqueeze(1).expand_as(h_v) * h_v
                    m_z = torch.sum(m_z, dim=2)

                    # h_z^s = U(h_z^(s-1), m_z^s)
                    # Add temporal dimension
                    h_z = self.update_fun(h_z.unsqueeze(0).contiguous(), m_z.unsqueeze(0))
                    hidden_node_states[s+1][:, :, z] = h_z

                    if s == self.e_step - 1:
                        pred_node_feats[t+1][:, :, z] = h_z.squeeze(0)

        return pred_adj_mat, pred_node_feats[self.m_step][:,:,num_nodes-1]
