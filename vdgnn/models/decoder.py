
import torch
import torch.nn as nn
import torch.nn.functional as F

from vdgnn.units import DynamicRNN


class DiscriminativeDecoder(nn.Module):
    def __init__(self, args, encoder):
        super(DiscriminativeDecoder, self).__init__()
        self.args = args
        # share word embedding
        self.word_embed = encoder.word_embed
        self.embed_size = args.embed_size
        self.rnn_hidden_size = args.rnn_hidden_size
        self.num_layers = args.num_layers
        # share embedding lstm
        self.option_rnn = encoder.node_rnn
        self.log_softmax = nn.LogSoftmax(dim=1)

    def init_weights(self, init_type='kaiming'):
        self.similarity_score.init_weights(init_type=init_type)

    def forward(self, enc_out, batch):
        options = batch['opt']
        options_len = batch['opt_len']
        # word embed options
        batch_size, num_rounds, num_options, max_opt_len = options.size()
        # options = options.view(batch_size * num_rounds, num_options, max_opt_len)
        options_len = options_len.view(-1, num_options)
        # batch_size, num_options, max_opt_len = options.size()
        options = options.contiguous().view(-1, num_options * max_opt_len)
        options = self.word_embed(options)
        options = options.view(-1, num_options, max_opt_len, self.embed_size)

        # score each option
        scores = []
        for opt_id in range(num_options):
            opt = options[:, opt_id, :, :]
            opt_len = options_len[:, opt_id]
            opt_embed = self.option_rnn(opt, opt_len)
            scores.append(torch.sum(opt_embed * enc_out, 1))

        # return scores
        scores = torch.stack(scores, 1)
        # print(scores.size())
        log_probs = self.log_softmax(scores)
        return log_probs
