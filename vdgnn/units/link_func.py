import torch
import torch.nn
from vdgnn.units.conv_lstm import ConvLSTM


class LinkFunction(torch.nn.Module):
    def __init__(self, link_def, input_size, link_hidden_size=512, link_hidden_layers=1, link_relu=False):
        super(LinkFunction, self).__init__()

        self.link_func = None
        self.learn_args = torch.nn.ParameterList([])
        self.learn_modules = torch.nn.ModuleList([])
        self.input_size = input_size
        self.link_hidden_size = link_hidden_size
        self.link_layer_num = link_hidden_layers
        self.link_relu = link_relu
        self.link_def = link_def.lower()

        if self.link_def == 'graphconv':
            self.init_graph_conv()
            self.link_func = self.l_graph_conv
        elif self.link_def == 'graphconvattn':
            self.init_graph_conv_atten()
            self.link_func = self.l_graph_conv_atten
        elif self.link_def == 'graphconvlstm':
            self.init_graph_conv_lstm()
            self.link_func = self.l_graph_conv_lstm
        else:
            raise NotImplementedError('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + link_def)

    def forward(self, edge_features):

        return self.link_func(edge_features)

    def get_definition(self):
        return self.link_def

    def l_graph_conv_atten(self, node_features):
        batch_size, node_feat_size, num_nodes  = node_features.size()
        last_layer_output = node_features.unsqueeze(-1).expand(-1, -1, -1, num_nodes)
        attn_feats = last_layer_output.permute(0, 1, 3, 2)
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        output = torch.sum(attn_feats * last_layer_output, dim=1)
        # for i in range(output.size()[0]):
        #     output[i, :, :] = torch.inverse(output[i, :, :])
        return output

    # Definition of linking functions
    def init_graph_conv_atten(self):
        for i in range(self.link_layer_num - 1):
            in_sz = self.input_size if i == 0 else self.link_hidden_size
            self.learn_modules.append(torch.nn.Conv2d(in_sz, self.link_hidden_size, 1))
            self.learn_modules.append(torch.nn.ReLU())
            # self.learn_modules.append(torch.nn.Dropout())
            # self.learn_modules.append(torch.nn.BatchNorm2d(hidden_size))

        in_sz = self.link_hidden_size if self.link_layer_num > 1 else self.input_size
        self.learn_modules.append(torch.nn.Conv2d(in_sz, self.input_size, 1))

    # GraphConv
    def l_graph_conv(self, edge_features):
        
        # print(edge_features.size())
        last_layer_output = edge_features
        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output[:, 0, :, :]

    def init_graph_conv(self):
        if not self.link_relu:
            self.learn_modules.append(torch.nn.ReLU())
            self.learn_modules.append(torch.nn.Dropout())
        for i in range(self.link_layer_num - 1):
            in_sz = self.input_size if i == 0 else self.link_hidden_size
            self.learn_modules.append(torch.nn.Conv2d(in_sz, self.link_hidden_size, 1))
            self.learn_modules.append(torch.nn.ReLU())
            # self.learn_modules.append(torch.nn.Dropout())
            # self.learn_modules.append(torch.nn.BatchNorm2d(hidden_size))

        in_sz = self.link_hidden_size if self.link_layer_num > 1 else self.input_size
        self.learn_modules.append(torch.nn.Conv2d(in_sz, 1, 1))
        # self.learn_modules.append(torch.nn.Sigmoid())

    # GraphConvLSTM
    def l_graph_conv_lstm(self, edge_features):
        last_layer_output = self.ConvLSTM(edge_features)

        for layer in self.learn_modules:
            last_layer_output = layer(last_layer_output)
        return last_layer_output[:, 0, :, :]

    def init_graph_conv_lstm(self):
        self.ConvLSTM = ConvLSTM(self.input_size, self.link_hidden_size, self.link_layer_num)
        self.learn_modules.append(torch.nn.Conv2d(self.link_hidden_size, 1, 1))
        self.learn_modules.append(torch.nn.Sigmoid())
