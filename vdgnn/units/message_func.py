import torch
import torch.nn as nn
import torch.autograd


class MessageFunction(torch.nn.Module):
    def __init__(self, message_def, message_size, node_feature_size, edge_feature_size, use_cuda=True):
        super(MessageFunction, self).__init__()
        self.msg_def = ''
        self.msg_func = None
        self.message_size = message_size
        self.node_feature_size = node_feature_size
        self.edge_feature_size = edge_feature_size
        self.use_cuda = use_cuda

        self.learn_args = torch.nn.ParameterList([])
        self.learn_modules = torch.nn.ModuleList([])
        self.__set_message(message_def)

    # Message from h_v to h_w through e_vw
    def forward(self, h_v, h_w, e_vw, args=None):
        return self.msg_func(h_v, h_w, e_vw, args)

    # Set a message function
    def __set_message(self, message_def):
        self.msg_def = message_def.lower()

        if self.msg_def == 'linear':
            self.init_linear()
            self.msg_func = self.m_linear
        elif self.msg_def == 'linear_edge':
            self.init_linear_edge()
            self.msg_func = self.m_linear_edge
        elif self.msg_def == 'linear_concat':
            self.init_linear_concat()
            self.msg_func = self.m_linear_concat
        elif self.msg_def == 'linear_concat_relu':
            self.init_linear_concat_relu()
            self.msg_func = self.m_linear_concat_relu
        else:
            raise NotImplementedError('WARNING!: Message Function has not been set correctly\n\tIncorrect definition ' + message_def)

    # Get the name of the used message function
    def get_definition(self):
        return self.msg_def
    
    def init_weights(self, init_type='kaiming'):
        if init_type == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif init_type == 'xavier':
            init_func == nn.init.xavier_normal_

        for m in self.learn_modules:
            init_func(m.weight.data)

    # Definition of message functions
    # Combination of linear transformation of edge features and node features
    def m_linear(self, h_v, h_w, e_vw, args):
        b_sz, e_sz, num_nodes = e_vw.size() 

        message = torch.zeros(b_sz, self.message_size, num_nodes, requires_grad=True)
        if self.use_cuda:
            message = message.cuda()

        for i_node in range(num_nodes):
            message[:, :, i_node] = self.learn_modules[0](e_vw[:, :, i_node]) + self.learn_modules[1](h_w[:, :, i_node])
        return message

    def init_linear(self):
        edge_feature_size = self.edge_feature_size
        node_feature_size = self.node_feature_size
        message_size = self.message_size
        self.learn_modules.append(nn.Linear(edge_feature_size, message_size, bias=True))
        self.learn_modules.append(nn.Linear(node_feature_size, message_size, bias=True))

    # Linear transformation of edge features
    def m_linear_edge(self, h_v, h_w, e_vw, args):
        message = torch.zeros(e_vw.size()[0], self.message_size, e_vw.size()[2], requires_grad=True)
        if self.use_cuda:
            message = message.cuda()

        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = self.learn_modules[0](e_vw[:, :, i_node])
        return message

    def init_linear_edge(self):
        edge_feature_size = self.edge_feature_size
        message_size = self.message_size
        self.learn_modules.append(nn.Linear(edge_feature_size, message_size, bias=True))

    # Concatenation of linear transformation of edge features and node features
    def m_linear_concat(self, h_v, h_w, e_vw, args):
        message = torch.zeros(e_vw.size()[0], self.message_size, e_vw.size()[2], requires_grad=True)
        if self.use_cuda:
            message = message.cuda()

        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = torch.cat([self.learn_modules[0](e_vw[:, :, i_node]), self.learn_modules[1](h_w[:, :, i_node])], 1)
        return message

    def init_linear_concat(self):
        edge_feature_size = self.edge_feature_size
        node_feature_size = self.node_feature_size
        message_size = self.message_size/2
        self.learn_modules.append(nn.Linear(edge_feature_size, message_size, bias=True))
        self.learn_modules.append(nn.Linear(node_feature_size, message_size, bias=True))

    # Concatenation of linear transformation of edge features and node features with ReLU
    def m_linear_concat_relu(self, h_v, h_w, e_vw, args):
        message = torch.zeros(e_vw.size()[0], self.message_size, e_vw.size()[2], requires_grad=True)
        if self.use_cuda:
            message = message.cuda()

        for i_node in range(e_vw.size()[2]):
            message[:, :, i_node] = torch.cat([self.learn_modules[0](e_vw[:, :, i_node]), 
                                               self.learn_modules[1](h_w[:, :, i_node])], 1)
        return message

    def init_linear_concat_relu(self):
        edge_feature_size = self.edge_feature_size
        node_feature_size = self.node_feature_size
        message_size = self.message_size/2
        self.learn_modules.append(nn.Linear(edge_feature_size, message_size, bias=True))
        self.learn_modules.append(nn.Linear(node_feature_size, message_size, bias=True))
