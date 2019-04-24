import torch

class UpdateFunction(torch.nn.Module):
    def __init__(self, update_def, message_size, node_feature_size, update_hidden_layers=1, update_bias=False, update_dropout=False):
        super(UpdateFunction, self).__init__()
        self.u_def = ''
        self.u_function = None
        self.message_size = message_size
        self.node_feature_size = node_feature_size
        self.update_hidden_layers = update_hidden_layers
        self.update_bias = update_bias
        self.update_dropout = update_dropout
        self.learn_args = torch.nn.ParameterList([])
        self.learn_modules = torch.nn.ModuleList([])
        self.__set_update(update_def)

    def forward(self, h_v, m_v, args=None):
        return self.u_function(h_v, m_v, args)

    # Set an update function
    def __set_update(self, update_def):
        self.u_def = update_def.lower()
        if self.u_def == 'gru':
            self.init_gru()
            self.u_function = self.u_gru
        else:
            raise NotImplementedError('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_def
    
    # Definition of update functions
    # GRU: node state as hidden state, message as input
    def u_gru(self, h_v, m_v, args):
        output, h = self.learn_modules[0](m_v, h_v)
        return h

    def init_gru(self):
        self.learn_modules.append(
            torch.nn.GRU(self.message_size, self.node_feature_size, 
                    num_layers=self.update_hidden_layers, 
                    bias=self.update_bias, dropout=self.update_dropout))
