import sys

import torch
import torch.nn as nn
import torch.autograd


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell

    args:
        input_size: number of channels of input
        hidden_size: number of channels of hidden state
        kernal_size: size of convolutional kernal [default=1]
        bias: whether to use bias in convolution [default=True]
    """

    def __init__(self, input_size, hidden_size, kernel_size=1, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) / 2
        self.bias = bias
        self.Gates = nn.Conv2d(input_size + hidden_size, 
                               4 * hidden_size, 
                               self.kernel_size, 
                               padding=self.padding,
                               bias=self.bias)

    def forward(self, input_, prev_state, use_cuda=True):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        state_size = [batch_size, self.hidden_size] + list(spatial_size)
        if prev_state is None or prev_state[0].size() != input_.size():
            prev_state = self._init_prev_states(state_size, use_cuda)

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), dim=1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = torch.chunk(gates, 4, dim=1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return hidden, cell

    @staticmethod
    def _init_prev_states(state_size, use_cuda):
        if use_cuda:
            prev_state = (
                torch.zeros(state_size, requires_grad=True).cuda(),
                torch.zeros(state_size, requires_grad=True).cuda()
            )
        else:
            prev_state = (
                torch.zeros(state_size, requires_grad=True),
                torch.zeros(state_size, requires_grad=True)
            )
        return prev_state


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, hidden_layer_num, kernel_size=1, bias=True):
        super(ConvLSTM, self).__init__()
        if hidden_layer_num < 1:
            sys.exit("Hidden layer number less than 1.")

        self.hidden_layer_num = hidden_layer_num
        self.layers = torch.nn.ModuleList()
        self.prev_states = list()
        for i in range(hidden_layer_num):
            if i == 0:
                self.layers.append(ConvLSTMCell(input_channels, hidden_channels, kernel_size))
            else:        
                self.layers.append(ConvLSTMCell(hidden_channels, hidden_channels, kernel_size))
        self._init_hidden_states()

    def forward(self, input_, reset=False):
        """
        args:
            input_: size of (b, t, c, h, w)
            reset: whether to init hidden states
        """
        if reset:
            self._init_hidden_states()
        else:
            for prev_state in self.prev_states:
                if prev_state:
                    prev_state[0].detach_()
                    prev_state[1].detach_()

        next_layer_input = input_
        for i, layer in enumerate(self.layers):
            prev_state = layer(next_layer_input, self.prev_states[i])
            next_layer_input = prev_state[0]
            self.prev_states[i] = prev_state

        return next_layer_input

    def _init_hidden_states(self):
        self.prev_states = [None for _ in range(self.hidden_layer_num)]
