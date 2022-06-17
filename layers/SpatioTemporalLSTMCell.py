# modified so that this behaves like transpose-conv2dLSTM
__author__ = 'yunbo'

import torch
import torch.nn as nn

class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, padding, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        #self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.ConvTranspose2d(in_channel, num_hidden*4, (filter_size,filter_size), (stride,stride), (padding,padding), bias=False),
                #nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_h = nn.Sequential(
                #nn.ConvTranspose2d(num_hidden, num_hidden*4, (filter_size,filter_size), (stride,stride), (padding,padding), bias=False),
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=3, stride=1, padding=1, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            '''
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            
            self.conv_o = nn.Sequential(
                nn.ConvTranspose2d(num_hidden * 1, num_hidden, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
            '''
            
        else:
            self.conv_x = nn.Sequential(
                nn.ConvTranspose3d(in_channel, num_hidden*4, (1,filter_size,filter_size), (1,stride,stride), (0,padding,padding), bias=False)
                #nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                #nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.ConvTranspose3d(num_hidden, num_hidden*4, (1,filter_size,filter_size), (1,stride,stride), (0,padding,padding), bias=False)
            )
            '''
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            
            
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 1, num_hidden, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
            )
            '''
            
        #self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, s[[tride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        #print('h_t_shaoe_in ', h_t.shape)
        h_concat = self.conv_h(h_t)
        
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        #print('c_t_shape: ', c_t.shape, 'f_t_shape: ', f_t.shape)
        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h)
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new









