import torch
import torch.nn as nn
import math


class MAUCell(nn.Module):

    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, tau, cell_mode):
        super(MAUCell, self).__init__()

        self.num_hidden = num_hidden
        # self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self.padding = filter_size // 2
        self.cell_mode = cell_mode
        self.d = num_hidden * height * width
        self.tau = tau
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 3 * num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([3 * num_hidden, height, width])
        )   #
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 3 * num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([3 * num_hidden, height, width])
        )
        self.conv_s_next = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, T_t, S_t, t_att, s_att): 
        
        #   T_t     ->   T^{k}_{t-1}
        #   S_t     ->   S^{k-1}_{t}
        #   t_att   ->   T^{k}_{t-tau:t-1}
        #   s_att   ->   S^{k-1}_{t-tau:t-1}

        s_next = self.conv_s_next(S_t)  #   Ws * S^{k-1}_{t}
        t_next = self.conv_t_next(T_t)  #   Wf * T^{k}_{t-1}

        #   Attention_Map模块
        weights_list = []
        for i in range(self.tau):
            weights_list.append((s_att[i] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d)) #   [tau, d]
        weights_list = torch.stack(weights_list, dim=0) #   [tau, d]
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1)) #   [tau, 1, 1, 1]
        weights_list = self.softmax(weights_list)   #   [tau, 1, 1, 1]

        #   MUL
        T_trend = t_att * weights_list  #   [tau, d, h, w]

        #   SUM
        T_trend = T_trend.sum(dim=0) #  [1, d, h, w]

        #   U_f
        t_att_gate = torch.sigmoid(t_next)  #   [1, d, h, w]

        #   T_AMI
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend    #   [1, d, h, w]

        #   U_t
        T_concat = self.conv_t(T_fusion)    #   [1, 2d, h, w]

        #   U_s
        S_concat = self.conv_s(S_t) #   [1, 2d, h, w]

        #   分别生成权重W_tu、W_tt、W_ts
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)   

        #   分别生成权重W_su、W_st、W_ss
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)

        #   U_t
        T_gate = torch.sigmoid(t_g)

        #   U_s
        S_gate = torch.sigmoid(s_g)

        #输出结果S^{k}_{t}、T^{k}_{t}
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s

        if self.cell_mode == 'residual':    
            S_new = S_new + S_t
        return T_new, S_new
