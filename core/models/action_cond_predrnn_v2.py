__author__ = 'yunbo'

import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_v2_action import SpatioTemporalLSTMCell
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.conv_on_input = self.configs.conv_on_input
        self.res_on_conv = self.configs.res_on_conv
        self.patch_height = configs.img_width // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size ** 2)
        self.action_ch = configs.num_action_ch
        self.rnn_height = self.patch_height
        self.rnn_width = self.patch_width

        if self.configs.conv_on_input == 1:
            self.rnn_height = self.patch_height // 4
            self.rnn_width = self.patch_width // 4
            self.conv_input1 = nn.Conv2d(self.patch_ch, num_hidden[0] // 2,
                                         configs.filter_size,
                                         stride=2, padding=configs.filter_size // 2, bias=False)
            self.conv_input2 = nn.Conv2d(num_hidden[0] // 2, num_hidden[0], configs.filter_size, stride=2,
                                         padding=configs.filter_size // 2, bias=False)
            self.action_conv_input1 = nn.Conv2d(self.action_ch, num_hidden[0] // 2,
                                                configs.filter_size,
                                                stride=2, padding=configs.filter_size // 2, bias=False)
            self.action_conv_input2 = nn.Conv2d(num_hidden[0] // 2, num_hidden[0], configs.filter_size, stride=2,
                                                padding=configs.filter_size // 2, bias=False)
            self.deconv_output1 = nn.ConvTranspose2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1] // 2,
                                                     configs.filter_size, stride=2, padding=configs.filter_size // 2,
                                                     bias=False)
            self.deconv_output2 = nn.ConvTranspose2d(num_hidden[num_layers - 1] // 2, self.patch_ch,
                                                     configs.filter_size, stride=2, padding=configs.filter_size // 2,
                                                     bias=False)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []
        self.beta = configs.decouple_beta
        self.MSE_criterion = nn.MSELoss().cuda()
        self.norm_criterion = nn.SmoothL1Loss().cuda()

        for i in range(num_layers):
            if i == 0:
                in_channel = self.patch_ch + self.action_ch if self.configs.conv_on_input == 0 else num_hidden[0]
            else:
                in_channel = num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.rnn_width,
                                       configs.filter_size, configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        if self.configs.conv_on_input == 0:
            self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch + self.action_ch, 1, stride=1,
                                       padding=0, bias=False)
        self.adapter = nn.Conv2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1], 1, stride=1, padding=0,
                                 bias=False)

    def forward(self, all_frames, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = all_frames.permute(0, 1, 4, 2, 3).contiguous()
        input_frames = frames[:, :, :self.patch_ch, :, :]
        input_actions = frames[:, :, self.patch_ch:, :, :]
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [self.configs.batch_size, self.num_hidden[i], self.rnn_height, self.rnn_width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        decouple_loss = []
        memory = torch.zeros([self.configs.batch_size, self.num_hidden[0], self.rnn_height, self.rnn_width]).cuda()

        for t in range(self.configs.total_length - 1):
            if t == 0:
                net = input_frames[:, t]
            else:
                net = mask_true[:, t - 1] * input_frames[:, t] + \
                      (1 - mask_true[:, t - 1]) * x_gen
            action = input_actions[:, t]

            if self.conv_on_input == 1:
                net_shape1 = net.size()
                net = self.conv_input1(net)
                if self.res_on_conv == 1:
                    input_net1 = net
                net_shape2 = net.size()
                net = self.conv_input2(net)
                if self.res_on_conv == 1:
                    input_net2 = net
                action = self.action_conv_input1(action)
                action = self.action_conv_input2(action)

            h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory, action)
            delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory, action)
                delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(0, self.num_layers):
                decouple_loss.append(torch.mean(torch.abs(
                    torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))
            if self.conv_on_input == 1:
                if self.res_on_conv == 1:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1] + input_net2, output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen + input_net1, output_size=net_shape1)
                else:
                    x_gen = self.deconv_output1(h_t[self.num_layers - 1], output_size=net_shape2)
                    x_gen = self.deconv_output2(x_gen, output_size=net_shape1)
            else:
                x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        loss = self.MSE_criterion(next_frames,
                                  all_frames[:, 1:, :, :, :next_frames.shape[4]]) + self.beta * decouple_loss
        next_frames = next_frames[:, :, :, :, :self.patch_ch]
        return next_frames, loss
