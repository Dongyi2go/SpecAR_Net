import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from sklearn.preprocessing import MinMaxScaler
# import scipy.signal as signal
import scipy.signal as signal
import numpy as np
from layers.ASPP import *


class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,3))
            self.bias = nn.Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, dtype = torch.complex64))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:,:2],1.4142135623730951)
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)

class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input.real.mean([0, 2, 3]).type(torch.complex64)
            mean_i = input.imag.mean([0, 2, 3]).type(torch.complex64)
            mean = mean_r + 1j * mean_i
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean

        input = input - mean[None, :, None, None]

        if self.training or (not self.training and not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input.numel() / input.size(1)
            Crr = 1. / n * input.real.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1. / n * input.imag.pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (input.real.mul(input.imag)).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = exponential_average_factor * Crr * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 0]

                self.running_covar[:, 1] = exponential_average_factor * Cii * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 1]

                self.running_covar[:, 2] = exponential_average_factor * Cri * n / (n - 1) \
                                           + (1 - exponential_average_factor) * self.running_covar[:, 2]

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input = (Rrr[None, :, None, None] * input.real + Rri[None, :, None, None] * input.imag).type(torch.complex64) \
                + 1j * (Rii[None, :, None, None] * input.imag + Rri[None, :, None, None] * input.real).type(
            torch.complex64)

        if self.affine:
            input = (self.weight[None, :, 0, None, None] * input.real + self.weight[None, :, 2, None,
                                                                        None] * input.imag + \
                     self.bias[None, :, 0, None, None]).type(torch.complex64) \
                    + 1j * (self.weight[None, :, 2, None, None] * input.real + self.weight[None, :, 1, None,
                                                                               None] * input.imag + \
                            self.bias[None, :, 1, None, None]).type(torch.complex64)
        return input

def complex_relu(input):
    return F.relu(input.real)+1j* F.relu(input.imag)

#Complex convolution Block
def apply_complexConv2D(fr, fi, input):

    return fr(input.real)-fi(input.imag) \
            + 1j*(fr(input.imag)+fi(input.real))

class ComplexConv2D_Block(nn.Module):

    def __init__(self ,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2D_Block, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self ,input):
        return apply_complexConv2D(self.conv_r, self.conv_i, input)

#Dual Channel Convolution Block
def apply_DualChannelConv2D(fr, fi, input):

    return fr(input.real)-1j*fi(input.imag)

class DualChannelConv2D_Block(nn.Module):

    def __init__(self ,in_channels, out_channels, kernel_size=3, stride=1, padding = 0,
                 dilation=1, groups=1, bias=True):
        super(DualChannelConv2D_Block, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self ,input):
        return apply_DualChannelConv2D(self.conv_r, self.conv_i, input)


class SpecAR_Block(nn.Module):
    def __init__(self, configs):
        super(SpecAR_Block, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.conv_layers=configs.conv_layers
        self.seasonal_patterns=configs.seasonal_patterns
        self.n_fft=configs.n_fft
        self.win_len=configs.win_len
        self.hop_length=configs.hop_length
        self.win_func=configs.win_func

        self.cnn = nn.ModuleList()
        if configs.backbone=="CompelxConv2d":
            Conv2D=ComplexConv2D_Block
        else:
            Conv2D=DualChannelConv2D_Block

        self.single_ComplexConv_dila0 = Conv2D(
            in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=1, stride=1, dilation=1, padding=0
        )
        self.cnn.append(self.single_ComplexConv_dila0)

        for k in range(1,self.conv_layers):
            double_ComplexConv_dila = nn.Sequential(
                Conv2D(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=3, stride=1,
                              dilation=k, padding=k),
                # ComplexBatchNorm2d(configs.d_ff, track_running_stats=False),
                Conv2D(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=3, stride=1,
                              dilation=k, padding=k)
            )
            self.cnn.append(double_ComplexConv_dila)

        self.l0=nn.Linear(2,1)

        self.l2 = nn.Linear(in_features=int((configs.seq_len + configs.pred_len+configs.hop_length)/configs.hop_length), out_features=(configs.seq_len + configs.pred_len))

    # Complex Multiplication
    def compl_mul1d(self, order, x, weights):

        return torch.complex(torch.einsum(order, x.real, weights.real) - torch.einsum(order, x.imag, weights.imag),
                                 torch.einsum(order, x.real, weights.imag) + torch.einsum(order, x.imag, weights.real))

    def forward(self, x):
        B, T, N = x.size()
        res_stft_list = []
        win_len=self.win_len
        n_fft=self.n_fft
        hop_length=self.hop_length

        #short-time Fourier transform
        if self.win_func=='hanning':
            window = torch.hann_window(win_len).type(torch.float).to(x.device)
        elif self.win_func=='hamming':
            window=torch.hamming_window(win_len).type(torch.float).to(x.device)
        elif self.win_func=='rectangle':
            window = torch.ones(win_len).type(torch.float).to(x.device)
        elif self.win_func=='triang':
            window = torch.from_numpy(signal.triang(win_len)).type(torch.float).to(x.device)
        else:
            window = torch.blackman_window(win_len).type(torch.float).to(x.device)

        for j in range(B):
            input_x = x[j, :, :].permute(1, 0)  # (512,192)

            res_stft = torch.stft(input=input_x, n_fft=n_fft, hop_length=hop_length, win_length=win_len, center=True,
                                   pad_mode='reflect', window=window, normalized=True, onesided=False,
                                   return_complex=True)
            res_stft_list.append(res_stft)

        res_stft_ = torch.stack(res_stft_list, dim=0)

        #attention
        res_stft_qk=self.compl_mul1d('ijxk,ijyk -> ijxy',res_stft_,res_stft_)
        res_stft_qk_=torch.complex(res_stft_qk.real.sigmoid(), res_stft_qk.imag.sigmoid())
        res_stft_atten=self.compl_mul1d('ijxy,ijyz->ijxz',res_stft_qk_,res_stft_)

        # complexConv2d
        result_list = []
        for i in range(self.conv_layers):
            res_stft_conv = complex_relu(self.cnn[i](res_stft_atten))
            result_list.append(res_stft_conv)  # list:((batch_size,512,192,193))

        # stack
        res_stack = torch.stack(result_list, dim=-1).mean(-1)# (batch_size,512,192,193)

        #complex -> real
        res_complex=torch.stack([res_stack.real,res_stack.imag],dim=-1)#(batch_size,512,192,193,2)
        res_complex_to_real = torch.squeeze(self.l0(res_complex),dim=4) # (batch_size,512,192,193)

        if hop_length !=1:
            res=self.l2( res_complex_to_real.mean(2))
        else:
            res = res_complex_to_real.mean(2)[:, :, :T]# (batch_size,512,192)

        res =torch.permute(res,(0,2,1))# (batch_size,192,512)

        return res

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.all_len = configs.seq_len + configs.pred_len

        self.model = nn.ModuleList([SpecAR_Block(configs)
                                    for _ in range(configs.e_layers)])

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

        self.conv1d=nn.Conv1d(in_channels=self.all_len,out_channels=self.all_len,kernel_size=3,stride=2,padding=1)
        self.l1=nn.Linear(in_features=int(configs.d_model/2),out_features=1)
        #**
        # self.l1 = nn.Linear(in_features=int(configs.d_model), out_features=1)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # (B,T,M)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        #isotonic regression
        seq_series=self.l1(self.conv1d(enc_out)).squeeze(-1) #(B,T)

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out,seq_series

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # (B,T,M)
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # isotonic regression
        seq_series = self.l1(self.conv1d(enc_out)).squeeze(-1) #(B,T)

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out,seq_series

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # print(enc_out.size())
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out ,seq_series= self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)#(B,T,N),(B,T)
            return dec_out[:, -self.pred_len:, :], seq_series

        if self.task_name == 'imputation':
            dec_out,seq_series = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask) #(B,T,N),(B,T)
            return dec_out,seq_series

        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]

        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, C]

        return None
