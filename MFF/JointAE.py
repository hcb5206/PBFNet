import torch
import torch.nn as nn
from ConnectionFunction import wavelet_fusion, laplacian_pyramid_fusion, nsct_fusion, energy_minimization_fusion, \
    gradient_field_fusion, feature_level_fusion, low_rank_matrix_decomposition_fusion, guided_filter_fusion
import time
from thop import profile


class DenseLayer_Encoder(nn.Module):
    def __init__(self, input_channels, growth_rate):
        super(DenseLayer_Encoder, self).__init__()
        self.DenseLayer = nn.Sequential(
            nn.Conv2d(input_channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(num_features=growth_rate, eps=1e-8, affine=True),
        )

    def forward(self, x, x_add, x_cat):
        out = self.DenseLayer(x)
        return out, torch.add(x_add, out), torch.cat([x_cat, torch.add(x_add, out)], dim=1)


class DenseBlock_Encoder(nn.Module):
    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock_Encoder, self).__init__()
        self.layers = nn.ModuleList()
        self.conv_res = nn.Conv2d(input_channels, growth_rate, kernel_size=1)
        for i in range(num_layers):
            layer = DenseLayer_Encoder(input_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def forward(self, x):
        new_feature_cat = None
        x_add = self.conv_res(x)
        x_cat = x
        features_add = [x]
        features_cat = [x]
        for layer in self.layers:
            new_feature_out, new_feature_add, new_feature_cat = layer(x, x_add, x_cat)
            x = new_feature_cat
            features_add.append(new_feature_out)
            for i in range(len(features_add) - 1):
                x_add = x_add + features_add[i + 1]
            features_cat.append(new_feature_add)
            x_cat = torch.cat(features_cat, dim=1)

        return new_feature_cat


class DenseLayer_Decoder(nn.Module):
    def __init__(self, input_channels, growth_rate):
        super(DenseLayer_Decoder, self).__init__()
        self.DenseLayer = nn.Sequential(
            nn.ConvTranspose2d(input_channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(num_features=growth_rate, eps=1e-8, affine=True)
        )

    def forward(self, x, x_add, x_cat):
        out = self.DenseLayer(x)
        return out, torch.add(x_add, out), torch.cat([x_cat, torch.add(x_add, out)], dim=1)


class DenseBlock_Decoder(nn.Module):
    def __init__(self, input_channels, num_layers, growth_rate):
        super(DenseBlock_Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.conv_res = nn.Conv2d(input_channels, growth_rate, kernel_size=1)
        for i in range(num_layers):
            layer = DenseLayer_Decoder(input_channels + i * growth_rate, growth_rate)
            self.layers.append(layer)

    def forward(self, x):
        new_feature_cat = None
        x_add = self.conv_res(x)
        x_cat = x
        features_add = [x]
        features_cat = [x]
        for layer in self.layers:
            new_feature_out, new_feature_add, new_feature_cat = layer(x, x_add, x_cat)
            x = new_feature_cat
            features_add.append(new_feature_out)
            for i in range(len(features_add) - 1):
                x_add = x_add + features_add[i + 1]
            features_cat.append(new_feature_add)
            x_cat = torch.cat(features_cat, dim=1)

        return new_feature_cat


class ReparameterizeLayer(nn.Module):
    def __init__(self, input_size, pool_size, hidden_size):
        super(ReparameterizeLayer, self).__init__()
        self.input_size = input_size
        self.pool_size = pool_size
        self.GAP = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.GMP = nn.AdaptiveMaxPool2d((pool_size, pool_size))
        self.w1 = nn.Linear(in_features=pool_size * pool_size, out_features=1)
        self.w2 = nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(
            self.w1(torch.add(self.GMP(x), self.GAP(x)).reshape(x.shape[0], x.shape[1], -1)).unsqueeze(dim=3))
        return self.w2(torch.add(x * att, x))


class JointAE(nn.Module):
    def __init__(self, input_size, hidden_size, AE_num_layers, pool_size, modal_sel, weigh=0.5):
        super(JointAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.AE_num_layers = AE_num_layers
        self.pool_size = pool_size
        self.modal_sel = modal_sel
        self.weigh = weigh

        self.encoder_A = DenseBlock_Encoder(input_channels=input_size, num_layers=AE_num_layers,
                                            growth_rate=hidden_size)
        self.encoder_B = DenseBlock_Encoder(input_channels=input_size, num_layers=AE_num_layers,
                                            growth_rate=hidden_size)
        self.Reparameterize = ReparameterizeLayer(input_size=(input_size + AE_num_layers * hidden_size) * 2,
                                                  pool_size=pool_size, hidden_size=hidden_size)
        self.decoder = DenseBlock_Decoder(input_channels=hidden_size, num_layers=AE_num_layers,
                                          growth_rate=hidden_size)
        self.output = nn.Conv2d(in_channels=hidden_size + AE_num_layers * hidden_size, out_channels=input_size,
                                kernel_size=1)

    def forward(self, x_A, x_B):
        simi = self.Reparameterize(torch.cat((self.encoder_A(x_A), self.encoder_B(x_B)), dim=1))
        if self.modal_sel == 'xA':
            return self.output(self.decoder(simi)) + x_A
        elif self.modal_sel == 'xB':
            return self.output(self.decoder(simi)) + x_B
        elif self.modal_sel == 'xAB':
            return self.output(self.decoder(simi)) + (self.weigh * x_A + (1 - self.weigh) * x_B)
        elif self.modal_sel == 'x_add':
            return self.output(self.decoder(simi)) + (x_A + x_B)
        elif self.modal_sel == 'x_max':
            return self.output(self.decoder(simi)) + torch.maximum(x_A, x_B)
        elif self.modal_sel == 'x_wav':
            return self.output(self.decoder(simi)) + wavelet_fusion(x_A, x_B, fusion_method='average').to(x_A.device)
        elif self.modal_sel == 'x_lap':
            return self.output(self.decoder(simi)) + laplacian_pyramid_fusion(x_A, x_B).to(x_A.device)
        elif self.modal_sel == 'x_nsct':
            return self.output(self.decoder(simi)) + nsct_fusion(x_A, x_B).to(x_A.device)
        elif self.modal_sel == 'x_ey':
            return self.output(self.decoder(simi)) + energy_minimization_fusion(x_A, x_B).to(x_A.device)
        elif self.modal_sel == 'x_gra':
            return self.output(self.decoder(simi)) + gradient_field_fusion(x_A, x_B).to(x_A.device)
        elif self.modal_sel == 'x_fea':
            return self.output(self.decoder(simi)) + feature_level_fusion(x_A, x_B).to(x_A.device)
        elif self.modal_sel == 'x_lr':
            return self.output(self.decoder(simi)) + low_rank_matrix_decomposition_fusion(x_A, x_B).to(x_A.device)
        elif self.modal_sel == 'x_gui':
            return self.output(self.decoder(simi)) + guided_filter_fusion(x_A, x_B).to(x_A.device)
        elif self.modal_sel == 'no':
            return self.output(self.decoder(simi))
        else:
            raise ValueError(f"Not in modal_sel selection range!")


# if __name__ == '__main__':
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     torch.manual_seed(1)
#     x1 = torch.rand(4, 3, 424, 624, dtype=torch.float32).to(device)
#     x2 = torch.rand(4, 3, 424, 624, dtype=torch.float32).to(device)
#     model = JointAE(input_size=3, hidden_size=64, AE_num_layers=2, pool_size=32, modal_sel='xB').to(device)
#     for name, param in model.named_parameters():
#         print(f"Parameter {name}: {param.numel()} elements")
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total number of parameters: {total_params}")
#     output = model(x1, x2)
#     print(output.shape)

# def calculate_flops_and_time(model, input_size=(1, 3, 424, 624), device='cuda:0'):
#     device = torch.device(device if torch.cuda.is_available() else 'cpu')
#     input_tensor_A = torch.randn(*input_size).to(device)
#     input_tensor_B = torch.randn(*input_size).to(device)
#     model = model.to(device)
#
#     flops, params = profile(model, inputs=(input_tensor_A, input_tensor_B))
#     params_in_million = params / 1e6
#     flops_in_gb = flops / 1e9
#     start_time = time.time()
#     model(input_tensor_A, input_tensor_B)
#     end_time = time.time()
#     forward_time = end_time - start_time
#
#     return flops_in_gb, params_in_million, forward_time
#
#
# if __name__ == "__main__":
#     model = JointAE(input_size=3, hidden_size=64, AE_num_layers=4, pool_size=64, modal_sel='xB')
#     flops, params, forward_time = calculate_flops_and_time(model, device='cuda:0')
#
#     print(f"FLOPs: {flops:.4f}G")
#     print(f"Parameters: {params:.4f}M")
#     print(f"Forward Time: {forward_time:.4f} seconds")
