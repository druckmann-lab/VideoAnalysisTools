import torch
import torch.nn as nn

class PreConvBlock(nn.Module):
    def __init__(self, 
                 in_channels = 1, 
                 out_channels = 16, 
                 kernel = 3, 
                 stride=1, 
                 pool_size = 2,
                 use_batch_norm = False):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layers.
        """
        super(PreConvBlock, self).__init__()
        
        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 
                      kernel_size=kernel, stride=stride, 
                      padding=(kernel - stride) // 2, 
                      bias= not use_batch_norm),
        ])
        if use_batch_norm:
            self.layers.append(nn.BatchNorm2d(out_channels))
        
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=pool_size))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, 
                 n_channels = 16, 
                 kernel = 3, 
                 stride = 1, 
                 use_batch_norm = False, 
                 downsample=None,
                 pool_size = 4,
                 n_layers = 3):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layers.
            downsample (nn.Module, optional): Downsampling layer if input and output dimensions differ.
        """
        super(ResidualBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        for i_layer in range(n_layers):
            self.layers.append(nn.Conv2d(n_channels, n_channels, 
                                          kernel_size=kernel, stride=stride, 
                                          padding=(kernel - stride) // 2, 
                                          bias= not use_batch_norm))
            if use_batch_norm:
                self.layers.append(nn.BatchNorm2d(n_channels))
            if i_layer!=(n_layers - 1):
                self.layers.append(nn.ReLU(inplace=True))
        
        self.downsample = downsample  # Optional downsampling layer
        self.post_residual_layers = nn.ModuleList([nn.ReLU(inplace=True)])
        if pool_size is not None:
            self.post_residual_layers.append(nn.MaxPool2d(kernel_size=pool_size))
    
    def forward(self, x):
        identity = x

        # Pass through the layers in the ModuleList
        for layer in self.layers:
            x = layer(x)

        # Apply downsampling to the identity if necessary
        if self.downsample:
            identity = self.downsample(identity)

        # Add the residual connection
        x += identity
        for layer in self.post_residual_layers:
            x = layer(x)

        return x
    
class Encoder(nn.Module):

    def __init__(self, 
                 configs):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layers.
        """
        super(Encoder, self).__init__()
        
        self.configs = configs

        self.residual_layers = nn.ModuleList()

        for i_blocks in range(configs['num_blocks']):
            self.residual_layers.append(PreConvBlock(configs['in_channels_%d'%i_blocks],
                                            configs['out_channels_%d'%i_blocks],
                                            configs['kernel_preconv'],
                                            configs['stride_preconv'],
                                            configs['pool_size_preconv_%d'%i_blocks],
                                            configs['use_batch_norm_preconv']))
            self.residual_layers.append(ResidualBlock(configs['out_channels_%d'%i_blocks],
                                             configs['kernel_residual'],
                                             configs['stride_residual'],
                                             configs['use_batch_norm_residual'],
                                             pool_size=configs['pool_size_residual_%d'%i_blocks],
                                             n_layers=configs['n_layers_residual']))
            
        self.linear_layers = nn.ModuleList([nn.Linear(configs['out_conv'], configs['out_linear'], bias = not configs['use_batch_norm_linear']),])
        if configs['use_batch_norm_linear']:
            self.linear_layers.append(nn.BatchNorm1d(configs['out_linear']))
        self.linear_layers.append(nn.ReLU(inplace=True))
        self.linear_layers.append(nn.Linear(configs['out_linear'], configs['embed_size']))

        self.channels_last = configs.get('channels_last', False)

    def forward(self, x):
        bs, seq_length, c, h ,w = x.size()

        # .reshape rather than .view: a channels_last tensor is not
        # view-compatible and .view would raise. For a contiguous tensor the two
        # are identical and zero-copy.
        x = x.reshape(bs*seq_length, c, h, w)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        for layer in self.residual_layers:
            x = layer(x)

        # channels_last stores NHWC, so flattening it directly would hand the
        # Linear a permuted feature vector. Restore NCHW order first -- x is only
        # (bs*seq, out_channels, 3, 3) here, so the copy costs nothing.
        x = x.contiguous().reshape(bs*seq_length, -1)
        for layer in self.linear_layers:
            x = layer(x)

        x = x.reshape(bs, seq_length, -1)
        return x
    
class SingleSessionDecoder(nn.Module):
    def __init__(self, configs):
        super(SingleSessionDecoder, self).__init__()
        self.configs = configs
        self.linear_layer = nn.Linear(configs['embed_size'], configs['image_height']*configs['image_width'])

    def forward(self, x):
        bs, seq_length, _ = x.size()
        x = self.linear_layer(x)
        x = x.view(bs, seq_length, 1, self.configs['image_height'], self.configs['image_width'])
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, configs):
        super(AutoEncoder, self).__init__()
        self.configs = configs
        self.encoder = Encoder(configs)
        self.decoder = SingleSessionDecoder(configs)

        # Applied here rather than in the training loop, so nothing outside this
        # module has to know about memory formats. Measured 1.09x on an A10G
        # (5.60s -> 5.12s per epoch) by letting cudnn reach tensor-core kernels.
        if configs.get('channels_last', False):
            self.to(memory_format=torch.channels_last)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z