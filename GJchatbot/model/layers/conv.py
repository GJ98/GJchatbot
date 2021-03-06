from torch import nn

class Conv(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size):
        """Convolution - acitvation function 신경망 클래스다"""

        super().__init__()

        self.conv = nn.Conv1d(in_channels=in_dim,
                              out_channels=out_dim,
                              kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_x = self.conv(x.float())
        relu_x = self.relu(conv_x)
        return relu_x
