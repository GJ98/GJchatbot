from torch import nn

class Linear(nn.Module):

    def __init__(self, in_dim, out_dim, acti):
        """Linear - activation function 신경망 클래스"""

        super().__init__()

        self.linear = nn.Linear(in_features=in_dim, 
                                out_features=out_dim)

        if acti is 'softmax':
            self.acti = nn.Softmax(dim=1)
        elif acti is 'relu':
            self.acti = nn.ReLU()

    def forward(self, x):
        linear_x = self.linear(x.float())
        acti_x = self.acti(linear_x)
        return acti_x
