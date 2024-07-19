import torch.nn as nn

class Converter(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Converter, self).__init__()

        medium_num=input_nc//2
        self.model = nn.Sequential(
            nn.Linear(input_nc, medium_num),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(0.2),
            nn.Linear( medium_num, medium_num),
            nn.InstanceNorm1d(1),
            nn.LeakyReLU(0.2),
            nn.Linear(medium_num, output_nc),
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        x=self.model(x)
        x=x.view(x.size(0),-1)
        return x

