from torch import nn


class TDNN(nn.Module):
    def __init__(self, input_channels, context_size):
        # takes a signal of size B x 16 x 15 and returns B x 3 x 1
        super(TDNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 8, 3, stride=1) # B x 8 x 13
        self.conv2 = nn.Conv1d(8, 3, 5, stride=1) # B x 3 x 9
        self.conv3 = nn.Conv1d(3, 3, 9, stride=1) # B x 3 x 1

    def forward(self, x):
        x = self.conv1(x).sigmoid()
        x = self.conv2(x).sigmoid()
        x = self.conv3(x).sigmoid()
        return x
    

if __name__ == '__main__':
    import torch
    x = torch.randn(2, 16, 15)
    model = TDNN(16, 15)
    print(model(x).shape)