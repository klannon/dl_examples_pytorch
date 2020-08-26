import torch

class SharedModel(torch.nn.Module):
    def __init__(self,H):
        super(SharedModel,self).__init__()
        self.linear1 = torch.nn.Linear(1,H)
        torch.nn.init.kaiming_normal_(self.linear1.weight,a=0.25, nonlinearity='leaky_relu')
        torch.nn.init.constant_(self.linear1.bias,0)
        self.activation = torch.nn.PReLU()
        self.linear2 = torch.nn.Linear(H,1)
        torch.nn.init.kaiming_normal_(self.linear2.weight,a=0.25, nonlinearity='leaky_relu')
        torch.nn.init.constant_(self.linear2.bias,0)

    def forward(self,x):
        vars = torch.chunk(x,x.shape[1],1)
        outs = []
        for v in vars:
            o = self.linear1(v)
            o = self.activation(o)
            o = self.linear2(o)
            outs.append(o)
        return torch.cat(outs,1)

class StandardizationLayer(torch.nn.Module):
    def __init__(self,N):
        super(StandardizationLayer,self).__init__()
        self.scale = torch.nn.Parameter(torch.ones(N))
        self.offset = torch.nn.Parameter(torch.zeros(N))

    def forward(self,x):
        return self.scale*(x+self.offset)

    def initializeWeights(self,scale,offset):
        self.scale = torch.nn.Parameter(torch.Tensor(scale))
        self.offset = torch.nn.Parameter(torch.Tensor(offset))

