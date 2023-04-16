import torch
import torch.nn as nn
import torch.nn.functional as F



class FuzzyLayer(nn.Module):

    def __init__(self, input_dim=4, rule_dim=16, output_dim=1, gauss_mean=None,gauss_sigma=None):
        super(FuzzyLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rule_dim = rule_dim
        if gauss_mean == None:
            print("随机化input MF均值")
            gauss_mean = torch.rand(self.input_dim, self.rule_dim)
            self.gauss_mean_parameter=nn.Parameter(gauss_mean)
        else:
            assert isinstance(gauss_mean, torch.Tensor), "gauss_mean 必须是一个张量"
            self.gauss_mean_parameter = nn.Parameter(gauss_mean)
        if gauss_sigma == None:
            print("随机化input MF方差")
            # gauss_sigma = torch.rand(self.input_dim, self.rule_dim)
            gauss_sigma = torch.ones(self.input_dim, self.rule_dim)
            self.gauss_sigma_parameter=nn.Parameter(gauss_sigma)
        else:
            assert isinstance(gauss_sigma, torch.Tensor), "gauss_sigma 必须是一个张量"
            self.guass_sigma_parameter = nn.Parameter(gauss_sigma)
        lamda_parameter=torch.rand(self.rule_dim)
        lamda_parameter=lamda_parameter.view(-1,1)
        self.lamda_parameter_col = nn.Parameter(lamda_parameter)
        
        
    def forward(self, input):
        input=input.unsqueeze(-1) #用于下一步广播减法
        input=(input-self.gauss_mean_parameter)/(torch.square(self.gauss_sigma_parameter))       
        input=torch.square(input)*(-0.5)
        input=torch.exp(torch.einsum("ijk->ik",input))  #i是batch维度 j是x输入的维度 k是rule的维度
        denominator=torch.sum(input,dim=-1)
        return (torch.mm(input,self.lamda_parameter_col))/denominator
            
        

if __name__ == "__main__":
    
    # model = FuzzyLayer().cuda()
    model = FuzzyLayer().cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    for _ in range(1000):
        input=torch.Tensor([[1,2,3,4],[1,2,3,4]]).cpu()
        preds=model(input)
        gts=torch.Tensor([5,5]).to("cpu")
        loss=F.l1_loss(preds, gts)
        print(f"\r loss:{loss}", end="")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(model.gauss_mean_parameter)