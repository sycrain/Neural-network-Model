import torch
from net import vgg16

device=torch.device('cuda'if torch.cuda.is_available() else "cpu")#电脑主机的选择
model = vgg16()
model.load_state_dict(torch.load('..\PTH_rate\Dbz10.pth'),device)
model.eval()
example = torch.ones(1,3,224,224)
torch.onnx.export(model,example,"..\PTH_rate/Dbz10.onnx",verbose = True,opset_version =11)
#verbose=True,  # 打印导出过程的详细信息
#opset_version=11  # 使用 opset 版本 11