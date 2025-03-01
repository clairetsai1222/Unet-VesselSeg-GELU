import torch
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 200)   # 构造一段连续的数据
x = Variable(x)      # 转换成张量
x_np = x.data.numpy()    #plt中形式需要numpy形式，tensor形式会报错

y_gelu= F.gelu(x).data.numpy()    #torch.nn.functional中调用GELU函数
plt.plot(x_np, y_gelu, c='red', label='GELU')
plt.grid()
plt.legend(loc='best')
plt.show()