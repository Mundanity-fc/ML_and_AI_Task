# 导入 MindSpore 模块和辅助模块
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context
from matplotlib import pyplot as plt

# 当前实验选择算力为 Ascend，如果在本地体验，参数 device_target 设置为"CPU”
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


# 生成模拟数据
# 根据线性函数 y = -5 * x + 0.1 生成模拟数据，并在其中加入少许扰动。
x = np.arange(-5, 5, 0.3)[:32].reshape((32, 1))
y = -5 * x + 0.1 * np.random.normal(loc=0.0, scale=20.0, size=x.shape)


# 建模
# 网络中增加 1x1 大小的 Dense 层
net = nn.Dense(1, 1)
# 确立损失函数
loss_fn = nn.loss.MSELoss()
# 确立优化器
opt = nn.optim.SGD(net.trainable_params(), learning_rate=0.01)
# 绑定损失函数
with_loss = nn.WithLossCell(net, loss_fn)
# 确立训练步长
train_step = nn.TrainOneStepCell(with_loss, opt).set_train()


# 使用模拟数据训练模型
for epoch in range(20):
    loss = train_step(ms.Tensor(x, ms.float32), ms.Tensor(y, ms.float32))
print('epoch: {0}, loss is {1}'.format(epoch, loss))


# 使用训练好的模型进行预测
wb = [x.asnumpy() for x in net.trainable_params()]
# 获取模型参数
w, b = np.squeeze(wb[0]), np.squeeze(wb[1])
# 输出模型参数
print('The true linear function is y = -5 * x + 0.1')
print('The trained linear model is y = {0} * x + {1}'.format(w, b))
# 从 -10 到 10，间隔为 5，进行预测
for i in range(-10, 11, 5):
    print('x = {0}, predicted y = {1}'.format(i, net(ms.Tensor([[i]], ms.float32))))


# 可视化
# %matplotlib inline
plt.scatter(x, y, label='Samples')
# 绘制回归模型
plt.plot(x, w * x + b, c='r', label='True function')
# 绘制真实模型
plt.plot(x, -5 * x + 0.1, c='b', label='Trained model')
# 增加图例
plt.legend()
# 显示图标（IDE中）
plt.show()
