# 导入 MindSpore 模块和辅助模块
import os
# os.environ['DEVICE_ID'] = '6'
import csv
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import dataset
from mindspore.train.callback import LossMonitor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P
#当前实验选择算力为 Ascend，如果在本地体验，参数 device_target 设置为"CPU”
context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


# 读取 Iris 数据集，并查看部分数据
with open('./iris.data') as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))
# 打印部分数据
print(data[40:60])


# 抽取样本
# 建立标签字典
label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
}
X = np.array([[float(x) for x in s[:-1]] for s in data[:100]], np.float32)
Y = np.array([[label_map[s[-1]]] for s in data[:100]], np.float32)


# 样本可视化
from matplotlib import pyplot as plt
# %matplotlib inline
plt.scatter(X[:50, 0], X[:50, 1], label='Iris-setosa')
plt.scatter(X[50:, 0], X[50:, 1], label='Iris-versicolor')
# X 坐标说明
plt.xlabel('sepal length')
# Y 坐标说明
plt.ylabel('sepal width')
# 增加图例
plt.legend()
# 显示图标（IDE中）
plt.show()


# 分割数据集
train_idx = np.random.choice(100, 80, replace=False)
test_idx = np.array(list(set(range(100)) - set(train_idx)))
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]


# 数据类型转换
XY_train = list(zip(X_train, Y_train))
ds_train = dataset.GeneratorDataset(XY_train, ['x', 'y'])
# ds_train.set_dataset_size(80)
ds_train = ds_train.shuffle(buffer_size=80).batch(32, drop_remainder=True)


# 模型建立与训练
# 可视化逻辑回归函数
coor_x = np.arange(-10, 11, dtype=np.float32)
coor_y = nn.Sigmoid()(ms.Tensor(coor_x)).asnumpy()
plt.plot(coor_x, coor_y)
plt.xlabel('x')
plt.ylabel('p')
plt.show()

# 建模
# 自定义 Loss
class Loss(nn.Cell):
    def __init__(self):
        super(Loss, self).__init__()
        self.sigmoid_cross_entropy_with_logits = P.SigmoidCrossEntropyWithLogits()
        self.reduce_mean = P.ReduceMean(keep_dims=False)
    def construct(self, x, y):
        loss = self.sigmoid_cross_entropy_with_logits(x, y)
        return self.reduce_mean(loss, -1)

net = nn.Dense(4, 1)
loss = Loss()
opt = nn.optim.SGD(net.trainable_params(), learning_rate=0.003)


# 模型训练
model = ms.train.Model(net, loss, opt)
model.train(10, ds_train, callbacks=[LossMonitor(per_print_times=ds_train.get_dataset_size())],
            dataset_sink_mode=False)


# 模型评估
x = model.predict(ms.Tensor(X_test)).asnumpy()
pred = np.round(1 / (1 + np.exp(-x)))
correct = np.equal(pred, Y_test)
acc = np.mean(correct)
print('Test accuracy is', acc)