import h5py
import numpy as np

# input, output, weight, bias are all matrix
# input data
with h5py.File('train_128.h5', 'r') as H:
    data = np.copy(H['data'])
# input label
with h5py.File('train_label.h5', 'r') as H:
    label = np.copy(H['label'])

print(data.shape)
print(label.shape)

# 10*128，对应10个label 0-9
# 测试用
b = np.arange(1280).reshape(10, 128)
c = np.array(range(0, 10))


def train_val_split(data, label, ratio=0.75, shuffle=False,):
    # ratio为测试/validation
    # shuffle为是否要打乱顺序
    # train,train_label为训练数据
    # val, val_label为验证数据
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(label)
    train = data[0: int(len(data)*ratio)]
    val = data[int(len(data)*ratio):]
    train_label = label[0: int(len(label)*ratio)]
    val_label = label[int(len(label)*ratio):]
    return train, val, train_label, val_label


# 测试
t1, v1, l1, l2 = train_val_split(b, c)
print(t1[:, 0])
print(v1[:, 0])
print(l1)
print(l2)

t1, v1, l1, l2 = train_val_split(b, c, shuffle=True)
print(t1[:, 0])
print(v1[:, 0])
print(l1)
print(l2)

# data和label的分离
t1, v1, l1, l2 = train_val_split(data, label)
print(t1[:, 0])
print(v1[:, 0])
print(l1)
print(l2)

t1, v1, l1, l2 = train_val_split(data, label, shuffle=True)
print(t1[:, 0])
print(v1[:, 0])
print(l1)
print(l2)