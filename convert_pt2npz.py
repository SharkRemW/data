import torch
import numpy as np
import os


dataset = 'disney'

dataset_dir = os.path.join('./data')

pt_path = os.path.join(dataset_dir, dataset + '.pt')
npz_path = os.path.join(dataset_dir, dataset + '.npz')

data = torch.load(pt_path) # in PyG format
print(data)

# 2. 保存到 .npz 文件
np.savez_compressed(
    npz_path,
    x=data.x.numpy(),
    edge_index=data.edge_index.numpy(),
    y=data.y.numpy(),
    # train_mask=data.train_mask.numpy(),
    # val_mask=data.val_mask.numpy(),
    # test_mask=data.test_mask.numpy(),
)

# 3. 加载数据
loaded_data = np.load(npz_path)

# 4. 验证函数
def verify(original_tensor, loaded_array, name):
    # 转换为 numpy 用于比较
    original_array = original_tensor.numpy() if isinstance(original_tensor, torch.Tensor) else original_tensor
    
    # 检查类型
    assert isinstance(loaded_array, np.ndarray), f"{name}: 加载类型不是 np.ndarray"
    
    # 检查形状
    assert original_array.shape == loaded_array.shape, f"{name}: 形状不一致 {original_array.shape} vs {loaded_array.shape}"
    
    # 检查数值
    assert np.allclose(original_array, loaded_array, atol=1e-6), f"{name}: 数值不一致"
    
    print(f"{name}: 验证通过 ✓")

# 5. 逐个验证变量
verify(data.x, loaded_data['x'], 'x')
verify(data.edge_index, loaded_data['edge_index'], 'edge_index')
verify(data.y, loaded_data['y'], 'y')
# verify(data.train_mask, loaded_data['train_mask'], 'train_mask')
# verify(data.val_mask, loaded_data['val_mask'], 'val_mask')
# verify(data.test_mask, loaded_data['test_mask'], 'test_mask')

print("所有变量验证完成！")