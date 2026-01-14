# AnyWord-3M 数据集下载指南

## 问题描述

在尝试使用ModelScope下载AnyWord-3M数据集时遇到以下错误：

```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('iic/AnyWord-3M')
```

**错误信息**:
- `ModuleNotFoundError: No module named 'datasets'`
- `AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'`
- `<Response [404]>` - 数据集不存在

## 解决方案

### 1. 安装依赖包

```bash
# 进入base环境
conda activate base

# 安装必要的依赖
pip install datasets==2.14.6
pip install addict
pip install pyarrow>=12.0.0,<15.0.0
```

**重要**: 需要安装兼容版本的`pyarrow`，建议使用12.x或14.x版本。

### 2. 正确的数据集名称

根据README和项目文档，AnyWord-3M数据集的正确下载方式：

```python
from modelscope.msdatasets import MsDataset

# 方法1: 直接加载数据集
ds = MsDataset.load(
    'iic/AnyWord-3M',
    split='train'  # 或 'test', 'validation'
)

# 查看数据集信息
print(f"数据集大小: {len(ds)}")
print(f"第一条数据: {ds[0]}")
```

### 3. 如果数据集名称404错误

如果遇到404错误，可能的原因：

1. **数据集尚未公开** - AnyWord-3M可能需要特定权限
2. **数据集名称变更** - 检查ModelScope官方页面
3. **需要登录** - 需要登录ModelScope账号

#### 解决方法: 登录ModelScope

```python
from modelscope.hub.api import HubApi
from modelscope.msdatasets import MsDataset

# 登录
api = HubApi()
api.login(token='your_modelscope_token')  # 在ModelScope官网获取token

# 然后加载数据集
ds = MsDataset.load('iic/AnyWord-3M')
```

获取token的步骤:
1. 访问 https://www.modelscope.cn/
2. 登录账号
3. 进入个人中心 -> 访问令牌
4. 创建新的访问令牌

### 4. 替代方案: 手动下载数据集

如果ModelScope下载失败，可以使用以下替代方案：

#### 方案A: 使用项目提供的下载脚本

```bash
# 使用项目中的下载脚本
python download_dataset.py
```

#### 方案B: 直接从网页下载

1. 访问 ModelScope 数据集页面: https://www.modelscope.cn/datasets/iic/AnyWord-3M
2. 手动下载所需文件
3. 放到指定目录

#### 方案C: 使用snapshot_download

```python
from modelscope import snapshot_download

# 下载模型/数据集到本地
cache_dir = snapshot_download(
    'iic/AnyWord-3M',
    cache_dir='./data/AnyWord-3M'  # 指定下载目录
)
```

### 5. 验证数据集

下载后验证数据集是否正确：

```python
import os
from datasets import load_from_disk

# 如果数据集保存为arrow格式
ds = load_from_disk('./data/AnyWord-3M')

# 查看数据集信息
print(f"数据集大小: {len(ds)}")
print(f"数据集特征: {ds.features}")

# 查看第一条数据
example = ds[0]
print("示例数据:")
for key, value in example.items():
    print(f"  {key}: {value}")
```

## 数据集格式

AnyWord-3M数据集通常包含以下字段：

```json
{
  "image": "图像数据",
  "text_lines": [
    {
      "text": "文本内容",
      "font": "字体名称",
      "color": [R, G, B],
      "bbox": [x1, y1, x2, y2],
      "language": "Chinese/English"
    }
  ],
  "caption": "简短描述",
  "long_caption": "长描述"
}
```

## 使用数据集进行训练

下载数据集后，在训练脚本中配置路径：

```python
# train.py 中的配置
dataset_dir = './data/AnyWord-3M'
train_annotation = f'{dataset_dir}/annotations/train.json'
test_annotation = f'{dataset_dir}/annotations/test.json'
```

## 常见问题

### Q1: ModuleNotFoundError: No module named 'datasets'
```bash
pip install datasets==2.14.6
```

### Q2: pyarrow版本冲突
```bash
pip install pyarrow==14.0.0  # 或 12.0.0
```

### Q3: 404错误 - 数据集不存在
- 检查数据集名称是否正确
- 确认是否需要登录ModelScope
- 访问官方数据集页面确认可用性

### Q4: 下载速度慢
- 使用国内镜像源
- 分批下载
- 使用ModelScope的CDN加速

## 完整下载脚本示例

```python
#!/usr/bin/env python
"""完整的AnyWord-3M数据集下载脚本"""

import os
from modelscope.msdatasets import MsDataset
from modelscope.hub.api import HubApi

def main():
    # 1. 登录ModelScope (可选，但对于私有数据集是必需的)
    # api = HubApi()
    # api.login(token='your_token_here')

    # 2. 下载数据集
    print("开始下载AnyWord-3M数据集...")

    try:
        # 加载训练集
        train_ds = MsDataset.load(
            'iic/AnyWord-3M',
            split='train',
            cache_dir='./cache/AnyWord-3M'  # 指定缓存目录
        )
        print(f"✓ 训练集加载成功: {len(train_ds)} 条")

        # 加载测试集
        test_ds = MsDataset.load(
            'iic/AnyWord-3M',
            split='test',
            cache_dir='./cache/AnyWord-3M'
        )
        print(f"✓ 测试集加载成功: {len(test_ds)} 条")

        # 3. 验证数据集
        print("\n查看第一条训练数据:")
        example = train_ds[0]
        for key, value in example.items():
            if isinstance(value, (str, int, float, list, dict)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: <{type(value).__name__}>")

        print("\n数据集下载完成！")

    except Exception as e:
        print(f"✗ 下载失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
```

## 参考资源

- ModelScope数据集文档: https://modelscope.cn/docs
- AnyText2项目: https://github.com/tyxsspa/AnyText2
- AnyWord-3M数据集: https://modelscope.cn/datasets/iic/AnyWord-3M/summary

---

**注意**: 如果您仍然无法下载数据集，建议直接联系项目作者或查看项目的Issues页面获取帮助。
