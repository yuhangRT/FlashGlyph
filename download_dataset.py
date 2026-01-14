#!/usr/bin/env python
"""
下载AnyWord-3M数据集的脚本
"""
# import os
# from modelscope.msdatasets import MsDataset

# def download_anyword_dataset():
#     """下载AnyWord-3M数据集"""
#     print("=" * 60)
#     print("AnyWord-3M 数据集下载脚本")
#     print("=" * 60)

#     # 尝试加载数据集
#     print("\n正在尝试加载 iic/AnyWord-3M 数据集...")

#     try:
#         # 方法1: 使用MsDataset.load
#         ds = MsDataset.load(
#             'iic/AnyWord-3M',
#             split='train'  # 可以是 'train', 'test', 'validation'
#         )

#         print(f"\n✓ 数据集加载成功!")
#         print(f"  - 数据集类型: {type(ds)}")
#         print(f"  - 数据集大小: {len(ds)} 条")

#         # 检查数据集特征
#         if hasattr(ds, 'features'):
#             print(f"  - 数据集特征: {ds.features}")

#         # 查看第一条数据
#         print(f"\n第一条数据示例:")
#         print(f"{ds[0]}")

#         # 获取数据集缓存目录
#         print(f"\n数据集缓存位置:")
#         if hasattr(ds, 'cache_files'):
#             for cache_file in ds.cache_files:
#                 print(f"  - {cache_file}")

#         return ds

#     except Exception as e:
#         print(f"\n✗ 数据集加载失败: {e}")
#         print("\n可能的原因:")
#         print("1. 数据集名称不正确")
#         print("2. 需要登录ModelScope账号")
#         print("3. 数据集访问权限受限")
#         print("\n建议:")
#         print("1. 访问 https://www.modelscope.cn/datasets/iic/AnyWord-3M 确认数据集名称")
#         print("2. 使用以下命令登录:")
#         print("   from modelscope.hub.api import HubApi")
#         print("   api = HubApi()")
#         print("   api.login(token='your_token')")
#         print("\n替代方案: 直接从项目页面下载数据集")

#         import traceback
#         traceback.print_exc()
#         return None


# def alternative_download():
#     """替代方案: 使用snapshot_download"""
#     print("\n" + "=" * 60)
#     print("尝试使用snapshot_download方法...")
#     print("=" * 60)

#     from modelscope import snapshot_download

#     # 可能的数据集名称
#     dataset_names = [
#         'iic/AnyWord-3M',
#         'iic/anyword-3m',
#         'iic/any_text',
#     ]

#     for name in dataset_names:
#         print(f"\n尝试下载: {name}")
#         try:
#             cache_dir = snapshot_download(name)
#             print(f"✓ 成功下载到: {cache_dir}")
#             return cache_dir
#         except Exception as e:
#             print(f"✗ 失败: {e}")

#     return None


# if __name__ == '__main__':
#     print("\n注意: 在运行此脚本之前，请确保:")
#     print("1. 已安装 modelscope: pip install modelscope")
#     print("2. 已安装 datasets: pip install datasets")
#     print("3. 网络连接正常\n")

#     # 尝试方法1
#     ds = download_anyword_dataset()

#     # 如果方法1失败，尝试方法2
#     if ds is None:
#         cache_dir = alternative_download()

#     print("\n" + "=" * 60)
#     print("下载脚本执行完毕")
#     print("=" * 60)


from modelscope import snapshot_download


# 下载模型/数据集到本地
# cache_dir = snapshot_download(
#     'iic/AnyWord-3M',
#     cache_dir='./data/AnyWord-3M'  # 指定下载目录
# )

models = snapshot_download(
    'iic/cv_anytext2',
    cache_dir='./cv_anytext2'  # 指定下载到当前目录的models文件夹
)