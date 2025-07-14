import os
from safetensors import safe_open
from safetensors.torch import save_file

# --- 配置区 ---
# 你的模型目录路径
model_dir = "model_c2h"
# 原始权重文件名
original_filename = "model.safetensors"
# 修正后要保存的新文件名
corrected_filename = "model_corrected.safetensors"
# --- 结束配置 ---

# 构造完整的文件路径
original_filepath = os.path.join(model_dir, original_filename)
corrected_filepath = os.path.join(model_dir, corrected_filename)

print(f"正在加载原始权重文件: {original_filepath}")

# 1. 创建一个新的字典来存放修正后的权重
new_state_dict = {}
original_metadata = None # 用于存储元数据

# 2. 使用 safe_open 来安全地读取文件，这样可以同时获取权重和元数据
with safe_open(original_filepath, framework="pt", device="cpu") as f:

    # 新增：读取原始元数据
    original_metadata = f.metadata()
    if original_metadata:
        print(f"成功读取到元数据: {original_metadata}")
    else:
        print("警告：原始文件中未发现元数据。")

    # 定义需要被替换的错误前缀和正确的前缀
    prefix_to_remove = "model.vision_encoder.vision_encoder."
    prefix_to_expect = "model.vision_encoder."

    keys_to_fix_found = 0
    print("开始遍历并修正权重名称...")

    # 3. 遍历所有权重键名
    for key in f.keys():
        # 获取权重张量
        tensor = f.get_tensor(key)

        if key.startswith(prefix_to_remove):
            # 如果键名以错误的双重前缀开头，就修正它
            new_key = prefix_to_expect + key[len(prefix_to_remove):]
            new_state_dict[new_key] = tensor
            keys_to_fix_found += 1
        else:
            # 其他键名保持不变
            new_state_dict[key] = tensor

    print(f"修正完成！共找到并修正了 {keys_to_fix_found} 个 vision_encoder 的权重名称。")


# 4. 保存修正后的权重到新的 safetensors 文件
# 修改：保存时将之前读取的元数据一并传入
print(f"正在保存修正后的权重到: {corrected_filepath}")
save_file(new_state_dict, corrected_filepath, metadata=original_metadata)

print("\n成功！新的权重文件已生成，并且保留了原始元数据。")
print("下一步，请在终端中执行命令来替换旧文件。")