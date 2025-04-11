import json

# 输入输出文件路径（可以改为其他文件名）
input_path = '/work/lei_sun/datasets/Minimalism/metadata_add_width_height.json'
output_path = input_path  # 或直接用 input_path 覆盖

old_prefix = "datasets/Minimalism/train/"
new_prefix = "train/"

# 加载列表格式的 JSON 文件
with open(input_path, 'r') as f:
    metadata_list = json.load(f)

# 遍历每个元素（都是 dict），检查是否有包含旧路径的值
for item in metadata_list:
    for key, value in item.items():
        if isinstance(value, str) and old_prefix in value:
            item[key] = value.replace(old_prefix, new_prefix)

# 写回修改后的结果
with open(output_path, 'w') as f:
    json.dump(metadata_list, f, indent=4)

print(f"替换完成，保存至 {output_path}")