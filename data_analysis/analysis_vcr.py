import json

# 输入文件路径
input_file_path = '/home/zhaotian/VL/data/vcr/val.jsonl'

# 初始化统计变量
total_data_count = 0
why_data_count = 0

# 打开并统计数据
with open(input_file_path, 'r') as infile:
    for line in infile:
        total_data_count += 1
        data = json.loads(line)
        # 检查 'question_orig' 或 'question' 字段是否包含 'why'
        if 'why' in data.get('question_orig', '').lower() :
            why_data_count += 1

# 计算比例
why_data_ratio = (why_data_count / total_data_count) * 100 if total_data_count > 0 else 0

# 输出结果
print(f"总数据数量: {total_data_count}")
print(f"包含 'why' 的数据数量: {why_data_count}")
print(f"包含 'why' 的数据比例: {why_data_ratio:.2f}%")
