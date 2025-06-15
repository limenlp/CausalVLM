from datasets import load_dataset

# 设置数据集名称和分割
dataset_name = "laion/laion400m"
split = "train"

# 输出文件路径
output_file = "captions.txt"

# 加载数据集的流式版本
dataset = load_dataset(dataset_name, split=split, streaming=True)

# 打开文件以写入 caption
with open(output_file, "w", encoding="utf-8") as f:
    for i, data in enumerate(dataset):
        # 确保字段存在且 caption 不为 None
        if 'caption' in data and data['caption'] is not None:
            # print(i)
            # print(data['caption'])  # 输出 caption
            f.write(data['caption'] + "\n")  # 写入 caption，每条一行

        # 每 10,000 条打印进度
        if (i + 1) % 10000 == 0:
            print(f"已处理 {i + 1} 条数据")

print(f"处理完成，数据保存到：{output_file}")
