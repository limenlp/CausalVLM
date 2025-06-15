def process_captions(file_path):
    """
    读取 captions.txt 文件并统计满足条件的数据数量及比例。

    Args:
        file_path (str): captions.txt 文件路径。

    Returns:
        dict: 包含总数据条数、满足条件的数据数量和比例。
    """
    # 初始化统计变量
    total_count = 0
    condition_count = 0

    # 定义条件关键词
    keywords = [
        "because", " cause", "lead to", "reason", "is the reason why",
        "is the effect of", " owe to ", "give rise to",
        "bring about to", "result in"
    ]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                total_count += 1  # 总数据条数
                if any(keyword in line for keyword in keywords):  # 检查条件
                    condition_count += 1

        # 计算比例
        proportion = (condition_count / total_count) * 100 if total_count > 0 else 0

        # 返回统计结果
        return {
            "total_count": total_count,
            "condition_count": condition_count,
            "proportion": proportion
        }

    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
    except Exception as e:
        print(f"读取文件时发生错误：{e}")

if __name__ == "__main__":
    # 文件路径
    file_path = "captions.txt"  # 修改为实际的文件路径

    # 处理文件并输出结果
    results = process_captions(file_path)
    if results:
        print(f"总数据条数: {results['total_count']}")
        print(f"满足条件的数据条数: {results['condition_count']}")
        print(f"满足条件的比例: {results['proportion']:.2f}%")
