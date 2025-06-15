import json

def extract_captions(json_file_path, output_file_path):
    """
    读取 COCO 数据集的 JSON 文件并提取所有 caption 写入到一个文件中。

    Args:
        json_file_path (str): COCO 的 JSON 文件路径。
        output_file_path (str): 输出文件路径，每行保存一个 caption。
    """
    try:
        # 打开并加载 JSON 文件
        with open(json_file_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        # 提取 annotations 中的 caption 字段
        captions = [annotation["caption"] for annotation in coco_data["annotations"]]

        # 将所有 caption 写入输出文件
        with open(output_file_path, "w", encoding="utf-8") as f:
            for caption in captions:
                f.write(caption + "\n")

        print(f"共提取 {len(captions)} 条 caption，并保存到文件：{output_file_path}")

    except FileNotFoundError:
        print(f"文件未找到：{json_file_path}")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")

if __name__ == "__main__":
    # 输入 JSON 文件路径
    json_file_path = "/home/shared/COCO/Image/annotations/captions_train2014.json"

    # 输出 caption 文件路径
    output_file_path = "coco_captions.txt"

    # 提取并保存 caption
    extract_captions(json_file_path, output_file_path)
