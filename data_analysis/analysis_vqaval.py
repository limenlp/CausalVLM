import json

# 文件路径
file_path = '/home/zhaotian/VL/data/vqa_coco/v2_OpenEnded_mscoco_val2014_questions.json'

try:
    # 加载JSON文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 初始化统计变量
    total_questions = len(data['questions'])
    why_questions_count = sum(1 for question in data['questions'] if 'why' in question['question'].lower())

    # 计算比例
    why_question_ratio = (why_questions_count / total_questions) * 100 if total_questions > 0 else 0

    # 输出结果
    print(f"总问题数量: {total_questions}")
    print(f"包含 'why' 的问题数量: {why_questions_count}")
    print(f"包含 'why' 的问题比例: {why_question_ratio:.2f}%")

except FileNotFoundError:
    print(f"文件未找到，请检查路径是否正确: {file_path}")
except json.JSONDecodeError:
    print("JSON 文件解析失败，请检查文件内容是否正确。")
except Exception as e:
    print(f"发生错误: {e}")
