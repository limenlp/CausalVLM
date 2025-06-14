import os
import pandas as pd
import torch
# from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from PIL import Image


# model_name = "llava-hf/llava-1.5-7b-hf"
# model = LlavaForConditionalGeneration.from_pretrained(model_name)
# processor = AutoProcessor.from_pretrained(model_name)
model_name = "lmsys/vicuna-7b-v1.5"  # 使用你指定的模型路径
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


csv_path = '/home/zhaotian/VL/CausalVLM/datasets/benchmarks/vqa_causal.csv'
image_dir = '/home/shared/COCO/Image/val2014/'


data = pd.read_csv(csv_path)


causal_words = [
    "is due to", "is caused by", "is a result of", "is the effect of",
    "is the consequence of", "because", "owe to", "result in", "cause",
    "lead to", "give rise to", "bring about to"
]



Yes_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("Yes"))
No_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("No"))

def get_yes_no_probabilities(question):

    inputs = tokenizer(question, return_tensors="pt")


    with torch.no_grad():
        outputs = model(**inputs)


    logits = outputs.logits



    softmax = torch.nn.functional.softmax(logits[0, -1], dim=-1)

    # 获取 "yes"、"Yes"、"no" 和 "No" 的概率
    yes_prob = softmax[Yes_token_id].item()
    no_prob = softmax[No_token_id].item()


    max_prob = torch.max(softmax).item()

    return yes_prob, no_prob


def get_yes_no_probabilities_old(prompt, image):
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=3, output_scores=True, return_dict_in_generate=True)
    logits = outputs.scores[0]  # 获取第一个 token 的 logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    yes_prob = probabilities[:, Yes_token_id].item()
    no_prob = probabilities[:, No_token_id].item()
    return yes_prob, no_prob


batch_size = 5


for causal_word in causal_words:
    if causal_word == "original":
        output_csv_path = "/home/zhaotian/VL/all_final_data/all_vqa_results/vicuna_original.csv"
    else:
        output_csv_path = f"/home/zhaotian/VL/all_final_data/all_vqa_results/vicuna_{causal_word.replace(' ', '_')}.csv"

    similarity_scores_df = pd.DataFrame(
        columns=['image_id', 'score_yes', 'score_no', 'score_yes_reverse', 'score_no_reverse'])
    counter = 0

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc=f"Processing {model_name} with {causal_word}"):
        try:
            image_id = row['image_id']
            sentence = row['sentence']
            reverse_sentence = row['reverse_sentence']

            if causal_word != "original":
                sentence = sentence.replace("is due to", causal_word).replace("is caused by", causal_word)
                reverse_sentence = reverse_sentence.replace("is due to", causal_word).replace("is caused by",
                                                                                              causal_word)

            
            prompt_sentence = f"USER: {sentence} does it reflect the proper causal relationship? ASSISTANT:"
            prompt_reverse_sentence = f"USER: {reverse_sentence} does it reflect the proper causal relationship? ASSISTANT:"

            score_yes, score_no = get_yes_no_probabilities(prompt_sentence)
            score_yes_reverse, score_no_reverse = get_yes_no_probabilities(prompt_reverse_sentence)

            new_row = {
                'image_id': image_id,
                'score_yes': score_yes,
                'score_no': score_no,
                'score_yes_reverse': score_yes_reverse,
                'score_no_reverse': score_no_reverse
            }
            similarity_scores_df = pd.concat([similarity_scores_df, pd.DataFrame([new_row])], ignore_index=True)

            counter += 1
            if counter % batch_size == 0:
                similarity_scores_df.to_csv(output_csv_path, index=False, mode='a',
                                            header=not os.path.exists(output_csv_path))
                similarity_scores_df = pd.DataFrame(
                    columns=['image_id', 'score_yes', 'score_no', 'score_yes_reverse', 'score_no_reverse'])
        except Exception as e:
            print(f"Error processing entry with image_id {image_id}: {e}")
            continue

    if not similarity_scores_df.empty:
        similarity_scores_df.to_csv(output_csv_path, index=False, mode='a', header=not os.path.exists(output_csv_path))

    final_data = pd.read_csv(output_csv_path)
    total_entries = final_data.shape[0]
    condition_met = final_data[((final_data['score_yes'] > final_data['score_no']) & (
                final_data['score_yes'] > final_data['score_yes_reverse'])) |
                               ((final_data['score_yes'] < final_data['score_no']) & (
                                           final_data['score_no'] < final_data['score_no_reverse']))]

    count_met = condition_met.shape[0]
    percentage_met = (count_met / total_entries) * 100 if total_entries > 0 else 0

    print(f"{causal_word}: Total entries: {total_entries}")
    print(f"{causal_word}: Entries meeting condition: {count_met}")
    print(f"{causal_word}: Percentage meeting condition: {percentage_met:.2f}%")
    print(f"Scores generated and saved to '{output_csv_path}'.\n")
