import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaForPreTraining, BertTokenizer, FlavaFeatureExtractor



csv_path = '/home/zhaotian/VL/CausalVLM/datasets/multichoice/multichoice.csv'


image_dir = '/home/shared/COCO/Image/val2014/'
output_csv_path = '/home/zhaotian/VL/all_final_data/multichoice/multichoice/multichoice_flava_1947.csv'    #clip base模型在【在第二次修改choice2的基础上修改了choice3的文件】上进行ABCD多选


device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = FlavaForPreTraining.from_pretrained("facebook/flava-full", cache_dir="../../model/").eval().to(device)
feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full", cache_dir="../../model/")
tokenizer = BertTokenizer.from_pretrained("facebook/flava-full", cache_dir="../../model/")
processor = FlavaProcessor.from_pretrained("facebook/flava-full", cache_dir="../../model/")


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)
    return inputs


def compute_similarity(image_tensor, texts):
    text_tokens = tokenizer(text=texts, return_tensors="pt", padding="max_length", max_length=77).to(device)
    with torch.no_grad():
        image_features = model.flava.get_image_features(**image_tensor).cpu().numpy()[:, 0, :]
        text_features = model.flava.get_text_features(**text_tokens).cpu().numpy()[:, 0, :]


    image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)

    similarities = (image_features @ text_features.T).squeeze(0)
    return similarities


column_names = ['image_id', 'choice0', 'choice1', 'choice2', 'choice3']
data = pd.read_csv(csv_path,names = column_names)


similarity_scores_df = pd.DataFrame(columns=['image_id', 'score0', 'score1', 'score2', 'score3'])


batch_size = 10
counter = 0

for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Processing entries"):
    try:
        image_id = row['image_id']
        choice0 = row['choice0']
        choice1 = row['choice1']
        choice2 = row['choice2']
        choice3 = row['choice3']
        image_path = os.path.join(image_dir, f"COCO_val2014_{str(image_id).zfill(12)}.jpg")

        if not os.path.exists(image_path):
            print(f"Image {image_path} not found. Skipping...")
            continue


        image_tensor = preprocess_image(image_path)


        similarities = compute_similarity(image_tensor, [choice0,choice1,choice2,choice3])
        score0 = similarities[0].item()
        score1 = similarities[1].item()
        score2 = similarities[2].item()
        score3 = similarities[3].item()


        new_row = pd.DataFrame(
            [{'image_id': image_id, 'score0': score0, 'score1': score1, 'score2': score2, 'score3': score3}])
        similarity_scores_df = pd.concat([similarity_scores_df, new_row], ignore_index=True)

        counter += 1


        if counter % batch_size == 0:
            similarity_scores_df.to_csv(output_csv_path, index=False, mode='a',
                                        header=not os.path.exists(output_csv_path))
            similarity_scores_df = pd.DataFrame(columns=['image_id', 'score0', 'score1', 'score2', 'score3'])  # 清空DataFrame
    except Exception as e:
        print(f"Error processing entry with image_id {image_id}: {e}")
        continue


if not similarity_scores_df.empty:
    similarity_scores_df.to_csv(output_csv_path, index=False, mode='a', header=False)

print(f"Similarity scores generated and saved to '{output_csv_path}'.")




sim_df = pd.read_csv(output_csv_path)
sim_df['image_id'] = sim_df['image_id'].astype(int)


csv_path1 = '/home/zhaotian/VL/script/vqa_coco/script/CausalTest/vqa_causal.csv'
data1_cols = ['image_id', 'sentence', 'reverse_sentence']
causal_df = pd.read_csv(csv_path1, names=data1_cols)
causal_df['image_id'] = causal_df['image_id'].astype(int)


filtered_ids = sim_df[
    (sim_df['score0'] > sim_df['score2']) &
    (sim_df['score0'] > sim_df['score3']) &
    (sim_df['score1'] > sim_df['score3']) &
    (sim_df['score1'] > sim_df['score2'])
]['image_id']


filtered_count = len(filtered_ids)
total_count = len(sim_df)
print(f"Filtered data count: {filtered_count}")
print(f"Total data count:    {total_count}")
print(f"Filtered proportion: {filtered_count/total_count:.2%}")


filtered_data = causal_df[causal_df['image_id'].isin(filtered_ids)]
filtered_output = output_csv_path.replace('.csv', '_rightchoice.csv')
filtered_data.to_csv(filtered_output, index=False)
print(f"Filtered causal data saved to: {filtered_output}")
