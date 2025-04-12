import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaForPreTraining, BertTokenizer, FlavaFeatureExtractor

csv_path = '../multichoice/right_choice/multichoice_rightchoice_flava.csv'
image_dir = '/home/shared/COCO/Image/val2014/'

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
    text_tokens = tokenizer(text=texts, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(device)
    with torch.no_grad():
        image_features = model.flava.get_image_features(**image_tensor).cpu().numpy()[:, 0, :]
        text_features = model.flava.get_text_features(**text_tokens).cpu().numpy()[:, 0, :]

    image_features /= np.linalg.norm(image_features, axis=-1, keepdims=True)
    text_features /= np.linalg.norm(text_features, axis=-1, keepdims=True)

    similarities = (image_features @ text_features.T).squeeze(0)
    return similarities

causal_words = [
    "is due to", "is caused by", "is a result of", "is the effect of",
    "is the consequence of", "because", "owe to", "result in", "cause",
    "lead to", "give rise to", "bring about to"
]

column_names = ['image_id', 'sentence', 'reverse_sentence']
data = pd.read_csv(csv_path, names=column_names)

for causal_word in causal_words:
    print(f"\n===== Processing causal word: {causal_word} =====")
    similarity_scores_df = pd.DataFrame(columns=['image_id', 'score', 'reverse_score', 'max_id'])

    for index, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {causal_word}"):
        try:
            image_id = row['image_id']
            sentence = row['sentence']
            reverse_sentence = row['reverse_sentence']

            if causal_word != "original":
                for original_causal in ["is due to", "is caused by"]:
                    sentence = sentence.replace(original_causal, causal_word)
                    reverse_sentence = reverse_sentence.replace(original_causal, causal_word)

            image_path = os.path.join(image_dir, f"COCO_val2014_{str(image_id).zfill(12)}.jpg")
            if not os.path.exists(image_path):
                print(f"[Warning] Image not found: {image_path}")
                continue

            image_tensor = preprocess_image(image_path)
            similarities = compute_similarity(image_tensor, [sentence, reverse_sentence])
            score = similarities[0].item()
            reverse_score = similarities[1].item()
            max_id = 0 if score >= reverse_score else 1

            similarity_scores_df.loc[len(similarity_scores_df)] = [image_id, score, reverse_score, max_id]
        except Exception as e:
            print(f"[Error] index {index}, image_id {row['image_id']}: {str(e)}")
            continue

    output_base_path = f'causal_vqa_flava_{causal_word.replace(" ", "_")}.csv'
    similarity_scores_df.to_csv(output_base_path, index=False)
    print(f"\nResults saved to {output_base_path}")

    try:
        data2 = pd.read_csv(output_base_path)
        max_id_counts = data2['max_id'].value_counts()
        total_count = len(data2)
        proportions = max_id_counts / total_count

        print(f"Statistics for FLAVA with causal word '{causal_word}':")
        print("总数据条数:", total_count)
        print("max_id 的值分布:")
        print(max_id_counts)
        print("max_id 的值比例:")
        print(proportions)
    except Exception as e:
        print(f"[Warning] Failed to read or compute stats for {output_base_path}: {str(e)}")
