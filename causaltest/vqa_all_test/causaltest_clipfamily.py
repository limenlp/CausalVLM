import os
import pandas as pd
import open_clip
import clip
import torch
from PIL import Image
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(model_name):
    if model_name == "negclip":
        model_path = '/home/zhaotian/VL/model/negclip.pth'
        if not os.path.exists(model_path):
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=model_path, quiet=False)
        model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=model_path, device=device)

    elif model_name == "clip_vit_b32":
        model, image_preprocess = clip.load("ViT-B/32", device=device, download_root='/home/zhaotian/VL/model')
    elif model_name == "clip_vit_l14":
        model, image_preprocess = clip.load("ViT-L/14", device=device, download_root='/home/zhaotian/VL/model')
    elif model_name == "causalclip":
        model_path = f'/home/haoxuan/causal/causaltest_vcr/checkpoints/epoch_14.pth'
        model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=model_path, device=device)
    elif model_name == "robustclip":
        model, _, image_preprocess = open_clip.create_model_and_transforms('hf-hub:chs20/fare2-clip')
        model = model.to(device)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model.eval(), image_preprocess

def preprocess_image(image_path, image_preprocess):
    image = Image.open(image_path).convert("RGB")
    return image_preprocess(image).unsqueeze(0).to(device)

def compute_similarity(image_tensor, texts, model, tokenizer):
    text_tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarities = (image_features @ text_features.T).squeeze(0)
    return similarities


def process_data(model_name, causal_word, model, image_preprocess, tokenizer):
    base_csv_path = f'/home/zhaotian/VL/CausalVLM/datasets/benchmarks/vqa_causal.csv'
    if causal_word == "original":
        output_base_path = f'allvqa_results/causal_{model_name}_original.csv'
    else:
        output_base_path = f'allvqa_results/causal_{model_name}_{causal_word.replace(" ", "_")}.csv'

    data = pd.read_csv(base_csv_path)
    similarity_scores_df = pd.DataFrame(columns=['image_id', 'score', 'reverse_score', 'max_id'])

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc=f"Processing {model_name} with {causal_word}"):
        try:
            image_id = row['image_id']
            sentence = row['sentence']
            reverse_sentence = row['reverse_sentence']


            if causal_word != "original":
                sentence = sentence.replace("is due to", causal_word).replace("is caused by", causal_word)
                reverse_sentence = reverse_sentence.replace("is due to", causal_word).replace("is caused by", causal_word)

            image_path = os.path.join('/home/shared/COCO/Image/val2014/', f"COCO_val2014_{str(image_id).zfill(12)}.jpg")

            if not os.path.exists(image_path):
                print(f"Image {image_path} not found. Skipping...")
                continue

            image_tensor = preprocess_image(image_path, image_preprocess)
            similarities = compute_similarity(image_tensor, [sentence, reverse_sentence], model, tokenizer)

            score = similarities[0].item()
            reverse_score = similarities[1].item()
            max_id = 0 if score >= reverse_score else 1

            new_row = pd.DataFrame(
                [{'image_id': image_id, 'score': score, 'reverse_score': reverse_score, 'max_id': max_id}])
            similarity_scores_df = pd.concat([similarity_scores_df, new_row], ignore_index=True)

        except Exception as e:
            print(f"Error processing entry with image_id {image_id}: {e}")
            continue

    similarity_scores_df.to_csv(output_base_path, index=False)
    print(f"Results saved to {output_base_path}")


    data2 = pd.read_csv(output_base_path)
    max_id_counts = data2['max_id'].value_counts()
    total_count = len(data2)
    proportions = max_id_counts / total_count


    print(f"\nStatistics for {model_name} with {causal_word}:")
    print("总数据条数:", total_count)
    print("\nmax_id的值分布:")
    print(max_id_counts)
    print("\nmax_id的值比例:")
    print(proportions)

# 主程序
def main():
    models = ["clip_vit_b32", "clip_vit_l14", "negclip", "causalclip", "robustclip"]
    # models = ["causalclip"]
    causal_words = [
        "is due to", "is caused by", "is a result of", "is the effect of",
        "is the consequence of", "because", "owe to", "result in", "cause",
        "lead to", "give rise to", "bring about to"
    ]

    for model_name in models:
        model, image_preprocess = load_model(model_name)
        tokenizer = open_clip.tokenize if model_name in ["negclip", "causalclip", "robustclip"] else clip.tokenize
        for causal_word in causal_words:
            process_data(model_name, causal_word, model, image_preprocess, tokenizer)

if __name__ == "__main__":
    main()
