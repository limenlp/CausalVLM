import os
import pandas as pd
import open_clip
import clip
import torch
from PIL import Image
from tqdm import tqdm


csv_path = '/home/zhaotian/VL/CausalVLM/datasets/multichoice/multichoice.csv'
column_names = ['image_id', 'choice0', 'choice1', 'choice2', 'choice3']
data = pd.read_csv(csv_path, names=column_names)
image_dir = '/home/shared/COCO/Image/val2014/'
output_dir = '/home/zhaotian/VL/all_final_data/multichoice/multichoice'
os.makedirs(output_dir, exist_ok=True)

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

def process_model(model_name, model, image_preprocess, tokenizer):
    output_csv_path = os.path.join(output_dir, f"multichoice_similarity_{model_name}.csv")
    similarity_scores_df = pd.DataFrame(columns=['image_id', 'score0', 'score1', 'score2', 'score3'])

    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc=f"Processing {model_name}"):
        try:
            image_id = row['image_id']
            choice0, choice1, choice2, choice3 = row['choice0'], row['choice1'], row['choice2'], row['choice3']
            image_path = os.path.join(image_dir, f"COCO_val2014_{str(image_id).zfill(12)}.jpg")

            if not os.path.exists(image_path):
                print(f"Image {image_path} not found. Skipping...")
                continue

            image_tensor = preprocess_image(image_path, image_preprocess)
            similarities = compute_similarity(image_tensor, [choice0, choice1, choice2, choice3], model, tokenizer)

            new_row = pd.DataFrame([
                {
                    'image_id': image_id,
                    'score0': similarities[0].item(),
                    'score1': similarities[1].item(),
                    'score2': similarities[2].item(),
                    'score3': similarities[3].item()
                }
            ])
            similarity_scores_df = pd.concat([similarity_scores_df, new_row], ignore_index=True)

        except Exception as e:
            print(f"Error processing entry with image_id {image_id}: {e}")
            continue

    similarity_scores_df.to_csv(output_csv_path, index=False)
    print(f"Similarity scores saved to {output_csv_path}")


    csv_path1 = '/home/zhaotian/VL/script/vqa_coco/script/CausalTest/vqa_causal.csv'
    similarity_scores_df = pd.read_csv(output_csv_path)
    similarity_scores_df['image_id'] = similarity_scores_df['image_id'].astype(int)

    data1_columns = ['image_id', 'sentence', 'reverse_sentence']
    data1 = pd.read_csv(csv_path1, names=data1_columns)
    data1['image_id'] = data1['image_id'].astype(int)


    filtered_ids = similarity_scores_df[(similarity_scores_df['score0'] > similarity_scores_df['score2']) &
                                        (similarity_scores_df['score0'] > similarity_scores_df['score3']) &
                                        (similarity_scores_df['score1'] > similarity_scores_df['score3']) &
                                        (similarity_scores_df['score1'] > similarity_scores_df['score2'])]['image_id']


    filtered_count = len(filtered_ids)
    total_count = len(similarity_scores_df)
    print(f"Filtered data count: {filtered_count}")
    print(total_count)
    print(f"Filtered data proportion: {filtered_count / total_count:.2%}")


    filtered_data = data1[data1['image_id'].isin(filtered_ids)]


    filtered_output_csv_path = os.path.join(output_dir, f"multichoice_rightchoice_{model_name}.csv")
    filtered_data.to_csv(filtered_output_csv_path, index=False)
    print(f"Filtered data for model {model_name} saved to {filtered_output_csv_path}.")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# NegCLIP
negclip_model_path = '/home/zhaotian/VL/model/negclip.pth'
if not os.path.exists(negclip_model_path):
    import gdown
    gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=negclip_model_path, quiet=False)
negclip_model, _, negclip_image_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained=negclip_model_path, device=device)
negclip_model = negclip_model.eval()
process_model("negclip", negclip_model, negclip_image_preprocess, open_clip.tokenize)

# CausalCLIP
causalclip_model_path = f'/home/haoxuan/causal/causaltest_vcr/checkpoints/epoch_14.pth'
causalclip_model, _, causalclip_image_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained=causalclip_model_path, device=device)
causalclip_model = causalclip_model.eval()
process_model("causalclip", causalclip_model, causalclip_image_preprocess, open_clip.tokenize)

# CLIP (ViT-B/32)
clip_b_model, clip_b_image_preprocess = clip.load("ViT-B/32", device=device, download_root='/home/zhaotian/VL/model')
clip_b_model = clip_b_model.eval()
process_model("clip_vit_b32", clip_b_model, clip_b_image_preprocess, clip.tokenize)

# CLIP (ViT-L/14)
clip_l_model, clip_l_image_preprocess = clip.load("ViT-L/14", device=device, download_root='/home/zhaotian/VL/model')
clip_l_model = clip_l_model.eval()
process_model("clip_vit_l14", clip_l_model, clip_l_image_preprocess, clip.tokenize)



# RobustCLIP
robustclip_model, _, robustclip_image_preprocess = open_clip.create_model_and_transforms(
    'hf-hub:chs20/fare2-clip')
robustclip_model = robustclip_model.to(device)
robustclip_model = robustclip_model.eval()
process_model("robustclip", robustclip_model, robustclip_image_preprocess, open_clip.tokenize)
