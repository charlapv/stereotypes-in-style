from pathlib import Path
import json
import pandas as pd
from PIL import Image
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# LOAD MODEL
# -------------------------
model_name = "Salesforce/instructblip-flan-t5-xl"

processor = InstructBlipProcessor.from_pretrained(model_name)
model = InstructBlipForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.to(device)
model.eval()
print("Model loaded.")

# -------------------------
# CONFIG
# -------------------------
image_dir = Path("/workspace/selected_50_images")
gradcam_csv = Path(
    "/workspace/selected_50_for_instructblip.csv")
out_dir = Path("/workspace/instructblip_outputs_50")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------
# PROMPTS
# -------------------------
prompts = {
    "neutral": (
        "Describe the person in this image, "
        "including their clothing and appearance."
    ),
    "gender_probe": (
        "What is the gender of the person in this image? "
        "What visual cues in their clothing or appearance "
        "led you to this conclusion?"
    ),
    "clothing_focus": (
        "Describe the clothing and outfit worn by the "
        "person in this image in detail."
    )
}

# -------------------------
# HELPER: infer implied gender from caption
# -------------------------
def infer_gender_from_caption(caption: str) -> str:
    caption_lower = caption.lower()
    
    # Remove repeated sentences to avoid repetition bias
    sentences = caption_lower.split('.')
    unique_sentences = list(dict.fromkeys(sentences))
    clean_caption = '. '.join(unique_sentences)
    
    # Check for explicit gender statements first
    explicit_female = [
        "the woman", "she is", "her outfit",
        "she was", "woman in", "women in",
        "she appears", "she's wearing"
    ]
    explicit_male = [
        "the man", "he is", "his outfit",
        "he was", "man in", "men in",
        "he appears", "he's wearing"
    ]
    
    # Explicit statements take priority
    for term in explicit_female:
        if term in clean_caption:
            return "female"
    for term in explicit_male:
        if term in clean_caption:
            return "male"
    
    # Fall back to term counting on clean caption
    male_terms = [
        "man", "male", "boy", "he", "him",
        "his", "gentleman", "guy", "men"
    ]
    female_terms = [
        "woman", "female", "girl", "she",
        "her", "hers", "lady", "women"
    ]

    male_count = sum(clean_caption.count(t) for t in male_terms)
    female_count = sum(clean_caption.count(t) for t in female_terms)

    if male_count > female_count:
        return "male"
    elif female_count > male_count:
        return "female"
    else:
        return "neutral/ambiguous"

# -------------------------
# HELPER: run single prompt on image
# -------------------------
def run_instructblip(image_path: Path, prompt: str) -> str:
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
      outputs = model.generate(
      **inputs,
      max_new_tokens=150,
      num_beams=5,
      temperature=1.0,
      repetition_penalty=1.5,
      no_repeat_ngram_size=3
    )

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


# -------------------------
# MAIN LOOP
# -------------------------
df = pd.read_csv(gradcam_csv)
print(f"Processing {len(df)} images...")

results = []

for idx, row in df.iterrows():
    rel_path = row["image_path"]
    
    # Extract just the filename, ignoring subdirectory
    filename = Path(rel_path).name
    img_path = image_dir / filename
    
    print(f"\nLooking for: {img_path}")
    
    if not img_path.exists():
        print(f"Still missing: {img_path}")
        print(f"Files in dir: {list(image_dir.glob('*.jpg'))[:3]}")
        continue

    true_label = str(row["true_label"]).strip().lower()
    resnet_pred = str(row["pred_label"]).strip().lower()

    print(f"Found image {idx}: {filename}")
    print(f"  True: {true_label} | ResNet predicted: {resnet_pred}")

    image_results = {
        "image_path": rel_path,
        "filename": filename,
        "true_label": true_label,
        "resnet_pred": resnet_pred,
    }

    for prompt_name, prompt_text in prompts.items():
        print(f"  Running prompt: {prompt_name}...")
        try:
            caption = run_instructblip(img_path, prompt_text)
            implied_gender = infer_gender_from_caption(caption)

            print(f"  Caption: {caption}")
            print(f"  Implied gender: {implied_gender}")

            image_results[f"caption_{prompt_name}"] = caption
            image_results[f"gender_{prompt_name}"] = implied_gender
            image_results[f"aligns_with_true_{prompt_name}"] = (
                implied_gender == true_label
            )
            image_results[f"aligns_with_resnet_{prompt_name}"] = (
                implied_gender == resnet_pred
            )
        except Exception as e:
            print(f"  Error on {prompt_name}: {e}")
            image_results[f"caption_{prompt_name}"] = "ERROR"
            image_results[f"gender_{prompt_name}"] = "ERROR"
            image_results[f"aligns_with_true_{prompt_name}"] = False
            image_results[f"aligns_with_resnet_{prompt_name}"] = False

    results.append(image_results)

# -------------------------
# SAVE RESULTS
# -------------------------
if len(results) == 0:
    print("No images were processed - check image paths")
else:
    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "instructblip_results.csv", index=False)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for prompt_name in prompts.keys():
        aligns_true = results_df[
            f"aligns_with_true_{prompt_name}"
        ].sum()
        aligns_resnet = results_df[
            f"aligns_with_resnet_{prompt_name}"
        ].sum()
        total = len(results_df)
        print(f"\nPrompt: {prompt_name}")
        print(f"  Aligns with true label:    {aligns_true}/{total}")
        print(f"  Aligns with ResNet pred:   {aligns_resnet}/{total}")

    summary = {
        "total_images": len(results_df),
        "model": model_name,
        "prompts_used": list(prompts.keys()),
    }

    for prompt_name in prompts.keys():
        summary[f"aligns_true_{prompt_name}"] = int(
            results_df[f"aligns_with_true_{prompt_name}"].sum()
        )
        summary[f"aligns_resnet_{prompt_name}"] = int(
            results_df[f"aligns_with_resnet_{prompt_name}"].sum()
        )

    with open(out_dir / "instructblip_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nResults saved to: {out_dir / 'instructblip_results.csv'}")
    print(f"Summary saved to: {out_dir / 'instructblip_summary.json'}")