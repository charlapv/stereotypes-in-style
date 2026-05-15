import pandas as pd

arc = pd.read_csv("/workspace/arcface_embeddings_60k.csv")
fclip = pd.read_csv("/workspace/fashionclip_embeddings_60k.csv")

# rename feature columns so they do not clash
arc_feature_cols = [c for c in arc.columns if c not in ["image_path", "gender"]]
fclip_feature_cols = [c for c in fclip.columns if c not in ["image_path", "gender"]]

arc = arc.rename(columns={c: f"arc_{c}" for c in arc_feature_cols})
fclip = fclip.rename(columns={c: f"fclip_{c}" for c in fclip_feature_cols})

# merge on image_path
fusion_df = arc.merge(
    fclip,
    on="image_path",
    how="inner",
    suffixes=("", "_fclip")
)

# keep one gender column and verify consistency
if "gender_fclip" in fusion_df.columns:
    mismatched = fusion_df[fusion_df["gender"] != fusion_df["gender_fclip"]]
    print("Mismatched labels:", len(mismatched))
    fusion_df = fusion_df.drop(columns=["gender_fclip"])

print("Fusion rows:", len(fusion_df))
print(fusion_df.head())

fusion_df.to_csv("/workspace/fusion_embeddings_60k.csv", index=False)
print("Saved: /workspace/fusion_embeddings_60k.csv")