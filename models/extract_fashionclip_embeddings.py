from pathlib import Path

import pandas as pd
from tqdm import tqdm
from fashion_clip.fashion_clip import FashionCLIP


def main():
    image_dir = Path("/workspace/data/subset_60000")
    labels_csv = Path("/workspace/richwear_metadata_60kclean.csv")
    out_csv = Path("/workspace/fashionclip_embeddings_60k.csv")

    df = pd.read_csv(labels_csv)

    required_cols = {"image_path", "gender"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {required_cols}. "
            f"Found: {df.columns.tolist()}"
        )

    print(f"Rows in labels CSV: {len(df)}")

    fclip = FashionCLIP("fashion-clip")

    # Keep only rows whose files exist
    valid_rows = []
    missing = 0

    for _, row in df.iterrows():
        img_path = image_dir / row["image_path"]
        if img_path.exists():
            valid_rows.append(row)
        else:
            missing += 1

    valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
    print(f"Valid image rows: {len(valid_df)}")
    print(f"Missing files: {missing}")

    batch_size = 64
    results = []
    failed = 0
    first_error = None

    for start in tqdm(range(0, len(valid_df), batch_size), desc="Extracting FashionCLIP embeddings"):
        batch_df = valid_df.iloc[start:start + batch_size]

        batch_paths = [str(image_dir / p) for p in batch_df["image_path"].tolist()]

        try:
            batch_embs = fclip.encode_images(batch_paths, batch_size=batch_size)
        except Exception as e:
            failed += len(batch_df)
            if first_error is None:
                first_error = repr(e)
            continue

        for row, emb in zip(batch_df.itertuples(index=False), batch_embs):
            record = {
                "image_path": row.image_path,
                "gender": row.gender
            }
            for i, val in enumerate(emb):
                record[f"f{i}"] = float(val)
            results.append(record)

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)

    print("\n=== FashionCLIP Extraction Summary ===")
    print(f"Rows in labels CSV : {len(df)}")
    print(f"Saved embeddings   : {len(out_df)}")
    print(f"Missing files      : {missing}")
    print(f"Failed processing  : {failed}")
    if first_error is not None:
        print(f"First error        : {first_error}")
    print(f"Saved to           : {out_csv}")

    if len(out_df) > 0:
        print("\nGender distribution in embeddings file:")
        print(out_df['gender'].value_counts())


if __name__ == "__main__":
    main()