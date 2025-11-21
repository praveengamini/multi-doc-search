from sklearn.datasets import fetch_20newsgroups
import os

def download_dataset():
    dataset = fetch_20newsgroups(subset='train')

    output_dir = "data/docs"
    os.makedirs(output_dir, exist_ok=True)

    # Save only first 200 docs
    for i, text in enumerate(dataset.data[:200]):
        file_path = os.path.join(output_dir, f"doc_{i:03d}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

    print("Saved 200 documents to data/docs/")


if __name__ == "__main__":
    download_dataset()
