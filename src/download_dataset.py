from datasets import load_dataset

from src.config import RAW_DATA_DIR


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("Tobi-Bueck/customer-support-tickets")
    df = dataset["train"].to_pandas()

    output_path = RAW_DATA_DIR / "customer_support_tickets.csv"
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print("Columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
