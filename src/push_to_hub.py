from datasets import load_dataset
from argparse import ArgumentParser


def main():
    """
    Main function to push dataset to Hugging Face Hub.
    """

    parser = ArgumentParser(description="Push dataset to Hugging Face Hub")
    parser.add_argument("filepath", help="Path to the JSONL file")
    parser.add_argument(
        "repo_name", help="Name of the Hugging Face Hub repository"
    )
    args = parser.parse_args()

    filepath = args.filepath
    repo_name = args.repo_name

    ls_dataset = load_dataset(
        "json",
        data_files={"train": filepath},
        split="train",
    )

    ls_dataset.push_to_hub(repo_name, private=True)


if __name__ == "__main__":
    main()
