import argparse
import json
import os
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives for training.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input JSONL data file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the SentenceTransformer model.")
    parser.add_argument("--num_negatives", type=int, default=15, help="Number of negatives per question-answer pair.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed dataset.")
    parser.add_argument("--output_format", type=str, required=False, help="n-tuple、labeled-pair", default="n-tuple")
    parser.add_argument("--prompts", type=str, required=False, help="", default="")
    args = parser.parse_args()

    
    train_dataset = load_dataset("json", data_files=args.data_path, split="train")

    
    prompt_length = len(args.prompts + " ") if args.prompts else 0
    
    def add_prompt_to_anchor(example):
        example["anchor"] = args.prompts + " " + example["anchor"] if args.prompts else example["anchor"]
        return example
    
    train_dataset = train_dataset.map(add_prompt_to_anchor)
    
    
    embedding_model = SentenceTransformer(
        model_name_or_path=args.model_path, trust_remote_code=True
    )

    
    hard_train_dataset = mine_hard_negatives(
        train_dataset,
        embedding_model,
        num_negatives=args.num_negatives,
        range_min=1,
        range_max=200,
        # max_score=0.8,
        # margin=0,
        sampling_strategy="random",
        use_multi_process=False,
        batch_size=64,
        output_format=args.output_format,
        use_faiss=True,
    )

    
    with open(args.output_path, "w") as f:
        for data in hard_train_dataset:
            if prompt_length > 0 and "anchor" in data:
                data["anchor"] = data["anchor"][prompt_length:]
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()