import argparse
import os
# os.environ["USE_FLASH_ATTENTION"] = "0" 

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

import swanlab
from swanlab.integration.huggingface import SwanLabCallback
# swanlab_callback = SwanLabCallback(project="zz-visualization")
# swanlab.init(
#   logdir='./swanlog',
#   mode="local",
# )


def main():
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name_or_path', default="/data/renyi/Model/gte-base", type=str)
        parser.add_argument('--train_data', type=str, required=True)
        parser.add_argument('--output_dir', type=str, required=True)
        parser.add_argument('--save_only_model', type=bool, default=False)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        parser.add_argument('--lr_scheduler_type', type=str, default="cosine")
        parser.add_argument('--fp16', action='store_true')
        parser.add_argument('--num_train_epochs', type=int, default=20)
        parser.add_argument('--per_device_train_batch_size', type=int, default=512)
        parser.add_argument('--mini_batch_size', type=int, default=128)
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
        parser.add_argument('--warmup_ratio', type=float, default=0.05)
        parser.add_argument('--gradient_checkpointing', action='store_true')
        parser.add_argument('--logging_steps', type=int, default=1)
        parser.add_argument('--save_strategy', type=str, default="epoch")
        parser.add_argument('--save_steps', type=int, default=500)
        parser.add_argument('--temperature', type=float, default=0.02)
        parser.add_argument('--max_seq_length', type=int, default=512)
        parser.add_argument('--save_total_limit', type=int, default=3)
        parser.add_argument('--prompts', type=str, default="")
        
        return parser.parse_args()

    
    args = get_args()
    
    print("args: ", args)
    
    # local_rank = int(os.environ["LOCAL_RANK"])
    model = SentenceTransformer(
        args.model_name_or_path,
        trust_remote_code=True,
        prompts={
            "retrieve": args.prompts,
        },
        # device=f"cuda:{local_rank}",
    )
    model.max_seq_length = args.max_seq_length
    
    dataset = load_dataset("json", data_files=args.train_data)
    train_dataset = dataset["train"]
    print(f"Dataset size: {len(train_dataset)}")
    if len(train_dataset) == 0:
        raise ValueError("Training set is empty. Please check whether 'train_data' is loaded correcetly.")
    
    save_steps = len(train_dataset) // args.per_device_train_batch_size
    
    loss = MultipleNegativesRankingLoss(model)
    # loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    
    train_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        logging_steps = args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        do_train=True,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # batch_sampler=BatchSamplers.BATCH_SAMPLER,
        run_name=args.output_dir,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        save_strategy=args.save_strategy,
        save_steps = save_steps,
        save_total_limit = args.save_total_limit,
        save_only_model = args.save_only_model,
        prompts={
            "anchor": args.prompts,
        } # type: ignore
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        loss=loss,
        # callbacks=[swanlab_callback]
    )
    
    trainer.train()

if __name__ == "__main__":
    
    main()