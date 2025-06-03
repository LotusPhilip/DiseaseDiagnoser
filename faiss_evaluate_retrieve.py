import argparse
import os
import faiss
import json
from sentence_transformers import SentenceTransformer, models
import numpy as np
from tqdm import tqdm
import torch
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="BAAI/bge-base-en", type=str)
    parser.add_argument('--corpus_file', default=None, type=str)
    parser.add_argument('--eval_query_file', default=None, type=str)
    parser.add_argument('--output_file', default=None, type=str)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--top_k', default=None, type=int)
    parser.add_argument('--use_gpu_for_searching', action='store_true', help='use faiss-gpu')
    parser.add_argument('--prompts', default="", type=str)

    return parser.parse_args()

def get_corpus_dataset(corpus_file):
    """
    Load the corpus dataset from a jsonl file.
    sample:
    knowledge_base: {"disease": "...", "desc": "..."}   ****
    """
    dataset = []
    with open(corpus_file, "r") as f:
        dataset = json.load(f)

    return dataset

def create_index(embeddings, use_gpu: bool = False):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)

    return index


def get_model(model_path, device):

    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)

    return model


def batch_search(index,
                 query,
                 topk: int = 10,
                 batch_size: int = 256):
    all_scores, all_inxs = [], []
    for start_index in tqdm(range(0, len(query), batch_size), desc="Batches"):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(np.asarray(batch_query, dtype=np.float32), k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs

def judge(str1: str, str2: str):
    str1_list = list(str1)
    sign = 0
    for char in str2:
        if char in str1_list:
            str1_list.remove(char)
        else:
            sign = 1
            break
    if sign == 0:
        return 1

    str2_list = list(str2)
    sign = 0
    for char in str1:
        if char in str2_list:
            str2_list.remove(char)
        else:
            sign = 1
            break
    if sign == 0:
        return 1

def evaluate(args):
    corpu_dataset = get_corpus_dataset(args.corpus_file)
    corpus = [data['desc'] for data in corpu_dataset]

    eval_dataset = []
    """
    eval_query_file: {"query": "....", "ICD": "....", "disease": "...."}
    query_input: {"disease": "...", "pat_summary": "..."}
    """
    with open(args.eval_query_file, "r") as f:
        eval_dataset = json.load(f)
        
    queries = [args.prompts + data['pat_summary'] for data in eval_dataset]

    for file in os.listdir(args.model_name_or_path):
        if file.startswith("checkpoint"):
            model_start_time = time.time()  

            model = get_model(os.path.join(args.model_name_or_path, file), args.device)
            p_vecs = model.encode(corpus, batch_size=64, device=args.device, show_progress_bar=True, normalize_embeddings = True)
            index = create_index(p_vecs, use_gpu=args.use_gpu_for_searching)

            q_vecs = model.encode(queries, batch_size=64, device=args.device, show_progress_bar=True, normalize_embeddings = True)
            res_scores, res_index = batch_search(index, q_vecs, topk=args.top_k, batch_size=64)

            '''
            ans_accurate = 0
            for res, data in zip(res_index, eval_dataset):
                icd_list = [corpu_dataset[id]['ICD'] for id in res]
                if data['ICD'] in icd_list:
                    ans_accurate += 1
            '''


            ans_fuzzy = 0
            with open('xxxxx.jsonl', 'w', encoding='utf-8') as output_file:
                for res, data in zip(res_index, eval_dataset):
                    disease_list = [corpu_dataset[id]['disease'] for id in res]
                    json.dump(disease_list, output_file, ensure_ascii=False)
                    output_file.write('\n')
                

                    for dis in disease_list:
                        if judge(data['disease'], dis):
                            ans_fuzzy += 1
                            break

            model_elapsed_time = time.time() - model_start_time  
            with open(args.output_file, "a") as f:
                f.write(json.dumps({"model": file,
                                    #"accuracy": ans_accurate/len(eval_dataset),
                                    "fuzzy_accuracy": ans_fuzzy/len(eval_dataset),
                                    "evaluation_time": model_elapsed_time
                                    }) + "\n")

            
            del model
            torch.cuda.empty_cache()


if __name__ == '__main__':
    args = get_args()
    evaluate(args)


