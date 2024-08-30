import argparse
import os

import torch
import numpy as np

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    default_data_collator
)

from utils import write_tensor_to_file

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def process_qa_dataset(model_name_or_path, dataset, data_dir):
    max_seq_length = 128
    doc_stride = 0

    model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    raw_datasets = load_dataset(dataset)

    column_names = raw_datasets["train"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if "validation" not in raw_datasets:
        raise ValueError("--do_eval requires a validation dataset")
    eval_examples = raw_datasets["validation"]
    # Validation Feature Creation
    eval_dataset = eval_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset",
    )

    eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])
    eval_dataloader = DataLoader(
        eval_dataset_for_model, collate_fn=default_data_collator, batch_size=1
    )

    model.eval()

    print(model.__class__.__name__)
    if model.__class__.__name__ == "MobileBertForQuestionAnswering":
        embeddings = model.mobilebert.embeddings
    elif model.__class__.__name__ == "BertForQuestionAnswering":
        embeddings = model.bert.embeddings

    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.no_grad():
            embedding_output = embeddings(
                input_ids=batch["input_ids"],
                token_type_ids=batch["token_type_ids"]
            )
        embedding_output = embedding_output.detach().numpy().astype(np.float64)
        attention_mask = (1.0 - batch["attention_mask"]).cpu().numpy() * np.finfo(np.float32).min

        folder = os.path.join(data_dir, str(step))
        os.makedirs(folder, exist_ok=True)
        write_tensor_to_file(os.path.join(folder, "embedding_output"), embedding_output)
        write_tensor_to_file(os.path.join(folder, "attention_mask"), attention_mask)


def process_glue_dataset(model_name_or_path, task_name, data_dir, finetuning=False):
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    raw_datasets = load_dataset("glue", task_name)

    sentence1_key, sentence2_key = task_to_keys[task_name]

    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
        result["labels"] = examples["label"]
        return result

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    if finetuning:
        sets = ["train", "validation"]
    else:
        sets = ["validation"]

    for set_name in sets:
        dataset = tokenized_datasets[set_name]
        dataloader = DataLoader(dataset, collate_fn=default_data_collator, batch_size=1)

        model.eval()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        for idx, batch in enumerate(tqdm(dataloader, desc=f"Running through {set_name} dataset")):
            inputs = {k: v.to(device) for k, v in batch.items()}
            embedding_output = model.mobilebert.embeddings(
                input_ids=inputs["input_ids"],
                token_type_ids=inputs["token_type_ids"]
            )
            embedding_output = embedding_output.detach().numpy().astype(np.float64)
            attention_mask = (1.0 - inputs["attention_mask"]).cpu().numpy() * np.finfo(np.float32).min
            labels = inputs["labels"].detach().numpy().astype(np.float64)

            if finetuning:
                folder = os.path.join(data_dir, set_name, f"{idx}_{int(labels[0])}")
            else:
                folder = os.path.join(data_dir, f"{idx}_{int(labels[0])}")
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            # write embedding output and attention mask
            write_tensor_to_file(os.path.join(folder, "arg0_1.bin"), embedding_output)
            write_tensor_to_file(os.path.join(folder, "arg1_1.bin"), attention_mask)


def main():
    parser = argparse.ArgumentParser(description="Generate datafiles from model.")
    parser.add_argument("--model_name_or_path", help="Path to model directory.", default=None, required=True)
    parser.add_argument("--dataset", help="Dataset to dump.", default="sst2")
    parser.add_argument("--output_dir", help="Path to output datafiles.", required=True)
    parser.add_argument("--finetuning", action="store_true", help="Dump dataset for finetuning.")
    args = parser.parse_args()

    if args.dataset == "squad":
        process_qa_dataset(args.model_name_or_path, args.dataset, args.output_dir)
    else:
        process_glue_dataset(args.model_name_or_path, args.dataset, args.output_dir, args.finetuning)


if __name__ == "__main__":
    main()
