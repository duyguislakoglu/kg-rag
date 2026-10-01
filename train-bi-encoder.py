import os
os.environ['WANDB_DISABLED'] = 'true'
import jsonlines
from sentence_transformers import InputExample, LoggingHandler, util

train_samples = []
dev_samples = []
test_samples = []
import pandas as pd
import logging
import random
import traceback
from datetime import datetime

from datasets import Dataset, load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    BinaryClassificationEvaluator,
    SequentialEvaluator,
)
from sentence_transformers.losses import ContrastiveLoss, OnlineContrastiveLoss
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

"""

with jsonlines.open('llm-project/cross-encoder/mintaka-plausibility-data-5neg.jsonl') as f:
    counter = 0
    for line in f.iter():
        # 1889678 mintaka
        # 2246387 webq
        if counter < 1511742:
            train_samples.append(InputExample(texts=[line["query"], line["pos"][0]], label=1))
            for i in range(len(line["neg"])):
            #for i in range(2):
                train_samples.append(InputExample(texts=[line["query"], line["neg"][i]], label=0))
        elif counter >= 1511742 and counter < 1700710:

            dev_samples.append(InputExample(texts=[line["query"], line["pos"][0]], label=1))
            for i in range(len(line["neg"])):
            #for i in range(2):
                dev_samples.append(InputExample(texts=[line["query"], line["neg"][i]], label=0))
        else:
            test_samples.append(InputExample(texts=[line["query"], line["pos"][0]], label=1))
            for i in range(len(line["neg"])):
            #for i in range(2):
                test_samples.append(InputExample(texts=[line["query"], line["neg"][i]], label=0))
        counter += 1

#print(train_samples)
print(len(train_samples))
#train_dataset = train_samples
#print(dev_samples)
print(len(dev_samples))
#eval_dataset = dev_samples

#print(test_samples)
print(len(test_samples))
#test_dataset = test_samples

import jsonlines
import pandas as pd
"""

# Initialize containers
train_df, eval_df, test_df = [], [], []

with jsonlines.open('llm-project/cross-encoder/webq-plausibility-data-5neg.jsonl') as f:
    counter = 0
    for line in f.iter():
        query = line["query"]
        pos = line["pos"][0]
        negs = line["neg"]

        if counter < 1797109:
            # Training set
            train_df.append({"text1": query, "text2": pos, "label": 1})
            for neg in negs:
                train_df.append({"text1": query, "text2": neg, "label": 0})

        elif counter < 2021748:
            # Dev set
            eval_df.append({"text1": query, "text2": pos, "label": 1})
            for neg in negs:
                eval_df.append({"text1": query, "text2": neg, "label": 0})

        else:
            # Test set
            test_df.append({"text1": query, "text2": pos, "label": 1})
            for neg in negs:
                test_df.append({"text1": query, "text2": neg, "label": 0})
        counter += 1

# Convert to DataFrames
train_df = pd.DataFrame(train_df)
eval_df = pd.DataFrame(eval_df)
test_df = pd.DataFrame(test_df)

# Optionally save to CSV or Parquet
train_df.to_csv("bienc-webq_train.csv",header=True, index=False)
eval_df.to_csv("bienc-webq_eval.csv", header=True, index=False)
test_df.to_csv("bienc-webq_test.csv",header=True,  index=False)

print(f"Train: {len(train_df)} rows, Dev: {len(eval_df)} rows, Test: {len(test_df)} rows")

"""
This scripts demonstrates how to train a sentence embedding model for question pair classification
with cosine-similarity and a simple threshold.

As dataset, we use Quora Duplicates Questions, where we have labeled pairs of questions being either duplicates (label 1) or non-duplicate (label 0).

As loss function, we use OnlineConstrativeLoss. It reduces the distance between positive pairs, i.e., it pulls the embeddings of positive pairs closer together. For negative pairs, it pushes them further apart.

An issue with constrative loss is, that it might push sentences away that are already well positioned in vector space.
"""


# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

# As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
#model_name = "stsb-distilbert-base"
model_name = "distilroberta-base"
model = SentenceTransformer(model_name)
num_train_epochs = 1
batch_size = 1024

output_dir = "output/webq-distilroberta-base-" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

################### Load Quora Duplicate Questions dataset ##################
# https://huggingface.co/datasets/sentence-transformers/quora-duplicates
#dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train")
#dataset = dataset.train_test_split(test_size=1000)
#train_dataset = dataset["train"].select(range(100000))
#eval_dataset = dataset["test"]

#train_df = pd.read_csv("bienc-mintaka_train.csv", header=0)
#eval_df = pd.read_csv("bienc-mintaka_eval.csv", header=0)

train_dataset = Dataset.from_pandas(train_df, split="train")
eval_dataset = Dataset.from_pandas(eval_df, split="eval")#.select(range(10000))

print(len(train_dataset))
print(len(eval_dataset))
print(eval_dataset[0])

# Negative pairs should have a distance of at least 0.5
margin = 0.5
# As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
train_loss = OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

################### Development  Evaluators ##################
# We add 3 evaluators, that evaluate the model on Duplicate Questions pair classification,
# Duplicate Questions Mining, and Duplicate Questions Information Retrieval
evaluators = []

###### Classification ######
# Given (question1, question2), is this a duplicate or not?
# The evaluator will compute the embeddings for both questions and then compute
# a cosine similarity. If the similarity is above a threshold, we have a duplicate.

binary_acc_evaluator = BinaryClassificationEvaluator(
    sentences1=eval_dataset["text1"],
    sentences2=eval_dataset["text2"],
    labels=eval_dataset["label"],
    name="quora-duplicates",
)

#binary_acc_evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples)
evaluators.append(binary_acc_evaluator)

# Create a SequentialEvaluator. This SequentialEvaluator runs all three evaluators in a sequential order.
# We optimize the model with respect to the score from the last evaluator (scores[-1])
#seq_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

logging.info("Evaluate model without training")
#seq_evaluator(model, epoch=0, steps=0)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # OCL benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    #eval_strategy="steps",
    #eval_steps=250,
    #save_strategy="steps",
    #save_steps=50,
    #save_total_limit=2,
    logging_steps=250,
    #run_name="online-contrastive-loss",  # Will be used in W&B if `wandb` is installed
)

#print(len(train_dataset))
#print(len(eval_dataset))
#train_dataset = load_dataset("csv", data_files="bienc-mintaka_train.csv")["train"]
#eval_dataset = load_dataset("csv", data_files="bienc-mintaka_eval.csv")["train"]

# Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=binary_acc_evaluator,
)
trainer.train()

# Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)

# (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
try:
    model.push_to_hub(f"webq-5neg-{model_name}-ocl", token = "", private=True)
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-ocl')`."
    )
