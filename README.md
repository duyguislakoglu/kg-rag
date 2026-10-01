# 1. KR2
# Datasets
You need to download WebQSP and Mintaka from the following link and place it to ../data:
[Dataset Drive](https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvcyFBajVKZXJWOFNNRHlpLXRKNVFPbGxsX2tzZjFmUkE%5FZT02SE9tSTI&id=F2C0487CB57A493E%21193993&cid=F2C0487CB57A493E)
(This link is provided by [KALMV](https://github.com/JinheonBaek/KALMV))

# The results
The results for the experiments can be seen in results section.

# Requirements
```sh
pip install -r requirements.txt
```

# Experiment Commands
 <u>Note that first you need to provide your HuggingFace token in the following files:</u>

models/language_models.py
  token = ""

models/retrievers.py
  login(token="")

## Experiment 1
```sh
python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1

python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1 --instanceof
python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1 --attach_instanceof
python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1 --instanceof --attach_instanceof

python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1 --subclassof
python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1 --attach_subclassof
python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1 --subclassof --attach_subclassof
```

### Llama-3.2-3B-Instruct
```sh
python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla model_name --meta-llama/Llama-3.2-3B-Instruct --decoder-only --batch_size 1
```

## Experiment 2
For this experiment, you need to download the trained models from [here](https://drive.google.com/drive/u/1/folders/13PavxkWErZeAPXfsZa9_jRZovUsDtlM4)
and place them into ./cross-encoder/finetuned-sbert-models/ folder.

The models are trained on mintaka-plausibility-data-5neg.jsonl and webq-plausibility-data-5neg.jsonl. These json files are created
by the triples that can be found mintaka and webq folders. The training scripts for the models are train-cross-encoder.py and train-bi-encoder.py.

```sh
python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1

python main.py --data_name Mintaka --retriever_name pretrained-sbert-cross-encoder --data_type KGQA --knowledge_base kg --exp_name pretrained --model_name bigscience/T0_3B --batch_size 1

python main.py --data_name Mintaka --retriever_name finetuned-sbert-cross-encoder --data_type KGQA --knowledge_base kg --exp_name sbertc_finetuned_5_neg_nooverlap --model_name bigscience/T0_3B --batch_size 1
```

### Multi-hop
```sh
python main.py --data_name Mintaka --data_type KGQA --knowledge_base kg --exp_name vanilla --model_name bigscience/T0_3B --batch_size 1 --multihop
```

### Bi-encoder
```sh
python main.py --data_name Mintaka --data_type KGQA —retriever_name bi-encoder-mintaka --knowledge_base kg --exp_name vanilla-biencoder --model_name bigscience/T0_3B --batch_size 1 --edit_output False
```

# Acknowledgement
This project is built on top of [KALMV](https://github.com/JinheonBaek/KALMV) repository.

The trained models and training datasets are available at
https://drive.google.com/drive/folders/13PavxkWErZeAPXfsZa9_jRZovUsDtlM4?

# 2. Temporal KGQA
# Datasets
The datasets are taken from: github.com/qianxinying/TimeR4/tree/main/datasets

# The results
The results for the experiments can be seen in results section.

# Experiment Commands
Note that first you need to provide your HuggingFace token in the following files:

models/language_models.py token = ""

models/retrievers.py login(token="")

The commands can be found in file main_table.sh.
