from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class GlobalArguments:
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for initialization"}
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Avoid using CUDA when available"}
    )
    debug: bool = field(
        default=False
    )
    stop: bool = field(
        default=False
    )
    exp_name: str = field(
        default='debug',
        metadata={"help": "Unique name of experiment"}
    )
    knowledge_base: str = field(
        default='kg',
    )
    edit_output: bool = field(
        default=False
    )
    num_edits: int = field(
        default=1,
    )
    descriptions: bool = field(
        default=False,
        metadata={"help": "Whether to consider descriptions of question entities"}
    )
    subclassof: bool = field(
        default=False,
        metadata={"help": "Whether to consider subclasses of question entities"}
    )
    instanceof: bool = field(
        default=False,
        metadata={"help": "Whether to consider instance of question entities"}
    )
    run_no_knowledge: bool = field(
        default=False,
        metadata={"help": "Whether to run a setting without retrieval."}
    )
@dataclass
class DataArguments:
    data_name: str = field(
        default='WebQuestions',
        metadata={"help": "Name of dataset"}
    )
    data_path: str = field(
        default='../data',
        metadata={"help": "Path of dataset"}
    )
    data_type: str = field(
        default='KGQA'
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size of dataset loaders"}
    )
    aliases: bool = field(
        default=True,
        metadata={"help": "Whether to consider aliases of answer entities"}
    )
    temporal: bool = field(
        default=False,
        metadata={"help": "Whether to return quadruples"}
    )

@dataclass
class LanguageModelArguments:
    model_type: str = field(
        default='flan',
        metadata={"help": "Type of pretrained model"}
    )
    model_name_or_path: str = field(
        default='google/flan-t5-base',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    max_source_length: int = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length"}
    )
    max_target_length: int = field(
        default=128,
        metadata={"help": "The maximum total sequence length for target text"}
    )
    max_new_token: int = field(
        default=256,
        metadata={"help": "The maximum new token for target texti"}
    )    
    cache_dir: Optional[str] = field(
        default='./cache',
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"}
    )
    device_map: str = field(
        default="auto"
    )
    question_prefix: str = field(
        default="Question: ",
        metadata={"help": "Text added prior to input question"}
    )
    question_postfix: str = field(
        default="Answer:",
        metadata={"help": "Text added next to input question"}
    )

    decoder_only: bool = field(
        default=False,
        metadata={"help": "Whether to use decoder-only model"}
    )
    chronosense_incontext: bool = field(
        default=False,
        metadata={"help": "Whether to use ChronoSense examples in the prompt"}
    )

@dataclass
class RetrieverArguments:
    use_retrieval: bool = field(
        default=True
    )
    retriever_name: str = field(
        default='mpnet',
        metadata={"help": "Name of retrieval model"}
    )
    retriever_top_k: int = field(
        default=10
    )
    retriever_batch_size: int = field(
        default=1024
    )
    retriever_sep_token: str = field(
        default=' '
    )
    index_dir: Optional[str] = field(
        default='./cache/pyserini'
    )
    qe_as_object: bool = field(
        default=False,
        metadata={"help": "Whether to consider the triples where question entity is an object."}
    )
    multihop: bool = field(
        default=False,
        metadata={"help": "Whether to consider second hop triples."}
    )
    kgbert_bert_model: str = field(
        default='bert-base-uncased',
        metadata={"help": "Name of BERT model used in KGBERT"}
    )
    kgbert_pretrained_dir: str = field(
        default='kgbert_pretrained',
        metadata={"help": "Directory for pre-trained KGBERT model"}
    )
    kgbert_max_seq_length: int = field(
        default=128
    )
    kgbert_plausibility: bool = field(
        default=False,
        metadata={"help": "Whether ranking with plausibility scores."}
    )
    score_threshold: float = field(
        default=-1
    )

    plain_reranker: bool = field(
        default=False,
        metadata={"help": "Whether using a plain reranker."}
    )
    no_answer_eval_setting: bool = field(
        default=False,
        metadata={"help": "Whether using 'No answer evaluation setting'."}
    )
    attach_subclassof: bool = field(
        default=False,
        metadata={"help": "Whether including subclassof info in the knowledge."}
    )
    
    attach_instanceof: bool = field(
        default=False,
        metadata={"help": "Whether including instanceof info in the knowledge."}
    )
    wikidata: bool = field(
        default=False,
        metadata={"help": "Whether to get subclassof/instanceof from the Wikidata."}
    )
    order_facts: bool = field(
        default=False,
        metadata={"help": "Whether to order events."}
    )
    recent_to_oldest: bool = field(
        default=True,
        metadata={"help": "Whether to order events recent to oldest."}
    )
    add_durations: bool = field(
        default=False,
        metadata={"help": "Whether to add event duration into prompt."}
    )
    add_relevancy_score: bool = field(
        default=False,
        metadata={"help": "Whether to add relevancy scores into prompt."}
    )
    use_temporal_relevancy_scores: bool = field(
        default=False,
        metadata={"help": "Whether to use temporal relevancy scores - distance based."}
    )
    use_positional_encoding: bool = field(
        default=False,
        metadata={"help": "Whether to use positional encoding."}
    )
    multitq_data: bool = field(
        default=False,
        metadata={"help": "Whether to use MultiTQ dataset."}
    )
    decomposition: bool = field(
        default=False,
        metadata={"help": "Whether to decompose the query."}
    )
    mintaka_cross: bool = field(
        default=False
    )
