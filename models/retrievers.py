import os
import itertools
from itertools import chain
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, LukeModel, AutoConfig
from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer
import torch
#from FlagEmbedding import FlagLLMReranker
from sentence_transformers import CrossEncoder
from metrics import accuracy
import copy
#import faiss
#from faiss import write_index, read_index
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import calendar
from datetime import datetime
import math
MODEL_NAMES = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "luke": "studio-ousia/luke-base",
    "bge": "BAAI/bge-base-en-v1.5",
    "nvidia": "nvidia/NV-Embed-v2",
    "bi-encoder-mintaka": "duyguislakoglu/mintaka5neg-distilroberta-base-ocl",
    "bi-encoder-webq": "duyguislakoglu/webq-5neg-distilroberta-base-ocl"}
from descriptions import get_subclassof_from_qid, get_instanceof_from_qid

from huggingface_hub import login
login(token="")

def normalize_month_case(date_str):
    return " ".join(
        part.capitalize() if part.isalpha() else part
        for part in date_str.split()
    )

def date_to_interval(date_str, fmt, granularity):
    date_str = normalize_month_case(date_str)
    dt = datetime.strptime(date_str, fmt)

    if granularity == "day":
        d = dt.strftime("%Y-%m-%d")
        return [d, d]

    if granularity == "month":
        last_day = calendar.monthrange(dt.year, dt.month)[1]
        return [
            f"{dt.year:04d}-{dt.month:02d}-01",
            f"{dt.year:04d}-{dt.month:02d}-{last_day:02d}",
        ]

    if granularity == "year":
        return [f"{dt.year:04d}-01-01", f"{dt.year:04d}-12-31"]

def extract_most_specific_interval(text):

    MONTHS = (
        "January|February|March|April|May|June|"
        "July|August|September|October|November|December"
    )

    pattern_specs = [
        (r"\b\d{4}-\d{2}-\d{2}\b", "%Y-%m-%d", "day"),
        (rf"\b\d{{1,2}} ({MONTHS}) \d{{4}}\b", "%d %B %Y", "day"),
        (rf"\b({MONTHS}) \d{{4}}\b", "%B %Y", "month"),
        (r"\b\d{4}-\d{2}\b", "%Y-%m", "month"),
        (r"\b\d{4}\b", "%Y", "year"),
    ]

    granularity_rank = {
        "day": 3,
        "month": 2,
        "year": 1,
    }

    matches = []

    for pattern, fmt, granularity in pattern_specs:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matches.append((
                granularity_rank[granularity],
                match.group(),
                fmt,
                granularity
            ))

    if not matches:
        return None

    _, date_str, fmt, granularity = max(matches)
    return date_to_interval(date_str, fmt, granularity)


class Retriever_KG(object):
    def __init__(self, args, data_type=None):
        super(Retriever_KG, self).__init__()
        self.num_of_triples = []
        self.args = args
        self.model = self.get_model()
        self.sem_model = None
        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        self.qid2label = {}
        self.qid2types = {}

        self.data_type = data_type
        if self.args.retriever_name == "finetuned-sbert-cross-encoder":
            self.sem_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",cache_folder = self.args.cache_dir)
        else:
            self.sem_model = self.model
    def get_model(self):

        if  self.args.retriever_name == "pretrained-sbert-cross-encoder":
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", default_activation_function=torch.nn.Sigmoid())

        if self.args.retriever_name == "finetuned-sbert-cross-encoder":
#            ce_name = "cross-encoder/finetuned-sbert-models/mintaka-0-neg-distilrobertabase2025-09-16_14-00-44"

            ce_name = "cross-encoder/finetuned-sbert-models/webq5neg2025-01-07_09-56-39"
#            ce_name = "cross-encoder/finetuned-sbert-models/mintaka5neg2025-01-09_10-30-30"
            if self.args.mintaka_cross:
                ce_name = "cross-encoder/finetuned-sbert-models/mintaka5neg2025-01-09_10-30-30"
            print(ce_name)

#            ce_name = "cross-encoder/xlgpu-cross-encoder/finetuned-sbert-models/1million-mintaka2025-04-10_23-40-46"
#            ce_name = "cross-encoder/finetuned-sbert-models/1million-mintaka2025-04-13_23-20-26"
            return CrossEncoder(ce_name)

        if self.args.retriever_name == "pretrained_bge_reranker":
            return FlagLLMReranker('BAAI/bge-reranker-v2-gemma')

        if self.args.retriever_name == "luke":
            return LukeModel.from_pretrained("studio-ousia/luke-base")

        if self.args.retriever_name == "vector_db":
            return SentenceTransformer("all-mpnet-base-v2")

        if self.args.retriever_name == "kgbert":
            model = BertForSequenceClassification.from_pretrained(self.args.kgbert_pretrained_dir, num_labels=2)
            device = torch.device("cpu" if torch.cuda.is_available  else "cpu")
            model.to(device)
            model.config.output_hidden_states = True
            model.config.return_dict = True
            return model

        print(MODEL_NAMES[self.args.retriever_name])
        return SentenceTransformer(
            MODEL_NAMES[self.args.retriever_name],
            cache_folder = self.args.cache_dir
        )

    def get_triplets(self, samples, kg):
        batch_question_entities = [list(set(
            [entity['name'] for entity in sample['question_entities']]))
            for sample in samples
        ]


        if self.args.qe_as_object:

            batch_raw_triplets = [list(itertools.chain(*
                [kg.get_triplets(entity)['raw_triplets'] +  kg.inverse_triplets[entity]['raw_triplets'] if entity in kg.inverse_triplets.keys() and  kg.inverse_triplets[entity]['raw_triplets'] != None else kg.get_triplets(entity)['raw_triplets'] for entity in question_entities]))
                for question_entities in batch_question_entities
            ]
            batch_ver_triplets = [list(itertools.chain(*
                [kg.get_triplets(entity)['ver_triplets'] + kg.inverse_triplets[entity]['ver_triplets'] if entity in kg.inverse_triplets.keys() and  kg.inverse_triplets[entity]['ver_triplets'] != None else kg.get_triplets(entity)['ver_triplets']  for entity in question_entities]))
                for question_entities in batch_question_entities
            ]


        elif self.args.multihop:
            first_hop_triplets = [list(itertools.chain(*
                [kg.get_triplets(entity)['raw_triplets'] for entity in question_entities]))
                for question_entities in batch_question_entities
            ]
            second_hop_triplets = [kg.get_triplets(triplet[2])['raw_triplets'] for triplet in first_hop_triplets[0]]
            second_hop_triplets = list(chain(*second_hop_triplets))

            first_hop_ver_triplets = [list(itertools.chain(*
                [kg.get_triplets(entity)['ver_triplets'] for entity in question_entities]))
                for question_entities in batch_question_entities
            ]
            second_hop_ver_triplets = [kg.get_triplets(triplet[2])['ver_triplets'] for triplet in first_hop_triplets[0]]
            second_hop_ver_triplets = list(chain(*second_hop_ver_triplets))

            self.num_of_first_hop = len(first_hop_triplets[0])
            self.num_of_second_hop = len(second_hop_triplets)

            batch_raw_triplets = [first_hop_triplets[0] +second_hop_triplets]
            batch_ver_triplets = [first_hop_ver_triplets[0] + second_hop_ver_triplets]
        else:

            batch_raw_triplets = [list(itertools.chain(*
                [kg.get_triplets(entity)['raw_triplets'] for entity in question_entities]))
                for question_entities in batch_question_entities
            ]
            batch_ver_triplets = [list(itertools.chain(*
                [kg.get_triplets(entity)['ver_triplets'] for entity in question_entities]))
                for question_entities in batch_question_entities
            ]


        assert len(samples) == \
               len(batch_question_entities) == \
               len(batch_raw_triplets) == \
               len(batch_ver_triplets)

        return batch_raw_triplets, batch_ver_triplets

    def encode(self, sentences):
        return self.model.encode(
            sentences,
            batch_size=self.args.retriever_batch_size,
            show_progress_bar=False,
        )

    def filter_triplets(self, raw_triplets, ver_triplets):
        #for index, triplet in enumerate(ver_triplets):
           # print(triplet)
           # print(len(triplet))

        indices = [
            index for index, triplet in enumerate(ver_triplets)
            if triplet[0] not in [None, ''] and \
               triplet[1] not in [None, ''] and \
               triplet[2] not in [None, '']
        ]

        if self.args.multihop:
            self.filtered_first_hop = (len([index for index in indices if index<self.num_of_first_hop]))


        return [raw_triplets[index] for index in indices], \
               [ver_triplets[index] for index in indices]

    def filter_triplets_no_answer(self, sample, raw_triplets, ver_triplets):
        #for index, triplet in enumerate(ver_triplets):
            # print(triplet)
            # print(len(triplet))

        indices = [
            index for index, triplet in enumerate(raw_triplets)
            if  triplet[0] not in [answer["name"] for answer in sample["answer_entities"]]  and \
                triplet[1] not in [answer["name"] for answer in sample["answer_entities"]]  and \
                triplet[2] not in [answer["name"] for answer in sample["answer_entities"]]
        ]
        return [raw_triplets[index] for index in indices], \
            [ver_triplets[index] for index in indices]

    def get_sem_subclassof_from_kg(self, entity, kg, query, sem_model=None):
        raw_triples = kg.get_triplet_copies(entity)["raw_triplets"]
        ver_triples = kg.get_triplet_copies(entity)["ver_triplets"]

        res = ""
        count = 0
        types = []
        for i in range(len(raw_triples)):
            if raw_triples[i][1] == "P279" and (ver_triples[i][2]!="" and ver_triples[i][2]!=None):
                count += 1
                types.append(ver_triples[i][2])

        if len(types) == 0:
            return "", 0
        if len(types) == 1:
            return types[0],1

        query_embedding = self.sem_model.encode([query])
        type_embeddings = self.sem_model.encode(types)


        similarities = util.dot_score(query_embedding, type_embeddings)[0].numpy()
        sorted_indices = np.argsort(-similarities)
        sorted_types = [types[index] for index in sorted_indices]

        #print("The query: " + query)
        #print("The types: " + str(types))
        #print("The similarities: "+ str(similarities))
        #print("The sorted types: " + str(sorted_types))

        return sorted_types[0], count
    def get_semantic_type_wikidata(self, types, query):
        if len(types)==1:
            return types[0]
        if len(types)==0:
            return ""
        type_embeddings = self.sem_model.encode(types)
        query_embedding = self.sem_model.encode([query])
        similarities = util.dot_score(query_embedding, type_embeddings)[0].numpy()
        sorted_indices = np.argsort(-similarities)
        sorted_types = [types[index] for index in sorted_indices]
        print("query: "+ query)
        print("types: "+ str(types))
        print("most relevant type: " +  sorted_types[0])
        return sorted_types[0]

    def get_sem_instanceof_from_kg(self, entity, kg, query, sem_model=None):
        raw_triples = kg.get_triplet_copies(entity)["raw_triplets"]
        ver_triples = kg.get_triplet_copies(entity)["ver_triplets"]

        res = ""
        count = 0
        types = []
        for i in range(len(raw_triples)):
            if raw_triples[i][1] == "P31" and (ver_triples[i][2]!="" and ver_triples[i][2]!=None):
                count += 1
                types.append(ver_triples[i][2])

        if len(types) == 0:
            return "", 0
        if len(types) == 1:
            return types[0],1
        query_embedding = self.sem_model.encode([query])

        type_embeddings = self.sem_model.encode(types)

        similarities = util.dot_score(query_embedding, type_embeddings)[0].numpy()
        sorted_indices = np.argsort(-similarities)
        sorted_types = [types[index] for index in sorted_indices]

        #print("The query: " + query)
        #print("The types: " + str(types))
        #print("The similarities: "+ str(similarities))
        #print("The sorted types: " + str(sorted_types))

        return sorted_types[0], count


    def get_subclassof_from_kg(self, entity, kg):
        raw_triples = kg.get_triplet_copies(entity)["raw_triplets"]
        ver_triples = kg.get_triplet_copies(entity)["ver_triplets"]

        res = ""
        count = 0
        for i in range(len(raw_triples)):
            if raw_triples[i][1] == "P279":
                count += 1
                if res == "":
                    res = ver_triples[i][2]
        return res, count


    def get_instanceof_from_kg(self, entity, kg):
        raw_triples = kg.get_triplet_copies(entity)["raw_triplets"]
        ver_triples = kg.get_triplet_copies(entity)["ver_triplets"]
        res = ""
        count = 0
        for i in range(len(raw_triples)):
            if raw_triples[i][1] == "P31":
                count += 1
                if res == "":
                    res = ver_triples[i][2]
        return res, count


    def add_info(self, raw_triples, ver_triples, kg, info_type, query=None):
        updated_triples = copy.deepcopy(ver_triples)

        for i in range(len(raw_triples)):
            subj_info = ""
            obj_info = ""
            triple = raw_triples[i]

            if info_type == "subclassof":
                if self.args.wikidata:
                    subj_info = self.get_semantic_type_wikidata(get_subclassof_from_qid(triple[0],self.sparql), query)
                    obj_info = self.get_semantic_type_wikidata(get_subclassof_from_qid(triple[2],self.sparql),query)
                else:
                    if query == None:
                        subj_info = self.get_subclassof_from_kg(triple[0], kg)[0]
                        obj_info =  self.get_subclassof_from_kg(triple[2], kg)[0]
                    else:
                        subj_info = self.get_sem_subclassof_from_kg(triple[0], kg, query)[0]
                        obj_info =  self.get_sem_subclassof_from_kg(triple[2], kg, query)[0]
                        #subj_info = self.get_subclassof_from_kg(triple[0], kg)[0]
                        #obj_info =  self.get_subclassof_from_kg(triple[2], kg)[0]


            elif info_type == "instanceof":
                if self.args.wikidata:
                    subj_info = self.get_semantic_type_wikidata(get_instanceof_from_qid(triple[0],self.sparql),query)
                    obj_info = self.get_semantic_type_wikidata(get_instanceof_from_qid(triple[2],self.sparql),query)
                else:
                    if query == None:
                        subj_info = self.get_instanceof_from_kg(triple[0], kg)[0]
                        obj_info = self.get_instanceof_from_kg(triple[2], kg)[0]
                    else:
                        subj_info = self.get_sem_instanceof_from_kg(triple[0], kg, query)[0]
                        obj_info = self.get_sem_instanceof_from_kg(triple[2], kg, query)[0]
                        #subj_info = self.get_instanceof_from_kg(triple[0], kg)[0]
                        #obj_info = self.get_instanceof_from_kg(triple[2], kg)[0]


            if subj_info !='' and subj_info != None:
                subj_info = " (" + subj_info + ")"
            else:
                subj_info= ''

            if obj_info !='' and obj_info != None:
                obj_info = " (" + obj_info + ")"

            else:
                obj_info= ''

            updated_triples[i][0] = updated_triples[i][0] + subj_info
            updated_triples[i][2] = updated_triples[i][2] + obj_info
        return updated_triples


    #def add_scores(ver_triples, scores):
    #    for i in len(ver_triples):
    #        ver_triples[i][2] =  ver_triples[i][2] + ". The relevancy score of this triple is"  + scores[i]
    #    return ver_triples

    def retrieve(self, samples, kg, return_ids=True, offset=0, *args, **kwargs):
        batch_raw_triplets, batch_ver_triplets = self.get_triplets(samples, kg)
        self.num_of_triples.append(len(batch_raw_triplets[0]))
        raw_results, ver_results = [], []
        score_list = []
        acc_at1, acc_at5, acc_at10, mrr = [], [], [], []
        acc_at15 = []
        hits = []
        print(self.data_type)
        if self.data_type == "TKGQA":
            import faiss
            from faiss import write_index, read_index
        for sample, raw_triplets, ver_triplets in zip(samples, batch_raw_triplets, batch_ver_triplets):
            assert len(raw_triplets) == len(ver_triplets)
            query = sample['question']

            print(query)

            if not self.args.multitq_data:
                match = re.search(r"\b\d{4}\b", query)
                if match:
                    question_time = match.group()
            else:
                question_time = extract_most_specific_interval(query)
                match = question_time
                print("Question Time: ")
                print(question_time)

            raw_triplets, ver_triplets = self.filter_triplets(raw_triplets, ver_triplets)

            if self.args.no_answer_eval_setting:
                raw_triplets, ver_triplets = self.filter_triplets_no_answer(sample, raw_triplets, ver_triplets)

            if self.args.attach_subclassof:
                ver_triplets = self.add_info(raw_triplets, ver_triplets, kg, "subclassof", query)
            if self.args.attach_instanceof:
                ver_triplets = self.add_info(raw_triplets, ver_triplets, kg, "instanceof", query)
# Comment the following if temporal
            if self.data_type!= "TKGQA" and len(ver_triplets) == 0:
                raw_results.append([])
                ver_results.append([])
                score_list.append([])
                continue
            if self.args.retriever_name == "kgbert":
                tokenizer = BertTokenizer.from_pretrained(self.args.kgbert_pretrained_dir, do_lower_case=True)

                if not self.args.kgbert_plausibility:
                    # 1. Take question embedding
                    tokens = ["[CLS]"] + tokenizer.tokenize(query) + ["[SEP]"]
                    segment_ids = [0] * len(tokens)
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    input_mask = [1] * len(input_ids)
                    padding = [0] * (self.args.kgbert_max_seq_length - len(input_ids))
                    input_ids += padding
                    input_mask += padding
                    segment_ids += padding

                    input_ids = torch.tensor([input_ids])
                    input_mask = torch.tensor([input_mask])
                    segment_ids = torch.tensor([segment_ids])

                    qemb = self.model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=None)#.hidden_states[0][0]

                    qemb = qemb.hidden_states[1][0,0,:]
                    query_embedding = [qemb]

                    # 2. Take triple embeddings
                    triplet_embeddings = []

                    for triplet in ver_triplets:
                        tokens = ["[CLS]"] + tokenizer.tokenize(triplet[0]) + ["[SEP]"]
                        segment_ids = [0] * len(tokens)

                        tokens += tokenizer.tokenize(triplet[1])  + ["[SEP]"]
                        segment_ids += [1] * (len(tokenizer.tokenize(triplet[1]) ) + 1)

                        tokens += tokenizer.tokenize(triplet[2])  + ["[SEP]"]
                        segment_ids += [0] * (len(tokenizer.tokenize(triplet[2])) + 1)
                        input_ids = tokenizer.convert_tokens_to_ids(tokens)

                        input_mask = [1] * len(input_ids)
                        padding = [0] * (self.args.kgbert_max_seq_length - len(input_ids))
                        input_ids += padding
                        input_mask += padding
                        segment_ids += padding

                        input_ids = torch.tensor([input_ids])
                        input_mask = torch.tensor([input_mask])
                        segment_ids = torch.tensor([segment_ids])

                        triplet_embedding = self.model(input_ids,attention_mask=input_mask, token_type_ids=segment_ids,labels=None)
                        triplet_embeddings.append(triplet_embedding.hidden_states[1][0,0,:])
                else:
                    # Take question+triple embeddings
                    question_triplet_embeddings = []

                    for triplet in ver_triplets:
                        tokens = ["[CLS]"] + tokenizer.tokenize(query) + ["[SEP]"] + tokenizer.tokenize(triplet[0]) + ["[SEP]"]
                        segment_ids = [0] * len(tokens)

                        tokens += tokenizer.tokenize(triplet[1])  + ["[SEP]"]
                        segment_ids += [1] * (len(tokenizer.tokenize(triplet[1]) ) + 1)

                        tokens += tokenizer.tokenize(triplet[2])  + ["[SEP]"]
                        segment_ids += [0] * (len(tokenizer.tokenize(triplet[2])) + 1)
                        input_ids = tokenizer.convert_tokens_to_ids(tokens)

                        input_mask = [1] * len(input_ids)
                        padding = [0] * (self.args.kgbert_max_seq_length - len(input_ids))
                        input_ids += padding
                        input_mask += padding
                        segment_ids += padding

                        input_ids = torch.tensor([input_ids])
                        input_mask = torch.tensor([input_mask])
                        segment_ids = torch.tensor([segment_ids])

                        question_triplet_embedding = self.model(input_ids,attention_mask=input_mask, token_type_ids=segment_ids,labels=None)

                        softmax = torch.nn.Softmax(dim=1)
                        scores = softmax(question_triplet_embedding.logits).detach().numpy()

                        question_triplet_embeddings.append(scores[0][1])

            elif self.args.retriever_name == "luke":
                tokenizer = AutoTokenizer.from_pretrained("studio-ousia/luke-base")
                entities = []
                entity_spans = []
                for qe in sample["question_entities"]:
                    entities.append(qe["mention"])
                    entity_spans.append((qe["span"]["start"], qe["span"]["end"]))
                encoding = tokenizer(query, entities=entities, entity_spans=entity_spans, add_prefix_space=True, return_tensors="pt")

                outputs = self.model(**encoding)
                query_embedding = [outputs.last_hidden_state[0][0]]

                triplet_embeddings = []
                for triplet in ver_triplets:
                    text =  f"{triplet[0]}{self.args.retriever_sep_token}{triplet[1]}{self.args.retriever_sep_token}{triplet[2]}"
                    triplet_entities = [triplet[0], triplet[2]]
                    triplet_entity_spans = [(0,len(triplet[0])), (len(text)-len(triplet[2]), len(text))]
                    triplet_encoding = tokenizer(text,entities=triplet_entities, entity_spans=triplet_entity_spans, add_prefix_space=True, return_tensors="pt")
                    triplet_embedding = self.model(**triplet_encoding).last_hidden_state[0][0]
                    triplet_embeddings.append(triplet_embedding)

            elif self.args.retriever_name == "vector_db":
                q_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")
                q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
                self.args.retriever_top_k = 15
                if kg.args.data_name == "TimeQuestions":
                    data_name = "timequestions"
                else:
                    data_name = "multitq"

                print(data_name)

                if data_name== "timequestions":
                    index = faiss.read_index("../data/timequestions/timestamp.index")
                else:
                    index = faiss.read_index("../data/multitq/multitq-kg.index")

                if hasattr(index, "make_direct_map"):
                    index.make_direct_map()          # builds id -> (list, offset)

                def order_facts(lst, recent_to_oldest=True):
                    if self.args.multitq_data:
                        return order_facts_multitq(lst, recent_to_oldest)

                    #handle invalid years here
                    intervals = []
                    for parts in lst:
                        if parts[3] !='1' and parts[4] =='2916' :
                            intervals.append([parts[3],int(parts[3])+0.5])
                        elif parts[3] =='1' and parts[4] !='2916' :
                            intervals.append([int(parts[4])-0.5,parts[4]])
                        elif parts[3] =='1' and parts[4] =='2916':
                            if self.args.recent_to_oldest: # timeless facts are always in the beginning
                                intervals.append([float("inf"),float("inf")])
                            else:
                                intervals.append([float("-inf"),float("-inf")])
                        else:
                            intervals.append([parts[3],parts[4]])
                    if not self.args.recent_to_oldest:
                        return sorted(range(len(intervals)), key=lambda i: (float(intervals[i][1]), float(intervals[i][0])))
                    new_order = sorted(range(len(intervals)), key=lambda i: (float(intervals[i][1]), float(intervals[i][0])), reverse=True)
                    return new_order

                def order_facts_multitq(lst, recent_to_oldest=True):
                    date_objects = [datetime.strptime(d[3], "%Y-%m-%d") for d in lst]

                    if not self.args.recent_to_oldest:
                        return sorted(range(len(date_objects)), key=lambda i: date_objects[i])
                    new_order = sorted(range(len(date_objects)), key=lambda i: date_objects[i], reverse=True)

                    return new_order

                def get_durations(from_quadruple_lst):
                    durations = []
                    for parts in from_quadruple_lst:
                        if parts[3] !='1' and parts[4] =='2916' :
                            durations.append(-1)
                        elif parts[3] =='1' and parts[4] !='2916' :
                            durations.append(-1)
                        elif parts[3] =='1' and parts[4] =='2916':
                            durations.append(-1)
                        else:
                            if parts[4]!=parts[3]:
                               #durations.append(int(parts[4])-int(parts[3])+1)
                                durations.append(int(parts[4])-int(parts[3]))
                            else:
                                durations.append(-1)
                    print(durations)
                    return durations

                def add_relevancy_score(from_corpus, distances):
                    for i in range(len(from_corpus)):
                        from_corpus[i][0] = from_corpus[i][0] + " (Relevancy score: " + str(distances[0][i])[:4]+")"
                    return from_corpus

                def add_durations(from_corpus, from_quadruple_lst):
                    durations = get_durations(from_quadruple_lst)

                    for i in range(len(from_corpus)):
                        if durations[i] != -1:
                            # Original
                             from_corpus[i][0] = from_corpus[i][0] +" (Duration: " + str(durations[i]) +" years)"
                            #from_corpus[i][0] = "(Duration: " + str(durations[i]) +" years) " + from_corpus[i][0]
                    return from_corpus

                def multitq_distance_based_score(question_time,day):
                    date = datetime.strptime(day, "%Y-%m-%d").date()
                    start = datetime.strptime(question_time[0], "%Y-%m-%d").date()
                    end = datetime.strptime(question_time[1], "%Y-%m-%d").date()

                    if start <= date <= end:
                        return float("inf")
                    elif date < start:
                        return (start - date).days
                    else:
                        return (date - end).days



                def distance_based_score(question_year, interval):
                    if interval == [-1,-1]:
                        return float("-inf")
                    if question_year >= interval[0] and question_year <= interval[1]:
                        print(question_year, interval)
                        return float("inf")
                    elif question_year < interval[0]:
                        return interval[0]-question_year
                    else:
                        return question_year-interval[1]

                def positional_enc_score(question_year, interval, time_vectors):
                    if interval == [-1,-1]:
                        return 0
                    return np.dot_score(time_vectors[question_year], time_vectors[int((interval[3]+interval[4])/2)])
                def normalize_list(values):
                    """
                    Normalizes a list to the range [0, 1].

                    Args:
                        values: A list of numbers.

                    Returns:
                        A new list with normalized values.
                    """
                    if not values:
                        return []

                    finite_values = [x for x in values if math.isfinite(x)]
                    if finite_values:
                        min_val = min(finite_values)
                        max_val = max(finite_values)

                    normalized_list = []
                    for value in values:
                        if value == float("inf"):
                            normalized_list.append(1)
                        elif value == float("-inf"):
                            normalized_list.append(0)
                        else:
                            if min_val == max_val:
                                normalized_list.append(0)
                            else:
                                normalized_list.append((1-((value - min_val) / (max_val - min_val)))* 0.9)
                    return normalized_list


                def get_time_scores(question_time, from_quadruple_lst):
                        if self.args.multitq_data:
                            days = [quad[3] for quad in from_quadruple_lst]
                            return [multitq_distance_based_score(question_time,day) for day in days]

                        question_year = int(question_time)
                        intervals = []
                        print(from_quadruple_lst)
                        for parts in from_quadruple_lst:
                            parts[3] = int(parts[3])
                            parts[4] = int(parts[4])

                            if parts[3] !=1 and parts[4] ==2916 :
                                intervals.append([parts[3],parts[3]])
                            elif parts[3] ==1 and parts[4] !=2916 :
                                intervals.append([parts[4],parts[4]])
                            elif parts[3] ==1 and parts[4] ==2916:
                                intervals.append([-1,-1])
                            else:
                                intervals.append([parts[3],parts[4]])
                        if self.args.use_positional_encoding:
                            time_vectors = getPositionEncoding(3000, 64)
                            return [positional_enc_score(question_year, interval, time_vectors) for interval in intervals]
                        return [distance_based_score(question_year, interval) for interval in intervals]

                # Code taken from machinelearningmastery.com
                def getPositionEncoding(seq_len, d, n=10000):
                    P = np.zeros((seq_len, d))
                    for k in range(seq_len):
                        for i in np.arange(int(d/2)):
                            denominator = np.power(n, 2*i/d)
                            P[k, 2*i] = np.sin(k/denominator)
                            P[k, 2*i+1] = np.cos(k/denominator)
                    return P

                if self.args.decomposition:

                    #subqueries = [query, query]

                    subqueries = re.findall(r"\[(.*?)\]", query)

                    q_emb1 = self.model.encode([subqueries[0]], convert_to_numpy=True).astype("float32")
                    q_emb1 /= np.linalg.norm(q_emb1, axis=1, keepdims=True)

                    q_emb2 = self.model.encode([subqueries[1]], convert_to_numpy=True).astype("float32")
                    q_emb2 /= np.linalg.norm(q_emb2, axis=1, keepdims=True)

                    distances1, corpus_ids1 = index.search(q_emb1, k=self.args.retriever_top_k)
                    print("Neighbor scores qemb1:", distances1)

                    corpus_list = []

                    with open(f"../data/{data_name}/formatted_no_unknown_years_timestamp.txt", "r", encoding="utf-8") as f:
                        corpus_list = [line.strip() for line in f]

                    quadruple_lst = []
                    with open(f"../data/{data_name}/kg/full.txt", "r", encoding="utf-8") as f:
                        quadruple_lst = [line.strip().split("\t") for line in f]


                    from_corpus1 = []
                    from_quadruple_lst1 = []
                    hits1 = []


                    for i in range(self.args.retriever_top_k):
                        #if corpus_list[corpus_ids1[0][i]].strip().split("\t")[0] != quadruple_lst[corpus_ids1[0][i]].strip().split("\t")[0]:
                        from_corpus1.append([corpus_list[corpus_ids1[0][i]], "",""])
                        from_quadruple_lst1.append(quadruple_lst[corpus_ids1[0][i]])
                        hits1.append(int(accuracy([sample],[corpus_list[corpus_ids1[0][i]]], kg, save_pkl=False)))


                    distances2, corpus_ids2 = index.search(q_emb2, k=self.args.retriever_top_k)
                    #print("Neighbor indices:", corpus_ids2)
                    print("Neighbor scores qemb2:", distances2)

                    from_corpus2 = []
                    from_quadruple_lst2 = []
                    hits2 = []


                    for i in range(self.args.retriever_top_k):
                        #if corpus_list[corpus_ids2[0][i]].strip().split("\t")[0] != quadruple_lst[corpus_ids2[0][i]].strip().split("\t")[0]:
                        from_corpus2.append([corpus_list[corpus_ids2[0][i]], "",""])
                        from_quadruple_lst2.append(quadruple_lst[corpus_ids2[0][i]])
                        hits2.append(int(accuracy([sample],[corpus_list[corpus_ids2[0][i]]], kg, save_pkl=False)))

                    merged = []

                    for i in range(self.args.retriever_top_k):
                        merged.append((
                            distances1[0][i],
                            from_corpus1[i],
                            from_quadruple_lst1[i],
                            hits1[i]
                        ))

                    for i in range(self.args.retriever_top_k):
                        merged.append((
                            distances2[0][i],
                            from_corpus2[i],
                            from_quadruple_lst2[i],
                            hits2[i]
                        ))

                    merged = sorted(merged, key=lambda x: x[0], reverse=True)
                    merged = merged[:self.args.retriever_top_k]
                    from_corpus = []
                    from_quadruple_lst = []
                    hits = []
                    distances = []
                    for item in merged:
                        distance, corpus_item, quadruple_item, hit = item
                        from_corpus.append(corpus_item)
                        from_quadruple_lst.append(quadruple_item)
                        distances.append(distance)
                        hits.append(hit)


                    if self.args.add_durations:
                        print(from_quadruple_lst)
                        from_corpus = add_durations(from_corpus, from_quadruple_lst)

                    ver_results.append(from_corpus)
                    raw_results.append(from_corpus)
                    # if len(ver_triplets) == 0:
                    #        raw_results.append([])
                    #        ver_results.append([])

                    score_list.append(distances)
                    #print(sample["answer_entities"])


                    sample_acc_at1 = hits[0]
                    if 1 in hits[:5]:
                        sample_acc_at5 = 1
                    else:
                        sample_acc_at5  = 0
                    if 1 in hits[:10]:
                        sample_acc_at10 = 1
                    else:
                        sample_acc_at10 = 0
                    if 1 in hits[:15]:
                        sample_acc_at15 = 1
                    else:
                        sample_acc_at15 = 0
                    acc_at1.append(sample_acc_at1)
                    acc_at5.append(sample_acc_at5)
                    acc_at10.append(sample_acc_at10)
                    acc_at15.append(sample_acc_at15)
                    if 1 in hits:
                        sample_mrr = 1/(hits.index(1)+1)
                    else:
                        sample_mrr = 0

                    mrr.append(sample_mrr)
                    continue

                distances, corpus_ids = index.search(q_emb, k=self.args.retriever_top_k)
                #print("Neighbor indices:", corpus_ids)
                print("Neighbor scores:", distances)

                corpus_list = []

                with open(f"../data/{data_name}/formatted_no_unknown_years_timestamp.txt", "r", encoding="utf-8") as f:
                    corpus_list = [line.strip() for line in f]

                if self.args.order_facts or self.args.add_durations or self.args.use_temporal_relevancy_scores:
                    quadruple_lst = []
                    with open(f"../data/{data_name}/kg/full.txt", "r", encoding="utf-8") as f:
                        quadruple_lst = [line.strip().split("\t") for line in f]


                from_corpus = []
                from_quadruple_lst = []


                for i in range(self.args.retriever_top_k):
                    #if corpus_list[corpus_ids[0][i]].strip().split("\t")[0] != quadruple_lst[corpus_ids[0][i]].strip().split("\t")[0]:
                    from_corpus.append([corpus_list[corpus_ids[0][i]], "",""])
                    if self.args.order_facts or self.args.add_durations or self.args.use_temporal_relevancy_scores:
                        from_quadruple_lst.append(quadruple_lst[corpus_ids[0][i]])
                    hits.append(int(accuracy([sample],[corpus_list[corpus_ids[0][i]]], kg, save_pkl=False)))


                if self.args.order_facts:

                    new_order = order_facts(from_quadruple_lst,self.args.recent_to_oldest)
                    from_corpus = [from_corpus[i] for i in new_order]
                    hits = [hits[i] for i in new_order]
                    distances = [[distances[0][i] for i in new_order]]


                if self.args.add_durations:
                    print(from_quadruple_lst)
                    from_corpus = add_durations(from_corpus, from_quadruple_lst)

                if self.args.add_relevancy_score:
                    from_corpus = add_relevancy_score(from_corpus, distances)

                if self.args.use_temporal_relevancy_scores and match: # question time detected
                    temporal_relevancy_scores = get_time_scores(question_time, from_quadruple_lst)
                    print(temporal_relevancy_scores)
                    temporal_relevancy_scores = normalize_list(temporal_relevancy_scores)
                    print("Normalized list: ")
                    print(temporal_relevancy_scores)

                    total_scores = [x + y for x, y in zip(distances[0], temporal_relevancy_scores)]
                    print(total_scores)
                    # remove if weighting
                    #total_scores = temporal_relevancy_scores

                    new_order = sorted(range(len(total_scores)), key=lambda i: -total_scores[i])
                    print(new_order)
                    print([total_scores[i] for i in new_order])
                    from_corpus = [from_corpus[i] for i in new_order]
                    hits = [hits[i] for i in new_order]
                    distances = [[distances[0][i] for i in new_order]]


                ver_results.append(from_corpus)
                raw_results.append(from_corpus)
               # if len(ver_triplets) == 0:
             #        raw_results.append([])
             #        ver_results.append([])

                score_list.append(distances[0])
                #print(sample["answer_entities"])


                sample_acc_at1 = hits[0]
                if 1 in hits[:5]:
                    sample_acc_at5 = 1
                else:
                    sample_acc_at5  = 0
                if 1 in hits[:10]:
                    sample_acc_at10 = 1
                else:
                    sample_acc_at10 = 0
                if 1 in hits[:15]:
                    sample_acc_at15 = 1
                else:
                    sample_acc_at15 = 0
                acc_at1.append(sample_acc_at1)
                acc_at5.append(sample_acc_at5)
                acc_at10.append(sample_acc_at10)
                acc_at15.append(sample_acc_at15)
                if 1 in hits:
                    sample_mrr = 1/(hits.index(1)+1)
                else:
                    sample_mrr = 0

                mrr.append(sample_mrr)
                continue

            else:
                if not self.args.retriever_name in ["pretrained-sbert-cross-encoder", "finetuned-sbert-cross-encoder", "pretrained_bge_reranker"]:
                    query_embedding = self.encode([query])

                #                    if self.args.multihop and "bi-encoder" in self.args.retriever_name:
                #                        triplet_embeddings = [
                #                                self.encode([f"{triplet[0]}{self.args.retriever_sep_token}{triplet[1]}{self.args.retriever_sep_token}{triplet[2]}"])[0]
                #                                for triplet in ver_triplets
                #                        ]

                #                    else:
                    triplet_embeddings = self.encode([
                        	f"{triplet[0]}{self.args.retriever_sep_token}{triplet[1]}{self.args.retriever_sep_token}{triplet[2]}"
                        	for triplet in ver_triplets
                        ])

            if self.args.kgbert_plausibility:
                similarities = np.array(question_triplet_embeddings)

            elif  self.args.plain_reranker:
                similarities = np.array([self.model.compute_score([query, triplet])[0]  for triplet in ver_triplets])
                #print(similarities)
            elif self.args.retriever_name in ["pretrained-sbert-cross-encoder", "finetuned-sbert-cross-encoder"]:
               # print(triplet)
                if self.args.multihop:

                    sim_first_hop = np.array([self.model.predict([query, " ".join(triplet[1:])])  for triplet in ver_triplets[:self.filtered_first_hop]])
                    sim_second_hop = np.array([self.model.predict([query, " ".join(triplet)])  for triplet in ver_triplets[self.filtered_first_hop:]])
                    similarities = np.concatenate([sim_first_hop, sim_second_hop])
                else:
                    similarities = np.array([self.model.predict([query, " ".join(triplet[1:])])  for triplet in ver_triplets])
                #print(similarities):

            elif "bi-encoder" in self.args.retriever_name:
                similarities = self.model.similarity(query_embedding, triplet_embeddings)[0].numpy()

            else:
                #query_embedding = [t.detach().numpy() for t in query_embedding]
                #triplet_embeddings = [t.detach().numpy() for t in triplet_embeddings]
                if self.args.multihop:
                    similarities = np.array([util.dot_score(query_embedding, triplet)[0][0] for triplet in triplet_embeddings])
                else:
                   similarities = util.dot_score(query_embedding, triplet_embeddings)[0].numpy()

            sorted_indices = np.argsort(-similarities)
            sorted_raw_triples = [raw_triplets[index] for index in sorted_indices]
            sorted_ver_triples = [ver_triplets[index] for index in sorted_indices]

            top_raw = sorted_raw_triples[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k]
            top_ver = sorted_ver_triples[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k]
            #raw_results.append(sorted_raw_triples[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k])
            #ver_results.append(sorted_ver_triples[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k])
            raw_results.append(top_raw)
            ver_results.append(top_ver)
            #print(-np.sort(-similarities)[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k])
            score_list.append(-np.sort(-similarities)[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k])


            for triple in sorted_raw_triples[offset*self.args.retriever_top_k:(offset+1)*self.args.retriever_top_k]:


                if bool(set([answer["name"] for answer in sample["answer_entities"]]) & set(triple[:3])):
                    hits.append(1)
                else:
                    hits.append(0)


            sample_acc_at1 = hits[0]
            if 1 in hits[:5]:
                sample_acc_at5 = 1
            else:
                sample_acc_at5  = 0

            if 1 in hits[:10]:
                sample_acc_at10 = 1
            else:
                sample_acc_at10 = 0
            if 1 in hits[:15]:
                sample_acc_at15 = 1
            else:
                sample_acc_at15 = 0

            acc_at1.append(sample_acc_at1)
            acc_at5.append(sample_acc_at5)
            acc_at10.append(sample_acc_at10)
            acc_at15.append(sample_acc_at15)

            if 1 in hits:
                sample_mrr = 1/(hits.index(1)+1)
            else:
                sample_mrr = 0

            mrr.append(sample_mrr)

        #print(np.mean(acc_at1), np.mean(acc_at5), np.mean(acc_at10), np.mean(mrr))

        return ver_results if return_ids == False \
                else (ver_results, raw_results, score_list, (hits,acc_at1, acc_at5, acc_at10, mrr,acc_at15))
