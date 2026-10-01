import os
import json
import copy

from typing import Dict
from multiprocessing import cpu_count

import torch
from torch.utils.data import Dataset, DataLoader

FOLDER_NAMES = {
    'WebQuestions': 'webq',
    'Mintaka': 'mintaka',
    'NaturalQuestions': 'nq_open',
    'HotpotQA': 'hotpot_qa',
    'CronQuestions': 'cronquestions',
    'TimeQuestions': 'timequestions',
    'MultiTQ': 'multitq'
    }

class QADataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind: int):
        return self.data[ind]

    def __iter__(self):
        for sample in self.data:
            yield sample

    @staticmethod
    def collate(x):
        return x

def get_qa_datasets(args, cpu_usage_ratio=1.0):
    datasets = {
        split: QADataset([
            json.loads(sample) for sample in \
            list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', f'{split}-samples.jsonl'), 'r'))
        ])
        for split in args.data_splits
    }
    data_loaders = {
        split: DataLoader(
            datasets[split],
            batch_size = args.batch_size,
            shuffle = split == 'train' and not isinstance(dataset, torch.utils.data.IterableDataset),
            num_workers = int(cpu_count() * cpu_usage_ratio),
            collate_fn = datasets[split].collate
        )
        for split, dataset in datasets.items()
    }
    return datasets, data_loaders

class KG(object):
    def __init__(self, args):
        super(KG, self).__init__()

        self.args = args
        self.kg_name = 'freebase' if 'FB' in args.data_name else 'wikidata'

        self.jsonl_entities = list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', 'entities.jsonl'), 'r'))
        self.jsonl_relations = list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', 'relations.jsonl'), 'r'))
        self.jsonl_triplets = list(open(os.path.join(args.data_path, f'{FOLDER_NAMES[args.data_name]}', f'{self.kg_name}-kg.jsonl'), 'r'))

        self.entities = self.init_entities(self.jsonl_entities)
        self.relations = self.init_relations(self.jsonl_relations)
        self.triplets = self.init_triplets(self.jsonl_triplets)

       # self.change_data_format(self.triplets, self.entities, self.relations)

    def init_entities(self, jsonl_entities):
        entities = {}
        for entity in jsonl_entities:
            entity = json.loads(entity)
            entities[entity['entity']] = {
                'mention': entity['mention'],
                'aliases': entity['aliases']
            }
        return entities

    def init_relations(self, jsonl_relations):
        relations = {}
        for relation in jsonl_relations:
            relation = json.loads(relation)
            relations[relation['relation']] = {
                'mention': relation['mention']
            }
        return relations

    def init_triplets(self, jsonl_triplets):
        self.inverse_triplets = {}

        triplets = {}

        for sample in jsonl_triplets:
            sample = json.loads(sample)
            if self.args.temporal:
                sample['triples'] = [(sample['id'], triplet[0], triplet[1], triplet[2]) for triplet in sample['triples']]
            else:
                sample['triples'] = [(sample['id'], triplet[0], triplet[1]) for triplet in sample['triples']]
            triplets[sample['id']] = {
                'raw_triplets': sample['triples'],
                'ver_triplets': self.verbalizer(sample['triples'])
            }

            for triplet in sample['triples']:
                if triplet[2] in self.inverse_triplets.keys():
                    self.inverse_triplets[triplet[2]]['raw_triplets'].append(triplet)
                    if len(self.verbalizer([triplet])[0]) == 2:
                        print(self.verbalizer([triplet])[0])

                    self.inverse_triplets[triplet[2]]['ver_triplets'].append(self.verbalizer([triplet])[0])
                else:
                    self.inverse_triplets[triplet[2]] = {'raw_triplets':[triplet],'ver_triplets':self.verbalizer([triplet])}

        #print(self.inverse_triplets)

        return triplets

   # def query_data_format(self, triplets, entities, relations):
        

    def change_data_format(self, triplets, entities, relations):
        with open(self.args.data_name+'_triples.txt','w') as file1:
            all_triples = set()
            for triplet in triplets:
                for raw_triple in triplets[triplet]["raw_triplets"]:
                    if raw_triple not in all_triples:
                        all_triples.add(raw_triple)
                        file1.write('\t'.join(raw_triple)+"\n")
        print("The triples are saved.")

        with open(self.args.data_name+'_entity2name.txt','w') as file2:
            for entity in entities:
                name = entities[entity]['mention']
                if entities[entity]['mention'] is None:
                    name = ""
                file2.write(entity + "\t" + name +"\n")
        print("The entities are saved.")

        with open(self.args.data_name+'_relation2name.txt','w') as file3:
            for relation in relations:
                name = relations[relation]['mention']
                name = name.replace("<inv>-", "inverse ")
                if relations[relation]['mention'] is None:
                    name = ""
                file3.write(relation + "\t" + name +"\n")
        print("The relations are saved.")

    def get_triplets(self, entity_name):
        return self.triplets[entity_name] \
            if entity_name in self.triplets.keys() \
            else {'raw_triplets': [], 'ver_triplets': []}

    def get_triplet_copies(self, entity_name):
        return copy.deepcopy(self.triplets[entity_name]) \
            if entity_name in self.triplets.keys() \
            else {'raw_triplets': [], 'ver_triplets': []}
    def get_aliases(self, entity_name):
        return self.entities[entity_name]['aliases'] \
            if entity_name in self.entities.keys() and self.entities[entity_name]['aliases'] is not None \
            else []

    def verbalizer(self, raw_triplets):
        if self.args.temporal:
            return [
            [
                self.entities[triplet[0]]['mention'] \
                    if triplet[0] in self.entities.keys()
                    else '',
                self.relations[triplet[1]]['mention'] \
                    if triplet[1] in self.relations.keys() \
                    else '',
                self.entities[triplet[2]]['mention'] \
                    if triplet[2] in self.entities.keys() \
                    else ('' if triplet[2][0] == 'Q' else triplet[2]),
                "(" + triplet[3][0] +  ", " + triplet[3][1] + ")"
            ]
            for triplet in raw_triplets
            ]
        return [
            [
                self.entities[triplet[0]]['mention'] \
                    if triplet[0] in self.entities.keys()
                    else '',
                self.relations[triplet[1]]['mention'] \
                    if triplet[1] in self.relations.keys() \
                    else '',
                self.entities[triplet[2]]['mention'] \
                    if triplet[2] in self.entities.keys() \
                    else ('' if triplet[2][0] == 'Q' else triplet[2])
            ]
            for triplet in raw_triplets
        ]
