import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer, util

import random
import pickle
import logging

from tqdm import tqdm

import numpy as np

import torch

from transformers import HfArgumentParser, set_seed
from configs import GlobalArguments, DataArguments, LanguageModelArguments, RetrieverArguments
from eval_by_question_type import eval_tq, eval_mtq 
from data import get_qa_datasets, KG
from models.language_models import LanguageModel
from models.retrievers import Retriever_KG
from metrics import accuracy, f1, em
from descriptions import get_desc_of_list_of_IDs, get_subclassof_list_of_IDs,  get_instanceof_list_of_IDs

from SPARQLWrapper import SPARQLWrapper, JSON


RETRIEVER_NAMES = {
    'kg': Retriever_KG,
}
#torch.cuda.set_device(3)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
class Runner(object):
    def __init__(self, glb_args, data_args, lm_args, ret_args):
        super(Runner, self).__init__()

        self.glb_args = glb_args
        self.data_args = data_args
        self.lm_args = lm_args
        self.ret_args = ret_args

        self.sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
        
        self.output_path = self.get_output_path()
        self.logger = self.get_logger()
        self.logger.info(f"Global Arguments: {glb_args}")
        self.logger.info(f"Dataset Arguments: {data_args}")
        self.logger.info(f"Language Model Arguments: {lm_args}")
        self.logger.info(f"Retriever Arguments: {ret_args}")

        set_seed(glb_args.seed)

        self.qa_datasets, self.qa_data_loaders = get_qa_datasets(data_args)
        self.kg = KG(data_args) if data_args.data_type == 'KGQA' or data_args.data_type == 'TKGQA' else None
        self.lang_model = LanguageModel(lm_args)
        data_type = data_args.data_type
        if data_args.data_name in ["TimeQuestions, MultiTQ"]:
            data_type= "TKGQA"
        self.retriever = RETRIEVER_NAMES[glb_args.knowledge_base](ret_args, data_type=data_type)
    def get_logger(self):
        logging.basicConfig(
            format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt = "%m/%d/%Y %H:%M:%S",
            level = logging.INFO,
            filename = f"{self.output_path}/logs.txt"
        )
        logger = logging.getLogger(__name__)
        return logger

    def retrieve_generate(self, samples, turn=0):
        questions = [sample['question'] for sample in samples]
        if self.glb_args.subclassof or self.glb_args.instanceof:

            question_entities = [sample['question_entities'] for sample in samples] # [[q1, q2], [q3]]
            subclasses = []
            instances = []
            question_entity_mentions = [] 
            
            for i, qe in enumerate(question_entities):
                # get list of qids
                question_entity_ids = [x['name'] for x in qe]
                question_entity_mentions.append([x['mention'] for x in qe])

                if self.ret_args.wikidata:
                    if self.glb_args.subclassof:
                        subclasses.append([self.retriever.get_semantic_type_wikidata(types_per_qe, questions[i]) for types_per_qe in get_subclassof_list_of_IDs(question_entity_ids,self.sparql)])
                        #subclasses.append(get_subclassof_list_of_IDs(question_entity_ids))
                    if self.glb_args.instanceof:
                        #instances.append(get_instanceof_list_of_IDs(question_entity_ids))
                        instances.append([self.retriever.get_semantic_type_wikidata(types_per_qe, questions[i]) for types_per_qe in get_instanceof_list_of_IDs(question_entity_ids,self.sparql)])

                else:
                    if self.glb_args.subclassof:
                        #subclasses.append([self.retriever.get_subclassof_from_kg(entity_id, self.kg)[0] for entity_id in question_entity_ids])
                        subclasses.append([self.retriever.get_sem_subclassof_from_kg(entity_id, self.kg, questions[i], self.retriever.sem_model)[0] for entity_id in question_entity_ids])
                    if self.glb_args.instanceof:
                        #instances.append([self.retriever.get_instanceof_from_kg(entity_id, self.kg)[0] for entity_id in question_entity_ids])  
                        instances.append([self.retriever.get_sem_instanceof_from_kg(entity_id, self.kg,questions[i], self.retriever.sem_model)[0] for entity_id in question_entity_ids])
            info = []
            #question_entity_mentions = [x['mention'] for x in qe]
#            print(question_entity_mentions) 
            for i in range(len(samples)):
                txt = ''
                for j, mention in enumerate(question_entity_mentions[i]):
                    if self.glb_args.subclassof:
                        if subclasses[i][j] != "":
                            txt += " " + mention + " is a subclass of " + subclasses[i][j] + "."
                    if self.glb_args.instanceof:
                        if instances[i][j] != "":    
                            txt += " " + mention + " is a " + instances[i][j] + "."
                info.append(txt)

            for i in range(len(samples)):
                if self.glb_args.subclassof:
                    questions[i] = questions[i] + info[i]
                    samples[i]['question']= questions[i]  

                if self.glb_args.instanceof:
                    questions[i] = questions[i] + info[i]
                    samples[i]['question']= questions[i]
  
        if self.glb_args.descriptions:

            question_entities = [sample['question_entities'] for sample in samples] # [[q1, q2], [q3]]

            descriptions = []
            for i, qe in enumerate(question_entities):
                # get list of qids
                question_entity_ids = [x['name'] for x in qe]
                descriptions.append(get_desc_of_list_of_IDs(question_entity_ids))

            for i in range(len(samples)):
                questions[i] = questions[i] + " " + descriptions[i]
                samples[i]['question'] = questions[i] 
        knowledges, knowledges_ids, similarities, ret_info = self.retriever.retrieve(samples, kg=self.kg, offset=turn) \
                                     if self.ret_args.use_retrieval else ([[] for _ in range(len(samples))], [[] for _ in range(len(samples))])
        
       # print(questions)
       # print(knowledges)
       # print(similarities)

        if self.ret_args.score_threshold != -1:

            violation = [n for n,i in enumerate(similarities[0]) if i <self.ret_args.score_threshold]
            if len(violation) == 0:
                 violation_index = len(knowledges[0])
            else: 
         #       print(violation)
                violation_index = violation[0]
         #   print(violation_index)           
            knowledges = [knowledges[0][:violation_index]]
         #   print("new knowledges: ", knowledges)
            knowledges_ids = [knowledges_ids[0][:violation_index]]
        #print("#######")    


        answers = self.lang_model.generate(questions, knowledges, self.glb_args.knowledge_base)
        if self.glb_args.run_no_knowledge:
            no_knowledge_answers = self.lang_model.generate(questions, [], self.glb_args.knowledge_base)
        else:
            no_knowledge_answers = []
        return questions, knowledges, knowledges_ids, answers, no_knowledge_answers, similarities, ret_info

    def edit(self, samples, questions, knowledges, knowledges_ids, answers, turn_id):
        ret_indices, gen_indices, cor_indices = \
            np.where(pred_labels == 0)[0], np.where(pred_labels == 1)[0], np.where(pred_labels == 2)[0]

        # Retrieval
        ret_samples = [samples[index] for index in ret_indices]
        _, ret_knowledges, ret_knowledges_ids, ret_answers, _, similarities, ret_metric_info = self.retrieve_generate(ret_samples, turn=turn_id) \
                                                             if len(ret_samples) != 0 else ([], [], [], [])

        # Generation
        gen_questions = [questions[index] for index in gen_indices]
        gen_knowledges = [knowledges[index] for index in gen_indices]
        get_answers = self.lang_model.generate(gen_questions, gen_knowledges, self.glb_args.knowledge_base) \
                      if len(gen_questions) != 0 else []

        # Merge
        knowledges, knowledges_ids = knowledges[:], knowledges_ids[:]
        for index, ret_index in enumerate(ret_indices):
            knowledges[ret_index], knowledges_ids[ret_index] = ret_knowledges[index], ret_knowledges_ids[index]

        answers = np.array(answers)
        answers[ret_indices] = ret_answers
        answers[gen_indices] = get_answers
        answers = answers.tolist()

        return answers

    def edit_loop(self, samples, questions, knowledges, knowledges_ids, answers):
        for turn_id in range(1, self.glb_args.num_edits+1):
            answers = self.edit(samples, questions, knowledges, knowledges_ids, answers, turn_id)
        return answers

    def eval(self):
        results = {
            'questions': [],
            'knowledges': [],
            'llm_answers': [],
            'similarities': [],
            'no_knowledge_answers': [],
            'ret_metric': []
        }

        for index, batch in enumerate(tqdm(self.qa_data_loaders['test'])):
            if self.glb_args.debug and index == 5: 
                break

            questions, knowledges, knowledges_ids, answers, no_knowledge_answers, similarities, ret_metric_info = self.retrieve_generate(batch)

            if self.glb_args.edit_output == True:
                answers = self.edit_loop(batch, questions, knowledges, knowledges_ids, answers)

            results['questions'].extend(questions)
            results['knowledges'].extend(knowledges)
            results['llm_answers'].extend(answers)
            results['no_knowledge_answers'].extend(no_knowledge_answers)
            results['similarities'].extend(similarities)
            results['ret_metric'].extend([ret_metric_info]) 
        acc_1 = []
        acc_5 = []
        acc_10 = []
        acc_15 = []
        mrr = []

        for i in range(len(results['ret_metric'])):
            if results['ret_metric'][i][1] == []: # empty knowledge
                acc_1.append(0)
                acc_5.append(0)
                acc_10.append(0)
                acc_15.append(0)
                mrr.append(0)
                continue
            acc_1.append(results['ret_metric'][i][1][0])
            acc_5.append(results['ret_metric'][i][2][0])
            acc_10.append(results['ret_metric'][i][3][0])
            mrr.append(results['ret_metric'][i][4][0])
            acc_15.append(results['ret_metric'][i][5][0])
        for i in range(len(results['questions'])):
            self.logger.info("############")
            self.logger.info(f"question: {results['questions'][i]}")
            self.logger.info(f"knowledge: ")
            for k in range(len(results['knowledges'][i])):
#                print(str(k+1)+ ". " + str(results['knowledges'][i][k]))
                self.logger.info(str(k+1)+ ". " + str(results['knowledges'][i][k]))
                self.logger.info(str(k+1)+ ". " + str(results['similarities'][i][k])) 
                    
            self.logger.info("##")
            if self.glb_args.run_no_knowledge:
                self.logger.info("no knowledge answer: ")
                self.logger.info(results['no_knowledge_answers'][i])
            self.logger.info("with knowledge answer: ")
            self.logger.info(results['llm_answers'][i])
            self.logger.info("ground truth answer(s): ")
            for answer_entity in self.qa_datasets['test'][i]["answer_entities"]:
                self.logger.info(answer_entity["mention"])
            self.logger.info("############")
        self.logger.info(f"len of acc_1, acc_5, acc_10, acc_15: {len(acc_1)},{len(acc_5)},{len(acc_10)},{len(acc_15)}")
        self.logger.info(f"acc_1 values: {acc_1}")
        self.logger.info(f"acc_5 values: {acc_5}")
        self.logger.info(f"acc_10 values: {acc_10}")
        self.logger.info(f"acc_15 values: {acc_15}")
        self.logger.info(f"mrr values: {mrr}") 
        self.logger.info(f"retriever_acc_1: {np.mean(acc_1)}")
        self.logger.info(f"retriever_acc_5: {np.mean(acc_5)}")
        self.logger.info(f"retriever_acc_10: {np.mean(acc_10)}")
        self.logger.info(f"retriever_acc_15: {np.mean(acc_15)}")

        self.logger.info(f"retriever_mrr: {np.mean(mrr)}")
        if self.glb_args.debug:
             self.qa_datasets['test'] =  self.qa_datasets['test'][:5]

        self.logger.info(f"generation_accuracy: {accuracy(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases)}")
        self.logger.info(f"generation_f1: {f1(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases)}")
        self.logger.info(f"generation_em: {em(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases)}")
        if self.glb_args.run_no_knowledge: 
            self.logger.info(f"no_knowledge_generation_accuracy: {accuracy(self.qa_datasets['test'], results['no_knowledge_answers'], self.kg, data_args.aliases)}")
            self.logger.info(f"no_knowledge_generation_f1: {f1(self.qa_datasets['test'], results['no_knowledge_answers'], self.kg, data_args.aliases)}")
            self.logger.info(f"no_knowledge_generation_em: {em(self.qa_datasets['test'], results['no_knowledge_answers'], self.kg, data_args.aliases)}")
        #print("The avg number of triples: " + str(np.mean(self.retriever.num_of_triples)))    
            out ={'samples': self.qa_datasets['test'],
             'llm_answers': results['llm_answers'],
             'generation_accuracy': accuracy(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases),
             'generation_f1': f1(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases),
             'generation_em': em(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases),
             'no_knowledge_generation_accuracy': accuracy(self.qa_datasets['test'], results['no_knowledge_answers'], self.kg, data_args.aliases),
             'no_knowledge_generation_f1': f1(self.qa_datasets['test'], results['no_knowledge_answers'], self.kg, data_args.aliases),
             'no_knowledge_generation_em': em(self.qa_datasets['test'], results['no_knowledge_answers'], self.kg, data_args.aliases)
             }
        else:
            out ={'samples': self.qa_datasets['test'],
             'llm_answers': results['llm_answers'],
             'generation_accuracy': accuracy(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases),
             'generation_f1': f1(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases),
             'generation_em': em(self.qa_datasets['test'], results['llm_answers'], self.kg, data_args.aliases)}

        with open(f'{self.output_path}/result.pkl', 'wb') as f:
             pickle.dump(out, f)
      
        if "TimeQuestions" in self.output_path and (not self.glb_args.debug):
            self.logger.info(f"{eval_tq(self.output_path)}")

        if "MultiTQ" in self.output_path and (not self.glb_args.debug):
            self.logger.info(f"{eval_mtq(self.output_path)}")

        return out

    def get_output_path(self):
        output_path = f"./results/{self.data_args.data_name}/{self.lm_args.model_name_or_path.split('/')[-1]}/{self.glb_args.exp_name}"
        if self.glb_args.debug:
            output_path += "_debug"	
        if self.ret_args.no_answer_eval_setting:
            output_path += "_no_answer"
        if self.glb_args.subclassof:
            output_path += "_sem_subclassinquestion"
        if self.glb_args.instanceof:
            output_path += "_sem_instanceinquestion"
        if self.ret_args.attach_subclassof:
            output_path += "_sem_subclassinknowledge"
        if self.ret_args.attach_instanceof:
            output_path += "_sem_instanceinknowledge"
        if self.ret_args.wikidata:
            output_path += "_wikidata"
        if self.ret_args.add_relevancy_score:
            output_path += "_w_rel_scores"
        if self.ret_args.order_facts:
            output_path += "_ordered"
            if self.ret_args.recent_to_oldest:
                output_path += "RO"
            else:
                output_path += "OR"
        if self.ret_args.add_durations:
            output_path += "_w_durations"
        if self.ret_args.use_temporal_relevancy_scores:
            output_path += "temporal_relevancy_scores"
        if self.ret_args.use_positional_encoding:
            output_path += "positional_encoding"      

        if not os.path.exists(output_path): os.makedirs(output_path)
        print(f"OUTPUT_PATH: {output_path}")
        return output_path

if __name__ == "__main__":

    parser = HfArgumentParser((GlobalArguments, DataArguments, LanguageModelArguments, RetrieverArguments))

    glb_args, data_args, lm_args, ret_args  = parser.parse_args_into_dataclasses()

    glb_args.device = lm_args.device = ret_args.device = torch.device("cuda" if torch.cuda.is_available() and not glb_args.no_cuda else "cpu")
    print(glb_args.device)
    glb_args.n_gpu = lm_args.n_gpu = ret_args.n_gpu = 0 if glb_args.no_cuda else torch.cuda.device_count()
    ret_args.cache_dir = lm_args.cache_dir
    ret_args.device_map = lm_args.device_map

#    data_args.data_splits = ['train', 'dev', 'test'] \
#        if data_args.data_name in ['Mintaka'] \
#        else ['train', 'test']
    data_args.data_splits = ['test']

    runner = Runner(glb_args, data_args, lm_args, ret_args)
    runner.eval()

    import shutil
    shutil.move("preds.pkl", runner.output_path)
    shutil.move("prompts.txt", runner.output_path)
    if glb_args.stop:   import pdb; pdb.set_trace()
