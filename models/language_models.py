import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from transformers import AutoModelForCausalLM, pipeline
#from unsloth import FastLanguageModel
#from peft import AutoPeftModelForCausalLM
import re
import json
from together import Together

class LanguageModel(object):
    def __init__(self, args):
        super(LanguageModel, self).__init__()

        self.args = args

        self.config = self.get_config()
        self.tokenizer = self.get_tokenizer()
        self.model = self.get_model()
        if "Llama-3.3-70B" in self.args.model_name_or_path:
            self.client = Together(api_key="")

    def get_config(self):

        if "duyguislakoglu/full_only_chronosense_Meta-Llama-3.1-8B-Instruct" in self.args.model_name_or_path or "Llama-3.3-70B" in self.args.model_name_or_path:
            return None

        config = AutoConfig.from_pretrained(
            self.args.model_name_or_path,
            cache_dir = self.args.cache_dir,
        )
        return config

    def get_tokenizer(self):
        if "Llama-3.3-70B" in self.args.model_name_or_path:
            return None
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            cache_dir = self.args.cache_dir,
            resume_download = True
        )

        if self.args.model_name_or_path in ["Mistral-7B-Instruct-v0.2"]:
           tokenizer.pad_token = tokenizer.eos_token


        return tokenizer

    def get_model(self):
        if "Llama-3.3-70B" in self.args.model_name_or_path:
            return None
        if self.args.decoder_only:
            if self.args.model_name_or_path == "meta-llama/Llama-3.1-8B-Instruct":
                pipe = pipeline(
                "text-generation",
                model=self.args.model_name_or_path,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="cuda")
                return pipe
            if self.args.model_name_or_path == "google/gemma-2-9b-it":
                pipe = pipeline(
                    "text-generation",
                    model=self.args.model_name_or_path,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device="cuda",  # replace with "mps" to run on a Mac device
                )
                return pipe
            if self.args.model_name_or_path == "meta-llama/Llama-2-7b-chat-hf":
                return pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf", device="cuda")

            if self.args.model_name_or_path == "meta-llama/Llama-3.2-3B-Instruct":
                return pipeline("text-generation",model=self.args.model_name_or_path,torch_dtype=torch.bfloat16,device_map="cuda")

            if "duyguislakoglu/" in self.args.model_name_or_path:
              #  model, tokenizer = FastLanguageModel.from_pretrained(
              #  model_name = self.args.model_name_or_path, # YOUR MODEL YOU USED FOR TRAINING
              #  max_seq_length = max_seq_length,
              #  dtype = dtype,
              #  load_in_4bit = load_in_4bit, )

              #  FastLanguageModel.for_inference(model)
               from peft import AutoPeftModelForCausalLM

               model = AutoPeftModelForCausalLM.from_pretrained(self.args.model_name_or_path)

               return model


            model = AutoModelForCausalLM.from_pretrained(self.args.model_name_or_path)
            model.to("cuda")
            return model
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.args.model_name_or_path,
            config = self.config,
            cache_dir = self.args.cache_dir,
            torch_dtype = torch.float16 if self.args.fp16 else None,
            device_map = self.args.device_map if self.args.n_gpu >= 0 else None,
            resume_download = True,
            token = ""
            )

        return model

    def verbalize_knowledge(self, knowledge, knowledge_base):
        if not knowledge:
            return ''

        if "duyguislakoglu/" in self.args.model_name_or_path:
            string = ''
            for triplet in knowledge:
                string += f"['{triplet[0]}'] "
            string = string[:-1]
            return string

        if len(knowledge[0]) == 3 and knowledge[0][1] == "" and knowledge[0][2] == "":
            string = 'Based on the historical facts, please answer the given question. Please keep the answer as simple as possible and return all the possible answers as a list.\nHistorical facts:'
            #string = 'Based on the historical facts, please answer the given question. Choose only one of the triples and use the following format: "<<answer triple>>". \nHistorical facts:'

            #string = 'Based on the historical facts, please answer the given question by selecting exactly one fact from the question. Output only in the following format: {"answer_triple": selected triple}. \nHistorical facts:'

            string += str([triplet[0] for triplet in knowledge])
            string += '\n'
            return string

        string = 'Below are facts in the form of the triple meaningful to answer the question. \n'

        for triplet in knowledge:
            string += f'({triplet[0]}, {triplet[1]}, {triplet[2]}), '
        string = string[:-2]
        string += '\n'
        return string

    def generate(self, questions, knowledges, knowledge_base='kg'):

        prompts = []
# (ordered from the oldest to recent)
# (ordered from most recent to oldest)
        alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the question based on the given facts.

### Input: {}

### Response:
"""

        chronosense_prompt = """
        The following are examples that illustrate different time reasoning tasks:

        # End timepoint
        Question: 'Event A' started in 1618. 'Event A' took exactly 18 years. Did 'Event A' end in the year 1636?
        Answer: True.

        # Intermediate timepoint
        Question: 'Event A' occurred between year 1635 and year 1654. Was 'Event A' happening in the year 1644?
        Answer: True.

        # Next occurrence
        Question: 'Event A' first occurred in year 1161. 'Event A' occurs every 38 years. Did 'Event A' occur again in the year 1199?
        Answer: True.

        # Before
        Question: 'Event A' occurred between year 1950 and year 1951. 'Event B' occurred between year 1992 and year 1993. Did 'Event A' occur before 'Event B' without any overlap between the two events?
        Answer: True.

        # After
        Question: 'Event A' occurred between year 1580 and year 1700. 'Event B' occurred between year 1342 and year 1343. Did 'Event A' occur after 'Event B' without any overlap between the two events?
        Answer: True.

        # Equals
        Question: 'Event A' occurred between year 2019 and year 2020. 'Event B' occurred between year 2019 and year 2020. Did 'Event A' begin and end in the same years as 'Event B'?
        Answer: True.

        # During
        Question: 'Event A' occurred between year 2013 and year 2014. 'Event B' occurred between year 2009 and year 2020. Did 'Event A' begin after 'Event B' began and end before 'Event B' ended, entirely contained within 'Event B'?
        Answer: True.

        # Contains
        Question: 'Event A' occurred between year 1986 and year 1995. 'Event B' occurred between year 1990 and year 1991. Did 'Event A' begin before 'Event B' began and end after 'Event B' ended, entirely encompassing 'Event B'?
        Answer: True.

        # Starts
        Question: 'Event A' occurred between year 2004 and year 2005. 'Event B' occurred between year 2004 and year 2008. Did 'Event A' begin in the same year as 'Event B', but end before 'Event B' ended?
        Answer: True.

        # Started by
        Question: 'Event A' occurred between year 2010 and year 2016. 'Event B' occurred between year 2010 and year 2011. Did 'Event A' begin in the same year as 'Event B', but end after 'Event B' ended?
        Answer: True.

        # Meets
        Question: 'Event A' occurred between year 1940 and year 1942. 'Event B' occurred between year 1942 and year 1945. Did 'Event A' end in the same year as 'Event B' began?
        Answer: True.

        # Met_by
        Question: 'Event A' occurred between year 1880 and year 1894. 'Event B' occurred between year 1868 and year 1880. Did 'Event A' begin in the same year as 'Event B' ended?
        Answer: True.

        # Overlaps
        Question: 'Event A' occurred between year 2016 and year 2022. 'Event B' occurred between year 2019 and year 2024. Did 'Event A' begin before 'Event B' began and end before 'Event B' ended, with some overlap?
        Answer: True.

        # Overlapped by
        Question: 'Event A' occurred between year 1822 and year 1844. 'Event B' occurred between year 1807 and year 1842. Did 'Event A' begin after 'Event B' began and end after 'Event B' ended, with some overlap?
        Answer: True.

        # Finishes
        Question: 'Event A' occurred between year 2011 and year 2012. 'Event B' occurred between year 1972 and year 2012. Did 'Event A' begin after 'Event B' began and end in the same year as 'Event B'?
        Answer: True.

        # Finished_by
        Question: 'Event A' occurred between year 1936 and year 1938. 'Event B' occurred between year 1937 and year 1938. Did 'Event A' begin before 'Event B' began and end in the same year as 'Event B'?
        Answer: True.

        """
        questions = [question+"?" if (question[-1]!="?" and question[-1]!=".") else question for question in questions]
        #if self.args.chronosense_incontext:
        #questions = [chronosense_prompt + question for question in questions]

        if "duyguislakoglu/" in self.args.model_name_or_path:

            if len(knowledges) == 0:

                prompts = [
                        alpaca_prompt.format(f'{question}')  \
                        for question in questions]
            else:

	            #prompts = [alpaca_prompt.format(f'{self.verbalize_knowledge(knowledge, knowledge_base)}\n{self.args.question_prefix}{question}')  \
                #        for (question, knowledge) in zip(questions, knowledges)]

                if self.args.chronosense_incontext:
                    prompts = [
                    alpaca_prompt.format(f'{chronosense_prompt}', f'{self.verbalize_knowledge(knowledge, knowledge_base)}\n{self.args.question_prefix}{question}') \
                    for (question, knowledge) in zip(questions, knowledges)]
                else:
                    prompts = [
                    alpaca_prompt.format(f'{self.verbalize_knowledge(knowledge, knowledge_base)}\n{self.args.question_prefix}{question}') \
                    for (question, knowledge) in zip(questions, knowledges)]

        elif "unsloth" in self.args.model_name_or_path or  "Llama-3.3-70B" in self.args.model_name_or_path:
            if len(knowledges) ==0:
                prompts = [
                    f'{self.args.question_prefix}{question}' for question in questions]
            else:
                prompts = [
        	        f'{self.verbalize_knowledge(knowledge, knowledge_base)}{self.args.question_prefix}{question}' \

            	     for (question, knowledge) in zip(questions, knowledges)
            	]

                if self.args.chronosense_incontext:
                    prompts = [
                        f'{self.verbalize_knowledge(knowledge, knowledge_base)}{chronosense_prompt}\n{self.args.question_prefix}{question}' \
                    for (question, knowledge) in zip(questions, knowledges)
                                                                                 ]


        elif len(knowledges) == 0:
            prompts = [
                f'{self.args.question_prefix}{question}\n{self.args.question_postfix}' for question in questions]
        else:
            prompts = [
                f'{self.verbalize_knowledge(knowledge, knowledge_base)}{self.args.question_prefix}{question}\n{self.args.question_postfix}' \

                 for (question, knowledge) in zip(questions, knowledges)
            ]
        print(prompts)
        if self.args.decoder_only:
            if self.args.model_name_or_path in ["meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.1-8B-Instruct","google/gemma-2-9b-it", "meta-llama/Llama-2-7b-chat-hf"]:
                messages = [{"role": "user", "content": prompts[0]}]
                outputs = [self.model(messages,max_new_tokens=self.args.max_new_token)[0]["generated_text"][-1]["content"].strip()]
                print(outputs)


            elif "duyguislakoglu/" in self.args.model_name_or_path:
                inputs = self.tokenizer([prompts[0]], return_tensors = "pt").to("cuda")
                outputs = self.model.generate(**inputs, max_new_tokens = self.args.max_new_token, use_cache = True)
                outputs = self.tokenizer.batch_decode(outputs)
                print(outputs[0])
                outputs = [outputs[0].split("Response:\n", 1)[1].strip()[:-len("<|eot_id|>")]]

                print(outputs)
            elif "Llama-3.3-70B" in self.args.model_name_or_path:
                response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[
                    {
                    "role": "user",
                    "content": prompts[0]
                    }
                ]
                )
                outputs = [response.choices[0].message.content]
                print(outputs)
            else:
                messages = [{"role": "user", "content": prompts[0]}]
                encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
                model_inputs = encodeds.to("cuda")
                generated_ids = self.model.generate(model_inputs, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=self.args.max_new_token, do_sample=True)

                outputs = self.tokenizer.batch_decode(generated_ids[:, model_inputs.shape[1]:])

                print(outputs)
                #match = re.search(r"<<\s*(.*?)\s*>>", outputs[0])
                #final_answer = match.group(1) if match else "unknown"
                #outputs[0] = final_answer
                #print(outputs)
                #data = json.loads(outputs[0])
                #answer_id = data["answer_triple"]
                #match = re.search(r'"answer_triple"\s*:\s*"([^"]*)"', outputs[0])
                #correct one
                #matches = re.findall(r"\{(.*?)\}", outputs[0])
                #outputs[0] = matches
                #print(outputs)

        else:
            input_tokens = self.tokenizer(prompts, max_length=self.args.max_source_length, padding='longest', truncation=True, return_tensors='pt').to(self.args.device)
            generated_tokens = self.model.generate(**input_tokens,
                max_new_tokens=self.args.max_target_length,
                do_sample=True,
                top_k=40,
                temperature=0.5,
            )

            outputs = self.tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True

            )
        return outputs
