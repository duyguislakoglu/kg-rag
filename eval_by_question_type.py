import pickle
import json

### TimeQuestions
def eval_tq(output_path):
    with open(f'preds.pkl', 'rb') as f:
        data = pickle.load(f)
        #print("The length of the data: ")
        #print(len(data))

    with open('../data/timequestions/questions/test.json') as f:
        original_questions = json.load(f)
    for (i, question) in enumerate(original_questions):
        if question["Answer"] ==[]:
            #print(question['Temporal question type'])
            original_questions.pop(i)
    for i in range(len(data)):
       
        if original_questions[i]["Question"] != data[i]["question"]:
            print(original_questions[i]["Question"])
            print(data[i]["question"])
            print("####")

    acc = 0
    total = {'Explicit':0, 'Ordinal':0, "Implicit":0, "Temp.Ans":0}
    answers ={'Explicit':0, 'Ordinal':0, "Implicit":0, "Temp.Ans":0}
    for i in range(len(data)):

        for typ in original_questions[i]['Temporal question type']:
            total[typ] += 1
        if data[i]["accuracy"] ==1:
            answers[typ] += 1
            acc += 1
    acc = acc / len(data)
    explicit = answers['Explicit']/total['Explicit']
    implicit = answers['Implicit']/total['Implicit']
    temp_ans = answers['Temp.Ans']/total['Temp.Ans']
    ordinal = answers['Ordinal']/total['Ordinal']

    #print(str(acc*100)[:5]+" & "+ str(explicit*100)[:5] +" & "+ str(implicit*100)[:5] +" & "+  str(temp_ans*100)[:5] +" & "+  str(ordinal*100)[:5])
    return {"Acc": str(acc*100)[:5], "explicit": str(explicit*100)[:5], "implicit": str(implicit*100)[:5], "temp_answer":str(temp_ans*100)[:5] , "ordinal":str(ordinal*100)[:5]}

### MultiTQ
def eval_mtq(output_path):
    with open(f'preds.pkl', 'rb') as f:
         preds = pickle.load(f)
    #print(preds[0])

    with open('../data/multitq/questions/test.json') as f:
        original_questions = json.load(f)
    #print(original_questions[0])

    for i in range(len(preds)):
        #print(preds[i]["gold_entities"])
        #print(original_questions[i]["answers"])
        if preds[i]["gold_entities"] != set(original_questions[i]["answers"]):
            print("Answers do not match.")
        #print("#######")

    acc = 0
    counts = {"entity":0, "time":0, "Multiple":0, "Single":0}
    for i in range(len(preds)):
        if (preds[i]['accuracy']==1):
            acc +=1
            counts[original_questions[i]["answer_type"]] += 1
            counts[original_questions[i]["qlabel"]] += 1

    acc = acc/len(preds)
    sing_acc = counts["Single"] / [q["qlabel"] for q in original_questions].count("Single")
    mult_acc = counts["Multiple"] / [q["qlabel"] for q in original_questions].count("Multiple")
    ent_acc = counts["entity"] / [q["answer_type"] for q in original_questions].count("entity")
    time_acc = counts["time"] / [q["answer_type"] for q in original_questions].count("time")
    #print("Acc, single, multi, entity, time")
    return {"Acc": acc, "single": sing_acc, "multi": mult_acc, "entity": ent_acc, "time":time_acc}

#print("Semantic retrieval only")
#print(eval_mtq("results/MultiTQ/multitqtrainqa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit/vector_db/"))

#print("Temporal retrieverr")
#print(eval_mtq("results/MultiTQ/multitqtrainqa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit/vector_dbtemporal_relevancy_scores/"))


#print("RO")
#print(eval_mtq("results/MultiTQ/multitqtrainqa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit/vector_db_orderedRO/"))

