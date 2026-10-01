#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --add_durations > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --add_relevancy_score > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --order_facts > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --add_durations --order_facts > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --add_relevancy_score --order_facts > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --order_facts --recent_to_oldest False > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --order_facts --recent_to_oldest False  --add_durations > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --order_facts --recent_to_oldest False --add_relevancy_score  > prompts.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --use_temporal_relevancy_scores > temporal1.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --add_durations --use_temporal_relevancy_scores > temporal2.txt

#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/qa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --add_relevancy_score --use_temporal_relevancy_scores > temporal3.txt

# MultiTQ
#python main.py --data_name MultiTQ --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/multitqtrainqa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only > multi-vanilla.txt
#python main.py --data_name MultiTQ --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/multitqtrainqa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --use_temporal_relevancy_scores --multitq_data > multi-temporal.txt
#python main.py --data_name MultiTQ --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/multitqtrainqa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only  --order_facts --multitq_data > multitq-RO.txt
#python main.py --data_name MultiTQ --data_type TKGQA --knowledge_base kg --exp_name vector_db --model_name duyguislakoglu/multitqtrainqa_and_entireChronoSense_Meta-Llama-3.1-8B-Instruct-bnb-4bit --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only  --order_facts --multitq_data --recent_to_oldest False > multitq-OR.txt

# TogetherAI
#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name togetherai-70b --model_name Llama-3.3-70B --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only > together-ai-prompts.txt
#python main.py --data_name TimeQuestions --data_type TKGQA --knowledge_base kg --exp_name togetherai-70b --model_name Llama-3.3-70B --batch_size 1 --edit_output False --retriever_name vector_db --decoder-only --use_temporal_relevancy_scores > together-ai-temporal-prompts.txt
