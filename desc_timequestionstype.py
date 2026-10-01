import requests
from requests import utils
import rdflib
from bs4 import BeautifulSoup
import wikipediaapi as w
import json
import nltk
import time
import random
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib.request
import urllib.error
import urllib.parse
import os

if os.path.exists("timequestions_type_qid2wikilabel.json"):
    with open("timequestions_type_qid2wikilabel.json", "r", encoding="utf-8") as f:
        timequestions_type_qid2wikilabel = json.load(f)
else:
    timequestions_type_qid2wikilabel= {}
    with open("timequestions_type_qid2wikilabel.json", "w", encoding="utf-8") as f:
        json.dump(timequestions_type_qid2wikilabel, f, ensure_ascii=False, indent=4)

if os.path.exists("timequestions_id2subclass.json"):
    with open("timequestions_id2subclass.json", "r", encoding="utf-8") as f:
        timequestions_id2subclass= json.load(f)
else:
    timequestions_id2subclass = {}
    with open("timequestions_id2subclass.json", "w", encoding="utf-8") as f:
        json.dump(timequestions_id2subclass, f, ensure_ascii=False, indent=4)

if os.path.exists("timequestions_id2instanceof.json"):
    with open("timequestions_id2instanceof.json", "r", encoding="utf-8") as f:
        timequestions_id2instanceof= json.load(f)
else:
    timequestions_id2instanceof = {}
    with open("timequestions_id2instanceof.json", "w", encoding="utf-8") as f:
        json.dump(timequestions_id2instanceof, f, ensure_ascii=False, indent=4)


def get_wikipedia_url_from_wikidata_id(wikidata_id, lang='en', debug=False):
    url = (
        'https://www.wikidata.org/w/api.php'
        '?action=wbgetentities'
        '&props=sitelinks/urls'
        f'&ids={wikidata_id}'
        '&format=json')
    json_response = requests.get(url).json()
    if debug: print(wikidata_id, url, json_response)

    entities = json_response.get('entities')
    if entities:
        entity = entities.get(wikidata_id)
        if entity:
            sitelinks = entity.get('sitelinks')
            if sitelinks:
                if lang:
                    # filter only the specified language
                    sitelink = sitelinks.get(f'{lang}wiki')
                    if sitelink:
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            return requests.utils.unquote(wiki_url)
                else:
                    # return all of the urls
                    wiki_urls = {}
                    for key, sitelink in sitelinks.items():
                        wiki_url = sitelink.get('url')
                        if wiki_url:
                            wiki_urls[key] = requests.utils.unquote(wiki_url)
                    return wiki_urls
    return None

def get_description_from_wikipedia_url(wikipedia_url):
    description = ""
    if wikipedia_url != None:
        respond = requests.get(wikipedia_url)
        soup = BeautifulSoup(respond.text, features="lxml")

        # https://stackoverflow.com/questions/67605758/how-to-match-and-remove-wikipedia-refences-with-python-and-re
        for tag in soup.find_all(class_="reference"):
            tag.decompose()

        l = soup.find_all('p')

        description = BeautifulSoup(str(l[0]), features="lxml").get_text()
        if len(l) > 1:
            gfg = BeautifulSoup(str(l[1]), features="lxml")
            description += gfg.get_text()
            if len(l) > 2:
                gfg = BeautifulSoup(str(l[2]), features="lxml")
                description += gfg.get_text()

    description = description.replace('\n',"")
    return description

def get_labels_from_wikidata_ids(wikidata_ids,sparql=None):
    #g = rdflib.ConjunctiveGraph('SPARQLStore')
    #g.open('https://query.wikidata.org/sparql')
     
    #sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    labels = []
    for wikidata_id in wikidata_ids:
        
        if wikidata_id in timequestions_type_qid2wikilabel.keys():
            labels.append(timequestions_type_qid2wikilabel[wikidata_id])
            
        else:
            query="""
       		PREFIX wd: <http://www.wikidata.org/entity/>
        	SELECT *
        	WHERE {
        	wd:""" + wikidata_id + """ rdfs:label ?label .
        	FILTER (langMatches(lang(?label), "EN"))}
        	LIMIT 1
       		"""
            label = None
    
#        sparql.setQuery(query)
#        sparql.setReturnFormat(JSON)
#        results = sparql.query().convert()
 
            results= run_sparql_query(sparql, query)
            print(results)
     #   print(results)
            if results == None or len(results["results"]["bindings"])==0:
                labels.append("")
            else:
                label = results["results"]["bindings"][0]["label"]["value"].split("/")[-1]
                if len(label) == 0:
                    labels.append("")
                else:
                    labels.append(label)
                    timequestions_type_qid2wikilabel[wikidata_id]=label
                    with open("timequestions_type_qid2wikilabel.json", "w", encoding="utf-8") as f:
                        json.dump(timequestions_type_qid2wikilabel, f, ensure_ascii=False, indent=4)
#    qresult = g.query(query)
#    for row in qresult:
#        label = row[0]
    
    #print(labels)

    return labels

def get_description_from_label(label):
    description = ""
    wiki = w.Wikipedia('en')
    page = wiki.page(label)

    if page.exists():
        description = page.summary

    return description

def get_wikipedia_id_from_wikidata_url(wikipedia_url):
    response = requests.get(url=wikipedia_url)
    soup = BeautifulSoup(response.text, 'lxml')
    wikipedia_id = soup.find('li', {'id' : 't-wikibase'})
    wikipedia_id = wikipedia_id.a['href'].rsplit('/')[-1]
    return wikipedia_id

def get_description_from_wikiID(wikiID):
    respond = requests.get("https://www.wikidata.org/wiki/" + wikiID)
    soup = BeautifulSoup(respond.text, features="lxml")
    for tag in soup.find_all(class_="reference"):
        tag.decompose()
    if len(str(soup).split("name=\"description\"")) < 2:
        description = "" # No description
    else:
        description = str(soup).split("name=\"description\"")[1].split("<meta content=")[1].split("\"")[1]
    return description

def get_title_from_wikiID(wikiID):
    response = requests.get("https://www.wikidata.org/wiki/" + wikiID)
    soup = BeautifulSoup(response.text, 'lxml')
    title = response.text.split("<title>")[1].split(" -")[0]
    return title

def get_first_sentence(text):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return ""
    return sentences[0]

def get_description_unified_method(wikiID):
    if wikiID is None:
        return ""
        
    url = get_wikipedia_url_from_wikidata_id(wikiID)

    # 1st method
    if url is not None:
        description = get_description_from_wikipedia_url(get_wikipedia_url_from_wikidata_id(wikiID))

    # 2nd method
    if url is None or (len(description) == 0 or len(description) == 1):
        print("The 1st method failed for " + wikiID + "\n")

        description = get_description_from_wikiID(wikiID)

        if len(description) == 0 or len(description) == 1:
            print("The 2nd method failed for " + wikiID + "\n")
    return get_first_sentence(description)

def from_entities_to_desc_json(data_path):
    with open(data_path + '/entities_with_descriptions.jsonl', 'w') as outfile:
        with open(data_path + '/entities.jsonl') as f:
            data = [json.loads(line) for line in f]

        for item in data:
            wikiID = item['entity']
            item["description"] = unified_method(wikiID)
            json.dump(item, outfile)
            outfile.write('\n')

def get_desc_of_list_of_IDs(lst):
    return ' '.join([get_description_unified_method(item) for item in lst])

def get_instanceof_from_qid(wikidata_id,sparql=None):
    if wikidata_id == None or (wikidata_id[0]!= "Q" or not wikidata_id[1:].isnumeric()):
        return ""
    if wikidata_id in timequestions_id2instanceof.keys():
        print("Found in qid2types")
        return timequestions_id2instanceof[wikidata_id]
    #g = rdflib.ConjunctiveGraph('SPARQLStore')
    #g.open('https://query.wikidata.org/sparql')

    #query="""
    #    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    #    PREFIX wd: <http://www.wikidata.org/entity/>
    #    SELECT *
    #    WHERE {
    #    wd:""" + wikidata_id + """ wdt:P31 ?y .
    #    }
    #"""

    query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?y
    WHERE {{
    wd:{wikidata_id} wdt:P31 ?y .
    }}
    LIMIT 1
    """.format(wikidata_id=wikidata_id)
    #print(query)

    results = run_sparql_query(sparql, query)

    #sparql.setQuery(query)
    #sparql.setReturnFormat(JSON)
    #results = sparql.query().convert()
    #print(results)

    if results == None:
        return ""
    qids = [b["y"]["value"].split("/")[-1] for b in results["results"]["bindings"]]    
    if len(qids) == 0:
        return ""

    #label = None
    #qresult = g.query(query)
    #for row in qresult:  
    #    label = row[0]
    #if label == None:
    #    return ""
    #qid = label.split("/")[-1]
    
    labels = get_labels_from_wikidata_ids(qids,sparql)
    #print(labels)
    timequestions_id2instanceof[wikidata_id] = labels
    with open("timequestions_id2instanceof.json", "w", encoding="utf-8") as f:
        json.dump(timequestions_id2instanceof, f, ensure_ascii=False, indent=4)
    print(wikidata_id+"->"+str(labels))
    return labels

def get_instanceof_list_of_IDs(lst,sparql=None):
    return [get_instanceof_from_qid(item,sparql) for item in lst]

def get_subclassof_from_qid(wikidata_id, sparql=None):
    if wikidata_id == None or (wikidata_id[0]!= "Q" or not wikidata_id[1:].isnumeric()):
        return ""
    if wikidata_id in timequestions_id2subclass.keys():
        print("Found in qid2types")
        return timequestions_id2subclass[wikidata_id]
    #g = rdflib.ConjunctiveGraph('SPARQLStore')
    #g.open('https://query.wikidata.org/sparql')

    #sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

    #query="""
    #    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    #    PREFIX wd: <http://www.wikidata.org/entity/>
    #    SELECT *
    #    WHERE {
    #    wd:""" + wikidata_id + """ wdt:279 ?y .
    #    }LIMIT 1
    #"""

    query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>

    SELECT ?y
    WHERE {{
    wd:{wikidata_id} wdt:P279 ?y .
    }}
    LIMIT 1
    """.format(wikidata_id=wikidata_id)

    #label = None
    #qresult = g.query(query)
    #for row in qresult:
    #    label = row[0]
    #if label == None:
    #    return ""
    #qid = label.split("/")[-1]
    
    results = run_sparql_query(sparql, query) 
    #sparql.setQuery(query)
    #sparql.setReturnFormat(JSON)
    #results = sparql.query().convert()
    #print(results)
    print(results)
    if results == None:
        return ""
    qids = [b["y"]["value"].split("/")[-1] for b in results["results"]["bindings"]]
    if len(qids) == 0:
        return ""
    labels = get_labels_from_wikidata_ids(qids,sparql)
    timequestions_id2subclass[wikidata_id] = labels
    with open("timequestions_id2subclass.json", "w", encoding="utf-8") as f:
        json.dump(timequestions_id2subclass, f, ensure_ascii=False, indent=4)
    print(wikidata_id+"->"+str(labels))
    return labels

def get_subclassof_list_of_IDs(lst, sparql=None):
    return [get_subclassof_from_qid(item,sparql) for item in lst]

def run_sparql_query(sparql, query: str, max_retries: int = 20):
    for attempt in range(max_retries):
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

        except urllib.error.HTTPError as e:
            if e.code in [429, 500, 502, 503, 504]:
                #wait = (2 ** attempt) + random.uniform(0, 1)
                wait = (attempt+1) * 5 
                print(f"HTTP {e.code}. Retrying in {wait:.2f} seconds...")
                time.sleep(wait)
            else:
                raise

        except urllib.error.URLError as e:
            #wait = (2 ** attempt) + random.uniform(0, 1)
            wait = (attempt+1) * 5
            print(f"Network error: {e}. Retrying in {wait:.2f} seconds...")
            time.sleep(wait)
    return results

all_entities = []
input_file = "../data/timequestions/kg/wd_id2entity_text.txt"

with open(input_file, "r", encoding="utf-8") as fin:
    for line in fin:
        all_entities.append(line.strip().split("\t")[0])
all_entities = list(all_entities)[62723:]
print(len(all_entities))
print(all_entities[:5])
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

get_instanceof_list_of_IDs(all_entities,sparql)

#get_subclassof_list_of_IDs(all_entities,sparql)
