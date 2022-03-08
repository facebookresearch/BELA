import argparse
import json
import pickle
import random
import os
import json
import glob
import torch
from nltk.tokenize import sent_tokenize
from bela.datamodule.joint_el_datamodule import EntityCatalogue


'''def convert2blink_old(title2wikidataID, f_out, dataset_path):
    with open(dataset_path) as f:
        for line in f:
            line = json.loads(line)
            output = {"context_left": '', "mention": '', "context_right": '',"mention": '', "query_id": "", "label_id": ""}
            line['id'] = subset + "_" + line['id']
            output["query_id"] = line['id']
            for ent in line["entities"]:
                mention = line['text'][ent['offset']:ent['offset']+ent['length']]
                context_left = line['text'][:ent['offset']]
                context_right = line['text'][ent['offset'] + ent['length']:]
                output["context_left"] = context_left
                output["mention"] = mention
                output["context_right"] = context_right
                if ent['entity_id'] in title2wikidataID:
                    output["label_id"] = int(title2wikidataID[ent['entity_id']][1:])
                    f_out.write(json.dumps(output))
                    f_out.write('\n')'''

def generate_data_indexer(base_path_entities, cluster_output_path, embeddings_dir, data_path):

    ent_catalogue_idx_path = base_path_entities + "en_bert_ent_idx.txt"
    novel_entity_idx_path = base_path_entities + "novel_entities_filtered.jsonl"
    ent_catalogue_known = EntityCatalogue(ent_catalogue_idx_path, None)
    ent_catalogue = EntityCatalogue(ent_catalogue_idx_path, novel_entity_idx_path, True)

    cluster2idcs = {}                                                                                                                                                          
    raw_text = {}
    counts = {}
    #entities = set()
    cluster_id = set()
    with open(cluster_output_path) as f:                                                                            
        for line in f:                                                                                                                                                         
            entity, cluster, idx = line.strip().split(", ")
            # entities.add(entity)
            
            ent_name = ent_catalogue.idx_reverse[int(entity)]
            if ent_name in counts:
                counts[ent_name] +=1
            else:
                counts[ent_name] = 1
            if cluster not in cluster2idcs:                                                                                                                                    
                cluster2idcs[cluster] = {}                                                                                                                                     
                cluster2idcs[cluster]["entities"] = [entity]                                                                                                                   
                cluster2idcs[cluster]["idcs"] = [idx]                                                                                                                          
            else:                                                                                                                                                              
                cluster2idcs[cluster]["entities"].append(entity)                                                                                                               
                cluster2idcs[cluster]["idcs"].append(idx) 
            raw_text[idx] = "" 
            
    entities = []
    embeddings_path_list = glob.glob(embeddings_dir + '*.t7')
    for embedding_path in sorted(embeddings_path_list):
        embeddings_buffer = torch.load(embedding_path, map_location='cpu')
        for embedding_batch in embeddings_buffer:
            for embedding in embedding_batch:
                entity, embedding = embedding[0], embedding[1:]
                entity = ent_catalogue.idx_reverse[int(entity)]
                entities.append(entity)

    with open(data_path) as f:
        i = 0
        for line in f:
            line = json.loads(line)
            gold_entities = line['entities_raw']
            sentences = sent_tokenize(line["text_raw"])
            sentence_offset = 0
            character_offset = len(sentences[sentence_offset])
            for entity in gold_entities:
                if entity["offset"] > character_offset:
                    sentence_offset += 1
                    character_offset += len(sentences[sentence_offset])
                    character_offset += 1
                if entity['entity_id'] not in ent_catalogue:
                    continue
                if entity['entity_id']!=entities[i]:
                    # print(line['data_example_id'], i, entity['entity_id'], entities[i])
                    continue            
                i +=1

                if entity['entity_id'] in ent_catalogue_known:
                    continue
                if str(i) in raw_text:
                    raw_text[str(i)] = sentences[sentence_offset]
                    cluster_id.add(line['data_example_id'])
            if i>=len(entities):
                break

    with open("data/blink/novel_entities_cluster.jsonl", 'w') as f:
        for cluster in cluster2idcs:
            ent_id = max(cluster2idcs[cluster]["entities"], key=cluster2idcs[cluster]["entities"].count)
            ent_name = ent_catalogue.idx_reverse[int(ent_id)]
            description = ''
            for i in cluster2idcs[cluster]['idcs']:
                description += raw_text[i]
                print(raw_text[i])
                description += " "
            out_dict = {"title": ent_name, "entity": ent_name, "text": description.strip()}
            f.write(json.dumps(out_dict))
            f.write("\n")

def generate_data_train_indexer(dataset_path, ouput_base_path):

    ent_catalogue_idx_path = "/data/home/kassner/BELA/data/blink/en_bert_ent_idx.txt"
    novel_entity_idx_path = "/data/home/kassner/BELA/data/blink/novel_entities_filtered.jsonl"
    ent_catalogue_known = EntityCatalogue(ent_catalogue_idx_path, None)
    ent_catalogue = EntityCatalogue(ent_catalogue_idx_path, novel_entity_idx_path, True)

    cluster2idcs = {}                                                                                                                                                          
    raw_text = {}
    counts = {}
    #entities = set()
    cluster_id = set()
    with open('output_clustering/2022-01-24-084433_greedy_t2_known_0.89_200000_10000.txt') as f:                                                                            
        for line in f:                                                                                                                                                         
            entity, cluster, idx = line.strip().split(", ")
            # entities.add(entity)
            
            ent_name = ent_catalogue.idx_reverse[int(entity)]
            if ent_name in counts:
                counts[ent_name] +=1
            else:
                counts[ent_name] = 1
            if cluster not in cluster2idcs:                                                                                                                                    
                cluster2idcs[cluster] = {}                                                                                                                                     
                cluster2idcs[cluster]["entities"] = [entity]                                                                                                                   
                cluster2idcs[cluster]["idcs"] = [idx]                                                                                                                          
            else:                                                                                                                                                              
                cluster2idcs[cluster]["entities"].append(entity)                                                                                                               
                cluster2idcs[cluster]["idcs"].append(idx) 
            raw_text[idx] = "" 
            
    entities = []
    input_path = '/fsx/kassner/hydra_outputs/main/2022-01-24-084433/0/*.t7'
    embeddings_path_list = glob.glob(input_path)
    for embedding_path in sorted(embeddings_path_list):
        embeddings_buffer = torch.load(embedding_path, map_location='cpu')
        for embedding_batch in embeddings_buffer:
            for embedding in embedding_batch:
                entity, embedding = embedding[0], embedding[1:]
                entity = ent_catalogue.idx_reverse[int(entity)]
                entities.append(entity)

    with open("/fsx/kassner/OSCAR/processed/cnn_bbc_news.jsonl") as f:
        i = 0
        for line in f:
            line = json.loads(line)
            gold_entities = line['entities_raw']
            sentences = sent_tokenize(line["text_raw"])
            sentence_offset = 0
            character_offset = len(sentences[sentence_offset])
            for entity in gold_entities:
                if entity["offset"] > character_offset:
                    sentence_offset += 1
                    character_offset += len(sentences[sentence_offset])
                    character_offset += 1
                if entity['entity_id'] not in ent_catalogue:
                    continue
                if entity['entity_id']!=entities[i]:
                    # print(line['data_example_id'], i, entity['entity_id'], entities[i])
                    continue
                i +=1
                if entity['entity_id'] in ent_catalogue_known:
                    continue
                if str(i) in raw_text:
                    raw_text[str(i)] = sentences[sentence_offset]
                    cluster_id.add(line['data_example_id'])
            if i>=len(entities):
                break
            

    with open("data/blink/known_entities_cluster.jsonl", 'w') as f:
        for cluster in cluster2idcs:
            ent_id = max(cluster2idcs[cluster]["entities"], key=cluster2idcs[cluster]["entities"].count)
            ent_name = ent_catalogue.idx_reverse[int(ent_id)]
            description = ''
            for i in cluster2idcs[cluster]['idcs']:
                description += raw_text[i]
                description += " "
            out_dict = {"title": ent_name, "entity": ent_name, "text": description}
            f.write(json.dumps(out_dict))
            f.write("\n")

    total_num_train = 50000
    total_num_test = 10000
    total_num_dev = 10000

    max_num_entities = 20000
    entities = set(counts.keys())
    
    num_train = 0
    num_test = 0
    num_dev = 0
    with open(ouput_base_path + "train.jsonl", "w") as f_train:
        with open(ouput_base_path + "test.jsonl", "w") as f_test:
            with open(ouput_base_path + "valid.jsonl", "w") as f_dev:
                with open(dataset_path) as f:
                    for line in f:
                        line = json.loads(line)
                        id = '_'.join(line['query_id'].split("_")[0:3])
                        if len(entities)>=max_num_entities:
                            if line['label_id'] not in entities:
                                continue
                        entities.add(line['label_id'])
                        if id not in cluster_id:
                            p = random.random()
                            if num_train<total_num_train and p<0.6:
                                f_train.write(json.dumps(line))
                                f_train.write('\n')
                                num_train +=1
                            elif num_test<total_num_test and p<0.8:
                                f_test.write(json.dumps(line))
                                f_test.write('\n')     
                                num_test +=1 
                            elif num_dev<total_num_dev and p>0.6:
                                f_dev.write(json.dumps(line))
                                f_dev.write('\n')     
                                num_dev +=1          


def generate_data_training_indexer_novel(dataset_path, ouput_base_path):

    ent_catalogue_idx_path = "/data/home/kassner/BELA/data/blink/en_bert_ent_idx.txt"
    novel_entity_idx_path = "/data/home/kassner/BELA/data/blink/novel_entities_filtered.jsonl"
    ent_catalogue_known = EntityCatalogue(ent_catalogue_idx_path, None)
    ent_catalogue = EntityCatalogue(ent_catalogue_idx_path, novel_entity_idx_path, True)

    cluster2idcs = {}                                                                                                                                                          
    raw_text = {}
    counts = {}
    #entities = set()
    cluster_id = set()
    with open('output_clustering/2022-01-24-084433_greedy_None_unknown_0.78_None_None.txt') as f:                                                                            
        for line in f:                                                                                                                                                         
            entity, cluster, idx = line.strip().split(", ")
            # entities.add(entity)
            
            ent_name = ent_catalogue.idx_reverse[int(entity)]
            if ent_name in counts:
                counts[ent_name] += 1
            else:
                counts[ent_name] = 1
            if cluster not in cluster2idcs:                                                                                                                                    
                cluster2idcs[cluster] = {}                                                                                                                                     
                cluster2idcs[cluster]["entities"] = [entity]                                                                                                                   
                cluster2idcs[cluster]["idcs"] = [idx]                                                                                                                          
            else:                                                                                                                                                              
                cluster2idcs[cluster]["entities"].append(entity)                                                                                                               
                cluster2idcs[cluster]["idcs"].append(idx) 
            raw_text[idx] = "" 
            
    entities = []
    input_path = '/fsx/kassner/hydra_outputs/main/2022-01-24-084433/0/*.t7'
    embeddings_path_list = glob.glob(input_path)
    for embedding_path in sorted(embeddings_path_list):
        embeddings_buffer = torch.load(embedding_path, map_location='cpu')
        for embedding_batch in embeddings_buffer:
            for embedding in embedding_batch:
                entity, embedding = embedding[0], embedding[1:]
                entity = ent_catalogue.idx_reverse[int(entity)]
                entities.append(entity)

    with open("/fsx/kassner/OSCAR/processed/cnn_bbc_news.jsonl") as f:
        i = 0
        for line in f:
            line = json.loads(line)
            gold_entities = line['entities_raw']
            sentences = sent_tokenize(line["text_raw"])
            sentence_offset = 0
            character_offset = len(sentences[sentence_offset])
            for entity in gold_entities:
                if entity["offset"] > character_offset:
                    sentence_offset += 1
                    character_offset += len(sentences[sentence_offset])
                    character_offset += 1
                if entity['entity_id'] not in ent_catalogue:
                    continue
                if entity['entity_id']!=entities[i]:
                    # print(line['data_example_id'], i, entity['entity_id'], entities[i])
                    continue
                i +=1
                if entity['entity_id'] in ent_catalogue_known:
                    continue
                if str(i) in raw_text:
                    
                    output = {"type": "unknown", "label_id": entity['entity_id'], "context_doc_id": line['data_example_id'], "label_title": entity['entity_id']}
                    mention = line['text_raw'][entity['offset']:entity['offset']+entity['length']]
                    output["mention"] = mention
                    context_left = line['text_raw'][:entity['offset']]
                    output["context_left"] = context_left
                    context_right = line['text_raw'][entity['offset'] + entity['length']:]
                    output["context_right"] = context_right
                    output["mention_id"] = line['data_example_id']
                    raw_text[str(i)] = output
                    cluster_id.add(line['data_example_id'])
            if i>=len(entities):
                break
            
        
        ent2desc = {}
        with open("data/blink/novel_entities_cluster.jsonl") as f:
            for line in f:
                line = json.loads(line)
                ent = line["entity"]
                description = line["text"]
                if ent not in ent2desc:
                    ent2desc[ent] = description
        
        with open(ouput_base_path + "train.jsonl", "w") as f_train:
            num_novel = 0
            for i in raw_text:
                out_dict = raw_text[i]
                ent = out_dict["label_title"]
                if ent in ent2desc:
                    out_dict["label"] = ent2desc[ent]
                    out_dict["context_doc_id"] = out_dict["context_doc_id"]
                    num_novel +=1
                    out_dict["mention_id"] = out_dict["context_doc_id"]
                    f_train.write(json.dumps(out_dict))
                    f_train.write("\n")

    total_num_train = num_novel
    total_num_test = 10000
    total_num_dev = 10000
    print(total_num_train)

    max_num_entities = 20000
    entities = set(counts.keys())

    ent2desc = {}
    with open("data/blink/entity.jsonl") as f:
        for line in f:
            line = json.loads(line)
            ent = line["entity"]
            description = line["text"]
            ent2desc[ent] = description
    
    num_train = 0
    num_test = 0
    num_dev = 0
    with open(ouput_base_path + "train.jsonl", "a") as f_train:
        with open(ouput_base_path + "test.jsonl", "w") as f_test:
            with open(ouput_base_path + "valid.jsonl", "w") as f_dev:
                with open(dataset_path) as f:
                    for line in f:
                        line = json.loads(line)
                        line['type'] = "known"
                        line['label'] = ent2desc[ent]
                        line['label_title'] = ent
                        id = '_'.join(line['query_id'].split("_")[0:3])
                        if line['label_id'] not in ent2desc:
                                continue
                        if len(entities)>=max_num_entities:
                            if line['label_id'] not in entities:
                                continue
                        entities.add(line['label_id'])
                        if id not in cluster_id:
                            p = random.random()
                            if num_train<total_num_train and p<0.6:
                                f_train.write(json.dumps(line))
                                f_train.write('\n')
                                num_train +=1
                            elif num_test<total_num_test and p<0.8:
                                f_test.write(json.dumps(line))
                                f_test.write('\n')     
                                num_test +=1 
                            elif num_dev<total_num_dev and p>0.6:
                                f_dev.write(json.dumps(line))
                                f_dev.write('\n')     
                                num_dev +=1               


def convert2blink(f_out, dataset_path):
    with open(dataset_path) as f:
        for line in f:
            line = json.loads(line)
            for j, ent in enumerate(line["entities_raw"]):
                output = {}
                line['id'] = line['data_example_id']
                output["query_id"] = line['data_example_id'] + "_" + str(j)
                mention = line['text_raw'][ent['offset']:ent['offset']+ent['length']]
                context_left = line['text_raw'][:ent['offset']]
                context_right = line['text_raw'][ent['offset'] + ent['length']:]
                output["context_left"] = context_left
                output["mention"] = mention
                output["context_right"] = context_right
                output["type"] = 'mention'

                #if ent['entity_id'] in title2wikidataID:
                #p = random.random()
                #if p<=0.2:
                output["label_id"] = ent['entity_id']
                f_out.write(json.dumps(output))
                f_out.write('\n')

'''def convert2arboEL(title2wikidataID, f_out, dataset_path):
    with open(dataset_path) as f:
        for line in f:
            line = json.loads(line)
            output = {}
            line['context_doc_id'] = line['data_example_id']
            for ent in line["entities_raw"]:
                mention = line['text_raw'][ent['offset']:ent['offset']+ent['length']]
                context_left = line['text_raw'][:ent['offset']]
                context_right = line['text_raw'][ent['offset'] + ent['length']:]
                output["context_left"] = context_left
                output["mention"] = mention
                output["context_right"] = context_right

                output["mention_id"] = ent['entity_id']
                output["type"] = 
                output["label"] = 
                output["label_title"] = ent['entity_id']
                if ent['entity_id'] in title2wikidataID:
                    p = random.random()
                    if p<=0.2:
                        output["label_id"] = int(title2wikidataID[ent['entity_id']][1:])
                        f_out.write(json.dumps(output))
                        f_out.write('\n')'''


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_wikidata",
        type=str,
    )
    parser.add_argument(
        "--base_path",
        type=str,
    )
    parser.add_argument(
        "--datasets",
        type=str,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/fsx/kassner/OSCAR/processed/cnn_bbc_news.jsonl"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default='/fsx/kassner/hydra_outputs/main/2022-01-24-084433/0/'
    )
    parser.add_argument(
        "--cluster_output_path",
        type=str,
    )
    parser.add_argument(
        "--base_path_entities",
        type=str,
        default="/data/home/kassner/BELA/data/blink/"
    )
    args, _ = parser.parse_known_args()


    datasets = args.datasets.split(',')
    output_path = args.output_path + "_".join(datasets) + ".jsonl"
    datasets = [args.base_path + "_".join(datasets) + ".jsonl"]
    print(output_path)
    if not os.path.exists(output_path):
        with open(output_path, "w") as f_out:
            for subset in datasets:
                convert2blink(f_out, subset)
    else:
        output_base_path = args.output_path + "OSCAR_indexing_with_novel/"
        generate_data_training_indexer_novel(output_path, output_base_path)

    #generate_data_indexer(args.base_path_entities, args.cluster_output_path, args.embeddings_dir, args.data_path)