import argparse
import tqdm
import pickle
import json

def write_out_for_labeling(paragraph, data_example_id, f_out):
    paragraph = paragraph.strip()
    output = {'text': paragraph, 'id': data_example_id}
    f_out.write(json.dumps(output))
    f_out.write("\n")


def process_wiki_based_data(base_dataset, lang):

    with open(base_dataset + "/" + lang + "/" + lang + "wiki0.pkl", "rb") as f:
        data = pickle.load(f)
    f_out_l = open(base_dataset + "/" + lang + "_internal.jsonl", "w")
    data_example_id = 0
    for d in tqdm.tqdm(data):
        if len(data[d]['anchors']) > 0:
            paragraph_id = data[d]['anchors'][0]['paragraph_id']
            entities = []
            for anchor in data[d]['anchors']:
                if anchor['wikidata_src'] == 'wikipedia':
                    paragraph_id_current = anchor['paragraph_id']
                    if paragraph_id_current == paragraph_id:
                        entities.append(anchor)
                    else:
                        if paragraph_id > 1 and len(entities) > 0:
                            keep = True
                            paragraph = data[d]['paragraphs'][paragraph_id]
                            if len(data[d]['paragraphs'][paragraph_id])>10:
                                for ent in entities:
                                    start_id = ent['start']
                                    end_id = ent['end']
                                    if start_id >=len(paragraph)-1 or end_id > len(paragraph)-1 or start_id==end_id:
                                        keep = False
                                if keep:
                                    write_out_for_labeling(data[d]['paragraphs'][paragraph_id], data_example_id, f_out_l)
                                    # write_out(entities, data[d]['paragraphs'][paragraph_id], data_example_id, f_out)
                                    data_example_id += 1
                        paragraph_id = anchor['paragraph_id']
                        entities = [anchor]
    f_out.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        help="Base folder with Wikipedia data.",
    )
    parser.add_argument(
        "--training_type",
        type=str,
    )
    parser.add_argument(
        "--base_dataset",
        type=str,
    )
    parser.add_argument(
        "--lang",
        type=str,
    )

    args, _ = parser.parse_known_args()
    #if args.data_type == "wiki":
    process_wiki_based_data(args.base_dataset, args.lang)
    #filter2id_set(args.base_dataset, args.lang)