import requests
from tqdm import tqdm
import time
import json


def run_query(query):
    url = 'https://query.wikidata.org/sparql'
    r = requests.get(url, params = {'format': 'json', 'query': query})
    i = 1
    while r.status_code != 200 and i <= 3:
        time.sleep(i)
        r = requests.get(url, params = {'format': 'json', 'query': query})
        i += 1
    
    if r.status_code != 200:
        return None

    return r.json()

def query_description_and_label(wikidata_id):
    query = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        PREFIX schema: <http://schema.org/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?label ?description
        WHERE {
        wd:""" + wikidata_id + """ schema:description ?description .
        wd:""" + wikidata_id + """ rdfs:label ?label
        FILTER ( lang(?description) = "en" )
        FILTER ( lang(?label) = "en" )
        }
        """
    data = run_query(query)
    if data is None:
        return None, None
    labels = [record["label"]["value"] for record in data["results"]["bindings"]]
    descriptions = [record["description"]["value"] for record in data["results"]["bindings"]]
    if len(labels) > 0 and len(descriptions) > 0:
        assert len(labels) == len(descriptions) == 1, f"Error for ID {wikidata_id}"
        return labels[0], descriptions[0]
    return None, None

def read_property_index(path):
    result = {}
    with open(path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            assert len(record) == 1
            pid = list(record.keys())[0]
            result[pid] = record[pid]
    return result


def main():
    property_index_path = '/fsx/matzeni/data/property_to_label.jsonl'
    property_index = read_property_index(property_index_path)
    result = {}
    print("Querying Wikidata...")
    for p in tqdm(property_index):
        result[p] = {}
        label, description = query_description_and_label(p)
        result[p]["label"] = property_index[p]
        result[p]["description"] = ""
        if label is not None and description is not None:
            result[p]["label"] = label
            result[p]["description"] = description
    
    with open('/fsx/matzeni/data/duck/properties.json', 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
