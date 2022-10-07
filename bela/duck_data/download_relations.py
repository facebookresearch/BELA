import pickle as pkl
from pathlib import Path
import requests
from xml.dom import minidom
from tqdm import tqdm
import time

def query_properties(wikidata_id):
    url = 'https://query.wikidata.org/sparql'
    query = '''
    SELECT DISTINCT ?wdLabel ?wd {
      VALUES (?ent) {
    ''' + \
    f"(wd:{wikidata_id})" + \
    '''
    }
    ?ent ?p ?statement .
    ?statement ?ps ?ps_ .

    ?wd wikibase:claim ?p.
    ?wd wikibase:statementProperty ?ps.

    OPTIONAL {
    ?statement ?pq ?pq_ .
    ?wdpq wikibase:qualifier ?pq .
    }

      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
    } ORDER BY ?wd ?statement ?ps_
    '''
    r = requests.get(url, params = {'format': 'json', 'query': query})
    i = 1
    while r.status_code != 200 and i <= 3:
        time.sleep(i)
        r = requests.get(url, params = {'format': 'json', 'query': query})
        i += 1
    
    if r.status_code != 200:
        tqdm.write(f"[WARNING] Could not run query for id: {wikidata_id}")
        return [], []

    data = r.json()
    property_labels = [record["wdLabel"]["value"] for record in data["results"]["bindings"]]
    property_ids = [record["wd"]["value"].split("/")[-1] for record in data["results"]["bindings"]]
    assert len(property_labels) == len(property_ids)
    return property_ids, property_labels

def read_entities():
    ned_dataset_paths = list(Path("/fsx/matzeni/WNED/wned-datasets/").rglob("*.xml"))
    entities = []
    for dataset_path in ned_dataset_paths:
        dom = minidom.parse(str(dataset_path))
        current_entities = [
            element.childNodes[0].nodeValue for element in dom.getElementsByTagName("wikiName")
            if len(element.childNodes) > 0
        ]
        current_entities = [e.strip() for e in current_entities if e is not None and e.strip() != ""]
        entities += current_entities
    return entities


def print_entity_stats(entities, wikipedia_to_wikidata):
    count = 0
    for e in entities:
        if e is not None and e in wikipedia_to_wikidata:
            count += 1
    coverage = count / len(entities)
    print(f"Coverage: {coverage:.4f}")
    print(f"Number of entities: {len(entities)}")


def build_entity_to_properties_mapping(entities, wikipedia_to_wikidata):
    entity_to_properties = {}
    property_ids_to_labels = {}

    for e in tqdm(entities):
        time.sleep(1)
        wikidata_ids = wikipedia_to_wikidata.get(e, None)
        if wikidata_ids is not None:
            for wikidata_id in wikidata_ids:
                property_ids, property_labels = query_properties(wikidata_id)
                entity_to_properties[wikidata_id] = property_ids
                for i, pid in enumerate(property_ids):
                    property_ids_to_labels[pid] = property_labels[i]
    return entity_to_properties, property_ids_to_labels

def main():
    wikipedia_to_wikidata_path = Path("/fsx/matzeni/data/en_title2wikidataID.pkl")
    print("Reading Wikipedia to Wikidata mapping...")
    with open(wikipedia_to_wikidata_path, "rb") as f:
        wikipedia_to_wikidata = pkl.load(f) 

    entities = read_entities()
    print()
    print_entity_stats(entities, wikipedia_to_wikidata)
    entity_to_properties, property_ids_to_labels = build_entity_to_properties_mapping(entities, wikipedia_to_wikidata)
    
    print("Saving...")
    with open("/fsx/matzeni/data/entity_to_properties.pkl", "wb") as f:
        pkl.dump(entity_to_properties, f)

    with open("/fsx/matzeni/data/property_ids_to_labels.pkl", "wb") as f:
        pkl.dump(property_ids_to_labels, f)

if __name__ == "__main__":
    main()
