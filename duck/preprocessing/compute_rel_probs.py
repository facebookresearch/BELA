from duck.common.utils import load_json
from tqdm import tqdm
import json

ENT_TO_REL_PATH = "/fsx/matzeni/data/duck/ent_to_rel.json"
OUTPUT_PATH = "/fsx/matzeni/data/duck/rel_probs.json"


def main():
    ent_to_rel = load_json(ENT_TO_REL_PATH)
    all_rels = set(r for rels in ent_to_rel.values() for r in rels)
    rel_to_ent = {r: set() for r in all_rels}
    for e, rels in tqdm(ent_to_rel.items()):
        for r in rels:
            rel_to_ent[r].add(e)

    all_rels = list(all_rels)
    rel_probs = {}
    for r1 in tqdm(all_rels):
        for r2 in all_rels:
            ents1 = rel_to_ent[r1]
            ents2 = rel_to_ent[r2]
            p_r1 = len(ents1) / len(ent_to_rel)
            p_r1_given_r2 = len(ents1 & ents2) / len(ents2)
            rel_probs[f"{r1}|{r2}"] = p_r1_given_r2
            rel_probs[r1] = p_r1
    
    with open("/fsx/matzeni/data/duck/rel_probs.json", "w") as f:
        json.dump(rel_probs, f)



if __name__ == "__main__":
    main()
