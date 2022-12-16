# Duck

This repository contains the source code for training Duck and replicating experimental results.


## Creating a conda environment

First, create and activate the conda environment as follows:
```
conda env create -f duck_env.yml
conda activate duck
```

## Downloading datasets

All the datasets are already available on the cluster at ```/fsx/matzeni/data/GENRE```. If you are running experiments elsewhere or need to downlaod the datasets again, you can run:

```
## Training and validation dataset
wget http://dl.fbaipublicfiles.com/KILT/blink-train-kilt.jsonl
wget http://dl.fbaipublicfiles.com/KILT/blink-dev-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/aida-train-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/aida-dev-kilt.jsonl

## Test datasets
wget http://dl.fbaipublicfiles.com/GENRE/ace2004-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/aquaint-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/aida-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/msnbc-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/clueweb-test-kilt.jsonl
wget http://dl.fbaipublicfiles.com/GENRE/wiki-test-kilt.jsonl
```

You will also need to download the following file, which contains Wikipedia entities and their descriptions.

```
wget http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
```
The file above is also available at ```/fsx/matzeni/data/duck/kilt_knowledgesource.jsonl```

Finally, we need a mapping from Wikipedia titles to Wikidata IDs. This can be downloaded from the following link:
```
wget https://dl.fbaipublicfiles.com/GENRE/lang_title2wikidataID-normalized_with_redirect.pkl
```
A preprocessed version of this file (with only the English language) is availble on the cluster at:
```/fsx/matzeni/data/duck/wikipedia_to_wikidata_en.pkl```.

## Preprocessing entities and relations
Before training the model, we need to preprocess the data.
Run the following command to tokenize entity descriptions with BERT.
```
python duck/preprocessing/build_catalogue.py \ 
    kb_path=/fsx/matzeni/data/duck/wikipedia_to_wikidata_en.pkl  \
    label_key=wikipedia_title \ 
    text_key=text \ 
    input_path=/fsx/matzeni/data/duck/kilt_knowledgesource.jsonl \
    output_tok_ids_path=/fsx/matzeni/data/duck/bert_ent_256_tok_ids.h5 \
    output_idx_path=/fsx/matzeni/data/duck/ent_idx.txt \
```
This will create two files:
* ```/fsx/matzeni/data/duck/ent_idx.txt``` is a list of the Wikidata IDs of the entities in ```kilt_knowledgesource.jsonl```
* ```/fsx/matzeni/data/duck/bert_ent_256_tok_ids.h5``` contains a matrix of the IDs of the first 256 tokens of the entities in ```ent_idx.txt```.


For duck we need both entities and relations. The file ```/fsx/matzeni/data/duck/properties.json``` contains all properties in Wikidata with their english label and description, which can be preprocessed as follows:
```
python duck/preprocessing/build_catalogue.py \ 
    label_key=label \ 
    text_key=description \ 
    input_path=/fsx/matzeni/data/duck/kilt_knowledgesource.jsonl \
    output_tok_ids_path=/fsx/matzeni/data/duck/bert_rel_256_tok_ids.h5 \
    output_repr_path=/fsx/matzeni/data/duck/bert_rel_256_repr.h5
    output_idx_path=/fsx/matzeni/data/duck/rel_idx.txt \
```

This creates the following files:
* ```/fsx/matzeni/data/duck/rel_idx.txt``` contains a list of the ids of each relation in Wikidata
* ```/fsx/matzeni/data/duck/bert_rel_256_tok_ids.h5``` is a matrix containing token IDs for relations, where each releation is represented as a text of the form ```"<label> [SEP] <description>"```
*  ```/fsx/matzeni/data/duck/bert_rel_256_repr.h5``` contains the representation of the ```[CLS]``` token for each relation.

## Additional requirements

Two additional files are required to train Duck:
* ```/fsx/matzeni/data/duck/ent_to_rel.json``` is a dictionary mapping entities to their properties in Wikidata.
* ```/fsx/matzeni/data/duck/stop_rels.jsonl``` contains relations that are supposed to be filtered out, as they are very frequent and do not provide any type information. The list was curated manually and mostly contains references to other knowledge bases.

## Training Duck with box embeddings

The configuration for training Duck is ```duck/conf/duck.yaml```. The paths to the files above should be appropriately set in the config file before training the model.  See ```duck/conf/duck.yaml``` for an example.
In order to train Duck with in-batch negatives and box embeddings generated based on the natural-language descriptions of relations, you can run:
```
python duck/main.py trainer=1host data.batch_size=32 run_name=duck_box
```
This trains the model by optimizing jointly an entity-disambiguation loss and a "duck" loss that pushes entities inside all boxes corresponding to relations that the entity has and outside randomly sampled boxes corresponding to relations that the entity does not have. The command above is supposed to run on 1 node with 8 GPUs and saves checkpoints to the directory denoted by the entry ```ckpt_dir``` in ```duck.yaml``` (defaults to ```/checkpoints/matzeni/duck/checkpoints```).


## Training Duck with a joint entity-relation encoder

You can train Duck with a joint entity-relation encoder that does not use box embeddings. This model creates a joint encoding of an entity and its relations by encoding the relations of an entity with a transformer encoder and then applying an attention module that updates entity representations by attending to relation encodings. This model gives the best results without hard negatives.
It can be trained by setting the ```ablations.joint_ent_rel_encoding``` flag: 

```
python duck/main.py trainer=1host ablations.joint_ent_rel_encoding=True data.batch_size=32 run_name=duck_joint
```

## Training Duck with pre-trained box embeddings

Duck can be trained using pre-trained box embeddings. You will need to train box embeddings for this (see below) or just use a pretrained model from```/checkpoints/matzeni/duck/checkpoints/r2b_wd```. For instance, you can run:

```
python duck/main.py trainer=1host data.batch_size=32 run_name=duck_box_pretrain rel_to_box_mode=/checkpoints/matzeni/duck/checkpoints/r2b_wd/r2b_wd_origin_epoch=373_kldiv_train=0.0010_last.ckpt
```

This will use pre-trained box embeddings of Wikidata relataions, generated by modeling conditional probabilities of relation estimated from co-occurrences.

## Comparison with BLINK

The code can be used to run BLINK by setting the ```ablations.blink``` flag as follows.

```
python duck/main.py run_name=blink trainer=1host ablations.blink=True data.batch_size=32
```
The command above will run a model that uses only the entity and mention encoders, relying on the dot product to score mentions against entities. The model is trained to minimize a cross-entropy objective.

## Training with hard negatives

In order to use hard negatives, you first need to train a model with in-batch negatives. This will save a model checkpoint in the ```ckpt_dir``` of your config file (defaults to ```/checkpoints/matzeni/duck/checkpoints```).

Then, you can use this checkpoint to mine negatives, by running the following command:

```
python duck/preprocessing/mine_negatives.py resume_path=</path/to/model/checkpoint> output_path=<path/to/output.json>
```
This outputs a json file with negatives for each entity, which can be used to train the model with hard negatives:

```
python duck/main.py resume_path=</path/to/model/checkpoint/with/batch/negatives> data.neighbors_path=</path/to/negatives.json> data.batch_size=16
```

## Training box embeddings only 

You can use the code in this repository to train box embeddings of relations in Wikidata in such a way that the overlap of two boxes reflects the conditional probability of relations. To this end, you first need to estimate ground-truth conditional probabilities from relation co-occurrences.

```
python compute_rel_probs.py
```

The command above reads the ```/fsx/matzeni/data/duck/ent_to_rel.json``` file and outputs a json file containing conditional (and marginal) probabilities of Wikidata relations.
A precomputed version is available at: ```/fsx/matzeni/data/duck/rel_probs.json```.

You can then train box embeddings by running:

```
python duck/main.py --config-name rel_to_box
```

By default, the command above uses the a Sigmoid parametrization of Gumbel boxes, with normal initialization and no regularization (the left corner is constraied to be negative and larger than -1, the right corner is constrained to be positive and smaller than 1).

## Building an index of entities according to a notion of similarity based on duck typing

This repository also contains an implementation of a binary index that can be used to retrieve entities that share the most relations with a target entity. The index is implemented in ```duck/preprocessing/duck_index.py```. In order to build the index and generate a dictionary of neighbors, run:

```
python duck/preprocessing/duck_neighbors.py 
```
The config file for this script is available at ```duck/conf/preprocessing/duck_neighbors.yaml```. The script reads the following files:
* the entity index: ```/fsx/matzeni/data/duck/ent_idx.txt```
* the relation index: ```/fsx/matzeni/data/duck/rel_idx.txt```
* the uniformative relations: ```/fsx/matzeni/data/duck/stop_relations.jsonl```
* the mapping from entities to relations: ```/fsx/matzeni/data/duck/ent_to_rel.json```

and generates as outputs a faiss index (if it does not exist yet) and a dictionary of neighbors, at the paths specified in the config file by the entries:
* index_path: ```/fsx/matzeni/data/duck/duck_index_flat```
* neighbors_path: ```/fsx/matzeni/data/duck/duck_neighbors.json```

The script uses a flat index if a gpu is available (as specified in the config file by the ```gpu``` entry) or a HNSW index on CPU. Both the index and the precomputed neighbors are available on the cluster.

## Structure of the code

All the code is contained in the ```duck``` directory and structured as follows.

* The  ```box_tensors``` folder contains the implementation of hard and gumbel boxes, including gumbel/hard intersection, bessel-approx/soft/hard volume and a ```BoxTensor``` class that acts as the main interface for using box tensors.
* The ```datamodule``` folder contains the interface for accessing, transforming and loading the datasets. The two main datamodules are ```duck_datamodule.py``` that handles the entity-disambiguation datasets and ```rel_to_box_datamodule.py``` that loads conditional probabilities of relations to train box embeddings only.
* The ```modules``` folder contains several neural modules used in the project.
* The ```preprocessing``` folder contains preprocessing scripts, including the tokenization of entities and relations, computing conditional probabilities of relations, a script for downloading relations of an entity from the Wikidata endpoint, building the entity index and mining negatives.
* The ```task``` directory contains the implementation of Duck and the Rel2Box model for learning box embeddings of relations.

## Note
The code logs metrics to Weights&Biases. You can disable this by setting ```wandb``` to ```false``` in ```duck/conf/duck.yaml```