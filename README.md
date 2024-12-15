# Motion Planning with Locality-Sensitive Hashing (LSH)

This repository implements a motion planning system that leverages Locality-Sensitive Hashing (LSH) to store and retrieve environment embeddings for efficient pathfinding in robot navigation. The code includes modules for generating paths, training a model to extract embeddings, and storing/retrieving data with LSH.

## How to Run

### Step 1: Generate Paths
Run the `generate_paths.py` script to create the paths required for the environment:

```bash
python generate_paths.py
```

##Step 2: Train the model
```bash
python model.py
```
##Step 3: Create the SimHash data structure and query similar environments
```bash
python lsh.py
```
