### Week 1

- Researched Graph libraries & decided to use Pytorch Geometric
- Created first feature encoders
- Constructed base graph for project

### Week 2 - Plans

- Get first full pipeline working
  - Research more about GNNs and implement first models
  - Try some (even smaller) LMs 
  - Implement training loop

- Notes W2:
  - Insert graph embedding at placeholder token position within the LLMs input embeddings... which is a step in the LLM
  - TallRec doesnt use the train/val/test split and there are multiple splits, so maybe easier to just use all and split there
  - Read this to understand train/test splitting of edges: https://github.com/pyg-team/pytorch_geometric/discussions/6923
  - OKAY!! So what we UPDATE is the EMBEDDING of our movie/user token basically, that is then used during forward pass as normal
  - 
- Questions W2:
  - I needed to add edges in both directions for PyG but my dataset only asks for one direction: problem?
  - Check w/ Kenza the embedding stuff



How does num layers of GNN / architecture influence results?
How does having ratings/structure of graph influence results?

Optional: Visualization of graph