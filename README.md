# Frances
A closed-loop metabolic pathway optimizer that optimizes a microbial production loop for a specific target molecule.

## Root function 
`optimize(target_molecule)` --> strain_design + pathway edits + predicted yield

Example:
```python
optimize(target_molecule="limonene", host_organism="E.coli") → { "knockout": [geneX], "overexpress": [geneY], "yield": 12.3 g/L }
```

## Tools
Data sources: MetaCyc, KEGG, BioCyc, FBA outputs  
Graph construction: NetworkX  
GNN + attention: PyTorch Geometric (GATConv)  
RL policy (strain edits): PPO, SBX
Flux simulation: CobraPy (proxy to FBA)

