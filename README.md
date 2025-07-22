# Frances
A closed-loop metabolic pathway optimizer that optimizes a microbial production loop for a specific target molecule.

Databases: MetaCyc, KEGG, BioCyc
Software: CobraPy
Literature mining for gene edit --> yield pairs

`optimize(target_molecule)` --> strain_design + pathway edits + predicted yield

Example:
```python
optimize(target_molecule="limonene", host_organism="E.coli") → { "knockout": [geneX], "overexpress": [geneY], "yield": 12.3 g/L }
```

## Steps
1. Ingests a target molecule + host organism
2. Runs GNN + Attention on the metabolic graph
3. Generates interventions (knockouts/overexpressions)
4. Simulates flux + yield (CobraPy or BiGG proxy)
5. Returns actionable strain design
6. Stores design + result for future reuse or retraining

All this so they can cut biomanufacturing time