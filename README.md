# Frances
A closed-loop metabolic pathway optimizer that optimizes a microbial production loop for a specific target molecule.

Databases: MetaCyc, KEGG, BioCyc
Software: CobraPy
Literature mining for gene edit --> yield pairs

`optimize(target_molecule)` --> strain_design + pathway edits + predicted yield

Example:
```python
optimize(target_molecule="limonene", host="E.coli") → { "knockout": [geneX], "overexpress": [geneY], "yield": 12.3 g/L }
```