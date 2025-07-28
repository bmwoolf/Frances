import networkx as nx

G = nx.DiGraph()


with open("iML1515.json", "r") as f:
    data = json.load(f)

G = nx.DiGraph()

for rxn in model["reactions"]:
    rxn_id = rxn["id"]
    G.add_node(rxn_id, type="reaction", genes=rxn.get("gene_reaction_rule", ""))

    for met_id, coeff in rxn["metabolites"].items():
        G.add_node(met_id, type="metabolite")

        if coeff < 0:
            G.add_edge(met_id, rxn_id)
        else:
            G.add_edge(rxn_id, met_id)

# add limonene as a metabolite
G.add_node("gpp_c", type="metabolite")
G.add_node("limonene", type="metabolite")

# add heterologous reaction node
G.add_node("LS", type="reaction", genes="limonene_synthase")

# connect it automatically
G.add_edge("gpp_c", "LS")       # gpp -> LS
G.add_edge("LS", "limonene_c")  # LS -> limonene
