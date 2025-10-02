# app.py
# Graph Analyzer — Streamlit Edition
# Run: streamlit run app.py

import io
import itertools
import random
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

# ------------------------------- Utilities ------------------------------- #

def _safe_int(x):
    try:
        return int(x)
    except:
        return x

def parse_edge_list(text: str, directed=False, multigraph=False):
    G = (nx.MultiDiGraph() if directed else nx.MultiGraph()) if multigraph else (nx.DiGraph() if directed else nx.Graph())
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper() == "END":
            continue
        parts = line.split()
        if len(parts) >= 2:
            u, v = _safe_int(parts[0]), _safe_int(parts[1])
            if len(parts) >= 3:
                try:
                    w = float(parts[2])
                    G.add_edge(u, v, weight=w)
                except:
                    G.add_edge(u, v)
            else:
                G.add_edge(u, v)
    return G

def parse_adjacency_list(text: str, directed=False, multigraph=False):
    G = (nx.MultiDiGraph() if directed else nx.MultiGraph()) if multigraph else (nx.DiGraph() if directed else nx.Graph())
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper() == "END":
            continue
        if ":" not in line:
            continue
        vtx, neigh = line.split(":", 1)
        v = _safe_int(vtx.strip())
        neighbors = [_safe_int(x) for x in neigh.strip().split()]
        for u in neighbors:
            G.add_edge(v, u)
    return G

def parse_adjacency_matrix(text: str, directed=False, multigraph=False):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper() == "END":
            continue
        rows.append([float(x) for x in line.split()])
    M = np.array(rows)
    n = M.shape[0]
    G = (nx.MultiDiGraph() if directed else nx.MultiGraph()) if multigraph else (nx.DiGraph() if directed else nx.Graph())
    G.add_nodes_from(range(n))
    if directed:
        for i in range(n):
            for j in range(n):
                if i != j and M[i, j] != 0:
                    G.add_edge(i, j)
    else:
        for i in range(n):
            for j in range(i + 1, n):
                if M[i, j] != 0 or M[j, i] != 0:
                    G.add_edge(i, j)
    return G

def visualize_graph(G: nx.Graph, layout_seed: int = 42, highlight_edges=None, highlight_nodes=None, title: Optional[str]=None):
    if G is None or G.number_of_nodes() == 0:
        st.info("No graph to visualize yet.")
        return
    # choose layout
    try:
        if not G.is_directed() and nx.is_bipartite(G):
            X, Y = nx.bipartite.sets(G)
            pos = {}
            pos.update({x: (0, i) for i, x in enumerate(sorted(X, key=str))})
            pos.update({y: (1, i) for i, y in enumerate(sorted(Y, key=str))})
        else:
            pos = nx.spring_layout(G, seed=layout_seed)
    except Exception:
        pos = nx.spring_layout(G, seed=layout_seed)

    plt.figure(figsize=(6.5, 5.5))
    # base edges
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        # draw parallel edges with slight transparency
        for u, v, k in G.edges(keys=True):
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=0.35, edge_color="#666666",
                                   connectionstyle="arc3,rad=0.08")
    else:
        nx.draw_networkx_edges(G, pos, alpha=0.35, edge_color="#666666")

    # base nodes
    nx.draw_networkx_nodes(G, pos, node_color="#9ecae1", node_size=550)
    nx.draw_networkx_labels(G, pos, font_weight="bold")

    # highlights
    if highlight_edges:
        nx.draw_networkx_edges(G, pos, edgelist=highlight_edges, width=3, edge_color="#d62728")
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(highlight_nodes), node_color="#ff7f0e", node_size=650)

    # title
    subtype = "Multi" if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) else ""
    prefix = "Directed " if G.is_directed() else ""
    ttl = title or f"{prefix}{subtype}Graph — |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}"
    plt.title(ttl)
    # Draw edge weights as labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green')
    plt.axis("off")
    st.pyplot(plt, clear_figure=True)


def degree_and_basic_stats(G: nx.Graph):
    st.subheader("Basic Counts")
    st.write(f"|V| = {G.number_of_nodes()}  |E| = {G.number_of_edges()}")
    st.subheader("Degree Function deg(v)")
    deg = dict(G.degree())
    st.write({str(k): v for k, v in deg.items()})
    st.subheader("Adjacency N(v)")
    st.write({str(v): list(G.neighbors(v)) for v in G.nodes()})

    st.subheader("Density")
    try:
        st.write(float(nx.density(G)))
    except Exception:
        st.write("undefined")

    if G.is_directed():
        st.write(f"Weakly CCs: {nx.number_weakly_connected_components(G)}")
        st.write(f"Strongly CCs: {nx.number_strongly_connected_components(G)}")
    else:
        st.write(f"Connected components: {nx.number_connected_components(G)}")
        try:
            if nx.is_connected(G) and G.number_of_nodes() <= 200:
                st.write(f"Diameter: {nx.diameter(G)}")
                st.write(f"Radius: {nx.radius(G)}")
                st.write(f"Center: {nx.center(G)}")
        except Exception:
            pass

    # Plots
    vals = list(deg.values())
    if len(vals) > 0:
        fig1 = plt.figure(figsize=(5.5, 3.5))
        plt.hist(vals, bins=max(1, len(set(vals))))
        plt.xlabel("Degree"); plt.ylabel("Frequency"); plt.title("Degree Distribution")
        st.pyplot(fig1, clear_figure=True)

        fig2 = plt.figure(figsize=(5.5, 3.5))
        nodes_sorted = list(sorted(deg, key=str))
        heights = [deg[v] for v in nodes_sorted]
        plt.bar(range(len(nodes_sorted)), heights)
        plt.xticks(range(len(nodes_sorted)), [str(v) for v in nodes_sorted], rotation=45, ha="right")
        plt.ylabel("Degree"); plt.title("Vertex Degrees")
        plt.tight_layout()
        st.pyplot(fig2, clear_figure=True)

# ------------------------------- App State ------------------------------- #

if "G" not in st.session_state:
    st.session_state.G = None
if "layout_seed" not in st.session_state:
    st.session_state.layout_seed = 42

# ------------------------------- Sidebar UI ------------------------------- #

st.sidebar.title("Graph Analyzer")
menu = st.sidebar.radio(
    "Menu",
    [
        "1. Create / Load Graph",
        "2. Check Walk / Path / Cycle",
        "3. Generate Subgraph",
        "4. Identify Special Graph",
        "5. Check Graph Isomorphism",
        "6. Perform Graph Operations",
        "7. Compute Graph-Valued Functions",
        "8. Export / Reset",
    ],
    index=0,
)

st.sidebar.caption("Tip: You can switch tabs anytime — the current graph is kept in memory.")
st.sidebar.number_input("Layout seed", min_value=0, max_value=10_000, value=st.session_state.layout_seed, key="layout_seed")

# ------------------------------- Main Panels ------------------------------- #

st.title("Graph Analyzer — Streamlit")
st.write("Interactive graph creation, visualization, and analysis.")

# Panel 1: Create / Load
if menu.startswith("1"):
    st.header("Create / Load Graph")
    col1, col2, col3 = st.columns(3)
    directed = col1.checkbox("Directed", value=False)
    multigraph = col2.checkbox("Multigraph", value=False)
    show_after = col3.checkbox("Auto-visualize", value=True)

    mode = st.radio(
        "Input Method",
        ["Type edges", "Paste adjacency list", "Paste adjacency matrix", "Random graph", "Upload .txt"],
        horizontal=True,
    )

    if mode == "Type edges":
        st.markdown("Enter one edge per line as `u v` or `u v weight`. Click **Create** when done.")
        t = st.text_area("Edges", "0 1\n1 2\n2 0\n1 3", height=150)
        if st.button("Create"):
            G_new = parse_edge_list(t, directed=directed, multigraph=multigraph)
            st.session_state.G = G_new
            st.session_state.G_original = G_new.copy()
            st.success("Graph created.")
            if show_after:
                visualize_graph(st.session_state.G, layout_seed=st.session_state.layout_seed)

    elif mode == "Paste adjacency list":
        st.markdown("Format per line: `vertex: neighbor1 neighbor2 ...`")
        t = st.text_area("Adjacency List", "0: 1 2\n1: 0 3\n2: 0 3\n3: 1 2", height=150)
        if st.button("Create"):
            G_new = parse_adjacency_list(t, directed=directed, multigraph=multigraph)
            st.session_state.G = G_new
            st.session_state.G_original = G_new.copy()
            st.success("Graph created.")
            if show_after:
                visualize_graph(st.session_state.G, layout_seed=st.session_state.layout_seed)

    elif mode == "Paste adjacency matrix":
        st.markdown("Paste rows of 0/1 (or weights). Nonzero means edge. Use square matrix.")
        t = st.text_area("Adjacency Matrix", "0 1 1 0\n1 0 0 1\n1 0 0 1\n0 1 1 0", height=150)
        if st.button("Create"):
            G_new = parse_adjacency_matrix(t, directed=directed, multigraph=multigraph)
            st.session_state.G = G_new
            st.session_state.G_original = G_new.copy()
            st.success("Graph created.")
            if show_after:
                visualize_graph(st.session_state.G, layout_seed=st.session_state.layout_seed)

    elif mode == "Random graph":
        n = st.number_input("Vertices", min_value=1, max_value=200, value=6)
        m = st.number_input("Edges", min_value=0, max_value=10_000, value=8)
        if st.button("Generate"):
            G = (nx.MultiDiGraph() if directed else nx.MultiGraph()) if multigraph else (nx.DiGraph() if directed else nx.Graph())
            G.add_nodes_from(range(int(n)))
            max_edges = (n * (n - 1)) if directed else (n * (n - 1)) // 2
            if not multigraph and m > max_edges:
                st.info(f"Edges capped to {int(max_edges)} for simple graphs.")
                m = int(max_edges)
            added = 0
            while added < m:
                u = random.randrange(int(n))
                v = random.randrange(int(n))
                if u == v:
                    continue
                if multigraph or (not G.has_edge(u, v)):
                    G.add_edge(u, v)
                    added += 1
            st.session_state.G = G
            st.success("Random graph generated.")
            if show_after:
                visualize_graph(st.session_state.G, layout_seed=st.session_state.layout_seed)

    elif mode == "Upload .txt":
        st.markdown("Upload edge list (`u v [w]`) or adjacency list (`v: n1 n2 ...`).")
        upl = st.file_uploader("Text file", type=["txt"])
        looks_like = st.selectbox("Interpret as", ["Auto-detect", "Edge list", "Adjacency list"])
        if upl and st.button("Create"):
            text = upl.read().decode("utf-8", errors="ignore")
            if looks_like == "Adjacency list" or (looks_like == "Auto-detect" and ":" in text):
                G = parse_adjacency_list(text, directed=directed, multigraph=multigraph)
            else:
                G = parse_edge_list(text, directed=directed, multigraph=multigraph)
            st.session_state.G = G
            st.success("Graph created from file.")
            if show_after:
                visualize_graph(st.session_state.G, layout_seed=st.session_state.layout_seed)

    if st.session_state.G is not None and not show_after:
        visualize_graph(st.session_state.G, layout_seed=st.session_state.layout_seed)

# Panel 2: Walk / Path / Cycle
elif menu.startswith("2"):
    st.header("Check Walk / Path / Cycle")
    import copy
    G = copy.deepcopy(st.session_state.get("G_original", None))
    if G is None:
        st.warning("Create a graph first.")
    else:
        st.markdown("Enter a vertex sequence separated by spaces (e.g., `0 1 2 0`).")
        seq_raw = st.text_input("Sequence", "")
        if st.button("Analyze sequence"):
            seq = [ _safe_int(x) for x in seq_raw.split() if x.strip() != "" ]
            if len(seq) < 2:
                st.error("Need at least 2 vertices.")
            else:
                is_walk = all(G.has_edge(seq[i], seq[i+1]) or G.has_edge(seq[i+1], seq[i]) for i in range(len(seq)-1))
                is_path = is_walk and (len(set(seq)) == len(seq))
                is_cycle = False
                if len(seq) >= 3 and seq[0] == seq[-1]:
                    core = seq[:-1]
                    is_cycle = all(G.has_edge(core[i], core[i+1]) for i in range(len(core)-1)) and len(set(core)) == len(core)
                st.write(f"Sequence: {' → '.join(map(str, seq))}")
                st.write(f"Is Walk:  {bool(is_walk)}")
                st.write(f"Is Path:  {bool(is_path)}")
                st.write(f"Is Cycle: {bool(is_cycle)}")
                epath = [(seq[i], seq[i+1]) for i in range(len(seq)-1)]
                visualize_graph(G, layout_seed=st.session_state.layout_seed, highlight_edges=epath, highlight_nodes=set(seq))
        else:
            visualize_graph(G, layout_seed=st.session_state.layout_seed)

# Panel 3: Subgraph
elif menu.startswith("3"):
    st.header("Generate Subgraph")
    import copy
    G = copy.deepcopy(st.session_state.get("G_original", None))
    if G is None:
        st.warning("Create a graph first.")
    else:
        nodes = list(G.nodes())
        chosen = st.multiselect("Select vertices for subgraph", nodes)
        if chosen:
            H = G.subgraph(chosen).copy()
            st.success(f"Subgraph created: |V|={H.number_of_nodes()}, |E|={H.number_of_edges()}")
            visualize_graph(G, layout_seed=st.session_state.layout_seed, highlight_nodes=set(chosen), title="Original (highlighted)")
            st.subheader("Subgraph View")
            visualize_graph(H, layout_seed=st.session_state.layout_seed, title="Subgraph")

# Panel 4: Identify Special Graph
elif menu.startswith("4"):
    st.header("Identify Special Graph")
    import copy
    G = copy.deepcopy(st.session_state.get("G_original", None))
    if G is None:
        st.warning("Create a graph first.")
    else:
        n, m = G.number_of_nodes(), G.number_of_edges()
        st.write(f"|V|={n} |E|={m}")
        if not G.is_directed():
            is_complete = (m == n * (n - 1) // 2)
        else:
            is_complete = (m == n * (n - 1))
        st.write(f"Complete: {bool(is_complete)}")
        try:
            is_bip = nx.is_bipartite(G)
        except Exception:
            is_bip = False
        st.write(f"Bipartite: {bool(is_bip)}")
        if G.is_directed():
            st.write(f"DAG: {nx.is_directed_acyclic_graph(G)}")
            st.write(f"Weakly connected: {nx.is_weakly_connected(G)}")
        else:
            st.write(f"Tree: {nx.is_tree(G)}")
            st.write(f"Connected: {nx.is_connected(G)}")
        degs = [d for _, d in G.degree()]
        st.write(f"Regular: {len(set(degs)) == 1}")
        visualize_graph(G, layout_seed=st.session_state.layout_seed, title="Special Graph Check")

# Panel 5: Isomorphism
elif menu.startswith("5"):
    st.header("Graph Isomorphism (compare with a 2nd graph)")
    import copy
    G1 = copy.deepcopy(st.session_state.get("G_original", None))
    if G1 is None:
        st.warning("Create a base graph first.")
    else:
        st.subheader("Second Graph Input")
        directed2 = st.checkbox("Second graph directed?", value=G1.is_directed())
        multigraph2 = st.checkbox("Second graph multigraph?", value=isinstance(G1, (nx.MultiGraph, nx.MultiDiGraph)))
        mode2 = st.selectbox("Input method", ["Type edges", "Paste adjacency list", "Paste adjacency matrix"])
        t2 = st.text_area("Second graph data", "0 1\n1 2\n2 0\n1 3" if mode2 == "Type edges" else "", height=150)
        if st.button("Check Isomorphism"):
            if mode2 == "Type edges":
                G2 = parse_edge_list(t2, directed=directed2, multigraph=multigraph2)
            elif mode2 == "Paste adjacency list":
                G2 = parse_adjacency_list(t2, directed=directed2, multigraph=multigraph2)
            else:
                G2 = parse_adjacency_matrix(t2, directed=directed2, multigraph=multigraph2)

            st.write(f"G1: |V|={G1.number_of_nodes()}, |E|={G1.number_of_edges()}")
            st.write(f"G2: |V|={G2.number_of_nodes()}, |E|={G2.number_of_edges()}")

            prelim = True
            if G1.number_of_nodes() != G2.number_of_nodes():
                st.error("Different number of vertices — NOT isomorphic.")
                prelim = False
            elif G1.number_of_edges() != G2.number_of_edges():
                st.error("Different number of edges — NOT isomorphic.")
                prelim = False
            elif sorted([d for _, d in G1.degree()]) != sorted([d for _, d in G2.degree()]):
                st.error("Different degree sequences — NOT isomorphic.")
                prelim = False

            if prelim:
                iso = nx.is_isomorphic(G1, G2)
                st.success(f"Isomorphic: {bool(iso)}")
            colA, colB = st.columns(2)
            with colA:
                st.subheader("Graph A")
                visualize_graph(G1, layout_seed=st.session_state.layout_seed, title="Graph A")
            with colB:
                st.subheader("Graph B")
                visualize_graph(G2, layout_seed=st.session_state.layout_seed, title="Graph B")

# Panel 6: Operations
elif menu.startswith("6"):
    st.header("Graph Operations")
    import copy
    G = copy.deepcopy(st.session_state.get("G_original", None))
    if G is None:
        st.warning("Create a graph first.")
    else:
        op = st.selectbox("Operation", ["Union (compose)", "Intersection", "Complement (current)", "Cartesian Product"])
        need_second = op in ["Union (compose)", "Intersection", "Cartesian Product"]
        if need_second:
            st.subheader("Second Graph")
            directed2 = st.checkbox("Second graph directed?", value=G.is_directed(), key="ops_directed2")
            multigraph2 = st.checkbox("Second graph multigraph?", value=isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)), key="ops_multigraph2")
            mode2 = st.selectbox("Second graph input", ["Type edges", "Paste adjacency list", "Paste adjacency matrix"], key="ops_mode2")
            t2 = st.text_area("Second graph data", "0 1\n1 2\n2 0\n1 3" if mode2 == "Type edges" else "", height=130, key="ops_text2")

        if st.button("Run Operation"):
            if need_second:
                if mode2 == "Type edges":
                    H = parse_edge_list(t2, directed=directed2, multigraph=multigraph2)
                elif mode2 == "Paste adjacency list":
                    H = parse_adjacency_list(t2, directed=directed2, multigraph=multigraph2)
                else:
                    H = parse_adjacency_matrix(t2, directed=directed2, multigraph=multigraph2)

            # Ensure both graphs have the same node set for intersection
            if need_second and op == "Intersection":
                all_nodes = set(G.nodes()) | set(H.nodes())
                for n in all_nodes:
                    if n not in G:
                        G.add_node(n)
                    if n not in H:
                        H.add_node(n)

            if op == "Union (compose)":
                # If both graphs have weights for the same edge, keep the max weight
                R = nx.compose(G, H)
                for u, v in R.edges():
                    w1 = G[u][v].get('weight') if G.has_edge(u, v) else None
                    w2 = H[u][v].get('weight') if H.has_edge(u, v) else None
                    if w1 is not None and w2 is not None:
                        R[u][v]['weight'] = max(w1, w2)
                    elif w1 is not None:
                        R[u][v]['weight'] = w1
                    elif w2 is not None:
                        R[u][v]['weight'] = w2
            elif op == "Intersection":
                # Only keep edges present in both, and keep the min weight if both have weights
                R = nx.intersection(G, H)
                for u, v in R.edges():
                    w1 = G[u][v].get('weight') if G.has_edge(u, v) else None
                    w2 = H[u][v].get('weight') if H.has_edge(u, v) else None
                    if w1 is not None and w2 is not None:
                        R[u][v]['weight'] = min(w1, w2)
                    elif w1 is not None:
                        R[u][v]['weight'] = w1
                    elif w2 is not None:
                        R[u][v]['weight'] = w2
            elif op == "Complement (current)":
                st.info("Complement is always computed on a simple undirected copy of the current graph.")
                ug = nx.Graph(G.to_undirected()) if G.is_directed() else nx.Graph(G)
                R = nx.complement(ug)
            elif op == "Cartesian Product":
                # For weights, set product of weights if both edges have weights, else 1
                R = nx.cartesian_product(G, H)
                for (u1, u2), (v1, v2) in R.edges():
                    w1 = G[u1][v1].get('weight') if G.has_edge(u1, v1) and G[u1][v1].get('weight') is not None else 1
                    w2 = H[u2][v2].get('weight') if H.has_edge(u2, v2) and H[u2][v2].get('weight') is not None else 1
                    R[(u1, u2)][(v1, v2)]['weight'] = w1 * w2
            else:
                st.stop()

            st.session_state.G = R
            st.success(f"Operation done. New graph: |V|={R.number_of_nodes()}, |E|={R.number_of_edges()}")
            visualize_graph(R, layout_seed=st.session_state.layout_seed, title="Operation Result")

# Panel 7: Graph-Valued Functions / Properties
elif menu.startswith("7"):
    st.header("Graph-Valued Functions & Properties")
    import copy
    G = copy.deepcopy(st.session_state.get("G_original", None))
    if G is None:
        st.warning("Create a graph first.")
    else:
        degree_and_basic_stats(G)
        st.subheader("Optional: Shortest Paths (small connected graphs)")
        if not G.is_directed() and G.number_of_nodes() <= 12:
            try:
                if nx.is_connected(G):
                    paths = dict(nx.all_pairs_shortest_path_length(G))
                    nodes = list(sorted(G.nodes(), key=str))
                    # render a distance matrix
                    import pandas as pd
                    mat = []
                    for i, a in enumerate(nodes):
                        row = []
                        for b in nodes:
                            row.append(paths[a].get(b, np.inf))
                        mat.append(row)
                    df = pd.DataFrame(mat, index=[str(x) for x in nodes], columns=[str(x) for x in nodes])
                    st.dataframe(df)
                    st.write(f"Diameter: {nx.diameter(G)} | Radius: {nx.radius(G)} | Center: {nx.center(G)} | Periphery: {nx.periphery(G)}")
            except Exception:
                pass
        st.subheader("Cliques / Independent Set (quick look)")
        try:
            cliques = list(nx.find_cliques(G)) if not G.is_directed() else []
            if cliques:
                max_clique = max(cliques, key=len)
                st.write(f"Largest clique: {max_clique} (size {len(max_clique)}) | Total cliques: {len(cliques)}")
        except Exception:
            st.write("Clique computation skipped/failed.")
        try:
            if not G.is_directed():
                indep = nx.maximal_independent_set(G)
                st.write(f"Maximal independent set (approx): {indep} (size {len(indep)})")
        except Exception:
            st.write("Independent set computation skipped/failed.")
        st.subheader("Visualize")
        visualize_graph(G, layout_seed=st.session_state.layout_seed, title="Current Graph")

# Panel 8: Export / Reset
elif menu.startswith("8"):
    st.header("Export / Reset")
    G = st.session_state.G
    if G is None:
        st.warning("No graph in memory.")
    else:
        # Export edge list
        edges_str = io.StringIO()
        for e in G.edges(data=True):
            if len(e) == 3 and "weight" in e[2]:
                edges_str.write(f"{e[0]} {e[1]} {e[2]['weight']}\n")
            else:
                edges_str.write(f"{e[0]} {e[1]}\n")
        st.download_button("Download Edge List (.txt)", edges_str.getvalue(), file_name="graph_edges.txt", mime="text/plain")

        # Export GraphML (simple graphs only)
        try:
            buf = io.BytesIO()
            nx.write_graphml(G, buf)
            st.download_button("Download GraphML (.graphml)", buf.getvalue(), file_name="graph.graphml", mime="application/xml")
        except Exception:
            st.caption("GraphML export requires a simple graph with GraphML-compatible attributes.")

        visualize_graph(G, layout_seed=st.session_state.layout_seed, title="Current Graph")
    if st.button("Reset (clear graph)"):
        st.session_state.G = None
        st.success("Cleared.")
