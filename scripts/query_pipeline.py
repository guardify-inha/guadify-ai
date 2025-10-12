"""
Final pipeline:
 - Load base graph (nodes.csv, edges.csv, embeddings.npy, faiss.index)
 - Load GCN model
 - Read candidate_clauses.jsonl, create embeddings for them
 - Attach candidates to graph via FAISS top-k neighbors
 - Run GNN inference and output sorted results
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
from sentence_transformers import SentenceTransformer
import pickle

OUT_DIR = os.path.join("..", "outputs")
MODEL_PTH = os.path.join(OUT_DIR, "gnn_model.pth")
NODES_FILE = os.path.join(OUT_DIR, "nodes.csv")
EMBED_FILE = os.path.join(OUT_DIR, "embeddings.npy")
EDGES_FILE = os.path.join(OUT_DIR, "edges.csv")
INDEX_FILE = os.path.join(OUT_DIR, "faiss.index")
META_FILE = os.path.join(OUT_DIR, "faiss_meta.pkl")
LABEL_MAP_FILE = os.path.join(OUT_DIR, "label_map.json")
RESULT_JSON = os.path.join(OUT_DIR, "final_results.json")
SUMMARY_TXT = os.path.join(OUT_DIR, "summary.txt")
CANDIDATES_FILE = os.path.join(OUT_DIR, "candidate_clauses.jsonl")

EMBED_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
SIMILARITY_TOP_K = 5
SIM_EDGE_THRESH = 0.40  # similarity threshold when connecting candidate->base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- GCN Classes -----------------
class SimpleGCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = torch.nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        # add self-loop and use symmetric normalization
        I = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
        A = adj + I
        D = A.sum(1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D + 1e-6))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        x = A_norm @ x
        x = self.lin(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, in_features, hidden_features=128, num_classes=2):
        super().__init__()
        self.g1 = SimpleGCNLayer(in_features, hidden_features)
        self.g2 = SimpleGCNLayer(hidden_features, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x, adj):
        x = F.relu(self.g1(x, adj))
        x = self.dropout(x)
        x = self.g2(x, adj)
        return x

# ----------------- Load Graph -----------------
def load_base_graph():
    nodes_df = pd.read_csv(NODES_FILE)
    edges_df = pd.read_csv(EDGES_FILE)
    base_embeddings = np.load(EMBED_FILE)
    faiss_index = faiss.read_index(INDEX_FILE)
    # load meta for safe reordering check
    with open(META_FILE, "rb") as f:
        faiss_meta = pickle.load(f)
    return nodes_df, edges_df, base_embeddings, faiss_index, faiss_meta

def build_base_adj(nodes_df, edges_df):
    num_base_nodes = len(nodes_df)
    base_adj = torch.zeros((num_base_nodes, num_base_nodes), dtype=torch.float)
    chunk_id_to_idx = {row['chunk_id']: int(row['node_index']) for _, row in nodes_df.iterrows()}
    for _, row in edges_df.iterrows():
        src, dst = row['src'], row['dst']
        if src in chunk_id_to_idx and dst in chunk_id_to_idx:
            i, j = chunk_id_to_idx[src], chunk_id_to_idx[dst]
            base_adj[i, j] = float(row.get('weight', 1.0))
            base_adj[j, i] = float(row.get('weight', 1.0))
    return base_adj, chunk_id_to_idx

def make_base_features(nodes_df, base_embeddings):
    """
    Build base feature matrix by concatenating embeddings and one-hot node-type vector.
    This version tries to preserve the label ordering used during embedding creation by loading
    OUT_DIR/label_map.json (which contains the mapping tag -> numeric label).
    The one-hot vector indices are ordered by ascending numeric label values present in nodes_df
    to ensure deterministic, reproducible mapping.
    """
    # load label map if available
    if os.path.exists(LABEL_MAP_FILE):
        try:
            with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
                saved_label_map = json.load(f)  # tag -> numeric label value
            # determine the set of numeric label values actually present in nodes_df
            label_vals_in_nodes = sorted(set(int(x) for x in nodes_df['label'].unique()))
            # Determine ordering: sort by numeric label value (this preserves numeric ordering from saved map)
            ordered_label_vals = [v for v in sorted(set(saved_label_map.values())) if v in label_vals_in_nodes]
            # If any label values in nodes_df aren't in saved map, append them at end (sorted)
            missing = [v for v in label_vals_in_nodes if v not in ordered_label_vals]
            ordered_label_vals += sorted(missing)
            unique_labels = ordered_label_vals
        except Exception as e:
            print("Warning: failed to load/parse label_map.json, falling back:", e)
            unique_labels = sorted(nodes_df['label'].unique())
    else:
        unique_labels = sorted(nodes_df['label'].unique())

    # build mapping from numeric label value -> one-hot index
    label_map_for_ohe = {label_val: i for i, label_val in enumerate(unique_labels)}
    num_types = len(unique_labels)

    # map nodes_df label values to one-hot indices (if label values are numeric)
    mapped_indices = nodes_df['label'].apply(lambda v: label_map_for_ohe.get(int(v), 0))
    node_types_mapped = mapped_indices
    one_hot_types = F.one_hot(torch.tensor(node_types_mapped.values), num_classes=num_types).float()

    x_embeddings = torch.tensor(base_embeddings, dtype=torch.float)
    # ensure embedding rows count matches nodes_df rows — caller should ensure order; we attempt alignment elsewhere
    base_x = torch.cat([x_embeddings, one_hot_types], dim=1)
    return base_x, label_map_for_ohe, num_types

# ----------------- GNN Inference -----------------
def run_gnn_inference(candidate_clauses, nodes_df, edges_df, base_embeddings, faiss_index, faiss_meta):
    if not candidate_clauses:
        return []

    num_base_nodes = len(nodes_df)
    base_adj, chunk_id_to_idx = build_base_adj(nodes_df, edges_df)

    # Ensure base_embeddings align with nodes_df order using faiss_meta (chunk_id mapping)
    try:
        # faiss_meta is list of dicts with 'chunk_id'
        meta_chunk_to_idx = {m.get('chunk_id'): i for i, m in enumerate(faiss_meta)}
        reordered = np.zeros_like(base_embeddings)
        mismatch = False
        for i, row in nodes_df.iterrows():
            cid = row['chunk_id']
            if cid in meta_chunk_to_idx:
                reordered[i] = base_embeddings[meta_chunk_to_idx[cid]]
            else:
                mismatch = True
                break
        if not mismatch:
            base_embeddings = reordered
        else:
            print("⚠️ Warning: couldn't fully reorder embeddings to match nodes.csv; proceeding with original order")
    except Exception as e:
        print("⚠️ Embedding re-order check failed:", e)

    base_x, label_map_for_ohe, num_types = make_base_features(nodes_df, base_embeddings)

    # embed new clauses
    embed_model = SentenceTransformer(EMBED_MODEL)
    new_texts = [c['text'] for c in candidate_clauses]
    new_embeddings = embed_model.encode(new_texts, convert_to_numpy=True).astype("float32")

    # FAISS 연결: candidate feature에 top-1 label one-hot 추가
    faiss.normalize_L2(new_embeddings)
    D, I = faiss_index.search(new_embeddings, SIMILARITY_TOP_K)

    # nearest node one-hot mapping: faiss index -> node_index using faiss_meta
    faiss_idx_to_node_index = {}
    try:
        for idx, m in enumerate(faiss_meta):
            node_idx = None
            cid = m.get('chunk_id')
            row = nodes_df[nodes_df['chunk_id'] == cid]
            if not row.empty:
                node_idx = int(row.iloc[0]['node_index'])
            else:
                node_idx = idx  # fallback to idx
            faiss_idx_to_node_index[idx] = node_idx
    except Exception:
        faiss_idx_to_node_index = {i: i for i in range(base_embeddings.shape[0])}

    base_one_hot = base_x[:, base_embeddings.shape[1]:]
    new_features = []
    new_edges = []
    for i in range(len(new_texts)):
        # top-1 mapping
        top_idx = int(I[i][0])
        mapped_idx = faiss_idx_to_node_index.get(top_idx, top_idx)
        nearest_one_hot = base_one_hot[mapped_idx].unsqueeze(0)
        candidate_feat = torch.cat([torch.tensor(new_embeddings[i]).unsqueeze(0), nearest_one_hot], dim=1)
        new_features.append(candidate_feat)
        # add edges only if similarity >= threshold
        for j in range(SIMILARITY_TOP_K):
            sim = float(D[i, j])
            if sim >= SIM_EDGE_THRESH:
                mapped = faiss_idx_to_node_index.get(int(I[i, j]), int(I[i, j]))
                new_edges.append((num_base_nodes + i, int(mapped)))

    new_x = torch.cat(new_features, dim=0) if new_features else torch.zeros((0, base_x.shape[1]))

    # expanded graph
    expanded_x = torch.cat([base_x, new_x], dim=0)
    expanded_adj = F.pad(base_adj, (0, len(new_texts), 0, len(new_texts)))
    for u, v in new_edges:
        expanded_adj[u, v] = 1.0
        expanded_adj[v, u] = 1.0

    # debug prints
    print("DEBUG: expanded_x", expanded_x.shape)
    print("DEBUG: expanded_adj row-sum min/max", expanded_adj.sum(1).min().item(), expanded_adj.sum(1).max().item())

    # load model
    model = Net(in_features=expanded_x.shape[1], hidden_features=128, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PTH, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(expanded_x.to(device), expanded_adj.to(device))
        print("DEBUG: logits mean/std", logits.mean().item(), logits.std().item())
        probs = F.softmax(logits, dim=1)
        # assume class 1 is 'unfair' as in training; if different, user should adjust
        unfair_probs = probs[num_base_nodes:, 1].cpu().numpy()

    results = [{"article": c.get("article", "N/A"),
                "text": c.get("text"),
                "unfair_prob_gnn": round(float(unfair_probs[i]), 4)}
               for i, c in enumerate(candidate_clauses)]
    results.sort(key=lambda x: x['unfair_prob_gnn'], reverse=True)
    return results

# ----------------- Save -----------------
def save_results(results):
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("==== GNN 불공정 약관 조항 분석 결과 ====\\n\\n")
        if not results:
            f.write("분석된 위험 후보 조항이 없습니다.\\n")
        for res in results:
            prob_percent = res['unfair_prob_gnn'] * 100
            f.write(f"🚨 [위험도: {prob_percent:.2f}%] - {res['article']}\\n")
            f.write(f"   - 원문: {res['text'][:300].strip()}...\\n")
            f.write("-"*60 + "\\n")

# ----------------- Main -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_file", help="Path to candidate_clauses.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.candidate_file):
        print("Candidate file not found. Run query_and_extract.py first.")
        exit(1)

    with open(args.candidate_file, "r", encoding="utf-8") as f:
        candidates = [json.loads(line) for line in f]

    nodes_df, edges_df, base_embeddings, faiss_index, faiss_meta = load_base_graph()
    results = run_gnn_inference(candidates, nodes_df, edges_df, base_embeddings, faiss_index, faiss_meta)
    save_results(results)
    print("Done. Results saved to", RESULT_JSON, "and", SUMMARY_TXT)
