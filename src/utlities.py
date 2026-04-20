import os
import json
import re

from collections import defaultdict, deque
from difflib import SequenceMatcher

from transformers import AutoTokenizer
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def count_tokens(text):
    """
    Count tokens in given text using BERT-based-uncased tokenizer
    :param text:
    :return:
    """
    return len(tokenizer.encode(text))


def build_i_to_l_mapping(nodes, edges):
    """
    This function develops the mapping between the proposition's inference node and its locution node.

    :param nodes:
    :param edges:
    :return:
    """
    id_to_node = {node["nodeID"]: node for node in nodes}
    child_map = defaultdict(list)
    parent_map = defaultdict(list)

    for edge in edges:
        child_map[edge["fromID"]].append(edge["toID"])
        parent_map[edge["toID"]].append(edge["fromID"])

    i_to_l_map = {}
    # for each "I" node parse its parent lines to find its "L"
    for node in nodes:
        if node["type"] == "I":
            visited = set()
            queue = deque([(node["nodeID"], [])])
            while queue:
                current, path = queue.popleft()
                visited.add(current)
                for parent in parent_map[current]:
                    if parent in visited:
                        continue
                    if id_to_node[parent]["type"] == "L":
                        i_to_l_map[node["nodeID"]] = parent
                        queue.clear()
                        break
                    queue.append((parent, path + [parent]))
    return i_to_l_map


def normalize_with_map(original: str):
    """
    Normalize text to lowercase alnum+space only, and return:
    - norm: normalized string
    - norm_to_orig: list mapping each norm char index -> original char index
    """
    norm_chars = []
    norm_to_orig = []
    for i, ch in enumerate(original):
        ch_low = ch.lower()
        if ch_low.isalnum() or ch_low.isspace():
            # collapse whitespace later, but keep mapping now
            norm_chars.append(ch_low if ch_low.isalnum() else " ")
            norm_to_orig.append(i)

    norm = "".join(norm_chars)
    # collapse spaces while maintaining a mapping
    collapsed = []
    collapsed_map = []
    prev_space = False
    for j, ch in enumerate(norm):
        if ch == " ":
            if prev_space:
                continue
            prev_space = True
            collapsed.append(" ")
            collapsed_map.append(norm_to_orig[j])
        else:
            prev_space = False
            collapsed.append(ch)
            collapsed_map.append(norm_to_orig[j])

    norm = "".join(collapsed).strip()
    # If stripped, adjust mapping accordingly
    # Find first/last non-space in collapsed
    if not norm:
        return "", []
    first = next((k for k, c in enumerate(collapsed) if c != " "), None)
    last = next((k for k in range(len(collapsed) - 1, -1, -1) if collapsed[k] != " "), None)
    norm_to_orig_final = collapsed_map[first:last + 1]
    return norm, norm_to_orig_final


def normalize_node(text: str) -> str:
    """
    Node normalization and basic cleaning:
    - remove speaker prefix
    - lowercase
    - remove non-alnum (keep spaces)
    - collapse whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r"^[^:]{1,30}\s*:\s*", "", text)
    text = re.sub(r"\[deleted\]", "", text, flags=re.IGNORECASE)
    text = text.lower()

    # smart quotes/dashes normalize
    text = (text.replace("’", "'").replace("‘", "'")
            .replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-"))

    # keep only alnum and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def best_span_match(
        node_norm: str, conv_norm: str, cursor_norm: int,
        min_ratio: float = 0.62,
        max_scan_tokens: int = 220,
        len_fuzz: int = 3
        ):
    """
    Find best matching span in conv_norm for node_norm starting at/after cursor_norm.
    Returns (start_norm, end_norm, ratio) or (None, None, 0.0). Useful for aligning the spans with the conversation text.

    :param node_norm:
    :param conv_norm:
    :param cursor_norm:
    :param min_ratio:
    :param max_scan_tokens:
    :param len_fuzz:
    :return:
    """

    if not node_norm:
        return None, None, 0.0

    # Fast path: exact substring
    exact = conv_norm.find(node_norm, cursor_norm)
    if exact != -1:
        return exact, exact + len(node_norm), 1.0

    # Tokenize conv and build token start indices (in conv_norm chars)
    # Also respect cursor by starting token search near it.
    conv_tokens = conv_norm.split()
    if not conv_tokens:
        return None, None, 0.0

    # Build token -> char start positions
    starts = []
    pos = 0
    for t in conv_tokens:
        # find next occurrence from pos (safe due to split reconstruction)
        idx = conv_norm.find(t, pos)
        starts.append(idx)
        pos = idx + len(t)

    node_tokens = node_norm.split()
    n = len(node_tokens)
    if n == 0:
        return None, None, 0.0

    min_len = max(2, n - len_fuzz)
    max_len = n + len_fuzz

    # Determine start token index based on cursor_norm
    start_tok = 0
    for i, st in enumerate(starts):
        if st >= cursor_norm:
            start_tok = i
            break

    best = (None, None, 0.0)

    # scan windows
    # limit scanning to avoid quadratic blowups
    scan_end = min(len(conv_tokens), start_tok + max_scan_tokens)

    for i in range(start_tok, scan_end):
        for L in range(min_len, max_len + 1):
            j = i + L
            if j > len(conv_tokens):
                break

            window_text = " ".join(conv_tokens[i:j])

            # quick prune: require at least 2 overlapping tokens
            overlap = len(set(node_tokens) & set(conv_tokens[i:j]))
            if overlap < 2 and n >= 4:
                continue

            ratio = SequenceMatcher(None, node_norm, window_text).ratio()
            if ratio > best[2]:
                # compute char span in conv_norm
                start_char = starts[i]
                end_char = starts[j - 1] + len(conv_tokens[j - 1])
                best = (start_char, end_char, ratio)

    if best[2] >= min_ratio:
        return best
    return None, None, best[2]


def process_argument_graph_recursive(data):
    nodes = data["nodes"]
    edges = data["edges"]
    id_to_node = {node["nodeID"]: node for node in nodes}

    relation_types = ("RA", "CA")

    def get_text(node_id):
        return id_to_node[node_id]["text"]

    def get_type(node_id):
        return id_to_node[node_id]["type"]

    def get_parents(nid):
        return [e["fromID"] for e in edges if e["toID"] == nid]

    def get_children(nid):
        return [e["toID"] for e in edges if e["fromID"] == nid]

    # Build I-to-L mapping
    inf_prop_mapping = build_i_to_l_mapping(nodes, edges)

    # Identify valid I nodes (must participate in RA/CA relation)
    valid_i_nodes = set()

    for edge in edges:
        from_type = get_type(edge["fromID"])
        to_type = get_type(edge["toID"])
        if from_type == "I" and to_type in relation_types:
            if edge["fromID"] not in valid_i_nodes:
                valid_i_nodes.add(edge["fromID"])
        elif to_type == "I" and from_type in relation_types:
            if edge["toID"] not in valid_i_nodes:
                valid_i_nodes.add(edge["toID"])

    # Select corresponding L nodes based on valid I-node involvement
    final_nodes = []
    added_l_ids = set()

    for i_id in valid_i_nodes:
        if i_id not in inf_prop_mapping:
            continue
        l_id = inf_prop_mapping[i_id]
        ya_ids = [pid for pid in get_parents(i_id) if get_type(pid) == "YA"]
        # the "I" node could have more than one "YA"

        # Select the YA whose parent is of type L for reasoning purposes
        selected_ya = None
        for ya_id in ya_ids:
            ya_parents = get_parents(ya_id)
            if any(get_type(p) == "L" for p in ya_parents):
                selected_ya = ya_id
                break

        # the first condition marks the node that provides the reasoning while the second condition enforces that the final node is unique and not processed before.
        if selected_ya and l_id not in added_l_ids:
            final_nodes.append({
                "reason": get_text(selected_ya),
                "node_id": l_id,
                "text": get_text(l_id)
                }
                )
            added_l_ids.add(l_id)

    # Extract rels — only if RA/CA has exactly one I source and one I target
    final_rels = []
    for node in nodes:
        # only process nodes that are of relevant types.
        if node["type"] in relation_types:
            ra_id = node["nodeID"]
            i_sources = [pid for pid in get_parents(ra_id) if get_type(pid) == "I"]
            i_targets = [cid for cid in get_children(ra_id) if get_type(cid) == "I"]
            # an CA/RA node may more than one "I" node as source or target.
            for i_src in i_sources:
                for i_tgt in i_targets:
                    if i_src in inf_prop_mapping and i_tgt in inf_prop_mapping:
                        # Find YA parents of RA/CA that further should have a L parent
                        ya_ids = [pid for pid in get_parents(ra_id) if get_type(pid) == "YA"]
                        selected_ya = None
                        for ya_id in ya_ids:
                            if any(get_type(p) == "TA" for p in get_parents(ya_id)):
                                selected_ya = ya_id
                                break
                        ya_text = get_text(selected_ya) if selected_ya else "Unknown"
                        # save its type
                        if node["type"] == "RA":
                            relation_type = "support"
                        elif node["type"] == "CA":
                            relation_type = "attack"
                        elif node["type"] == "MA":
                            relation_type = "rephrasing"
                        else:
                            raise ValueError(f"Unknown node type: {node['type']}")

                        final_rels.append({
                            "reason": ya_text,
                            "source_id": inf_prop_mapping[i_src],
                            "target_id": inf_prop_mapping[i_tgt],
                            "relation_type": relation_type
                            }
                            )

    return final_nodes, final_rels


def validate_and_remap(final_nodes, final_rels):
    """
    Remap node IDs to be sequential, but preserve original occurrence order.

    :param final_nodes:
    :param final_rels:
    :return:
    """

    # Preserve order of first occurrence
    seen = {}
    next_id = 1
    for n in final_nodes:
        old_id = n["node_id"]
        if old_id not in seen:
            seen[old_id] = str(next_id)
            next_id += 1

    remap = seen  # mapping from old_id to new sequential id

    # Remap nodes keeping original order
    remapped_nodes = [{
        "justification": n["reason"],
        "id": remap[n["node_id"]],
        "text": n["text"]
        } for n in final_nodes]

    # Remap rels, filter out those pointing to missing nodes
    remapped_rels = [{
        "justification": r["reason"],
        "source_id": remap[r["source_id"]],
        "target_id": remap[r["target_id"]],
        "relation_type": r["relation_type"]
        } for r in final_rels if r["source_id"] in remap and r["target_id"] in remap]

    # Find unused nodes
    used_ids = set(r["source_id"] for r in remapped_rels) | set(r["target_id"] for r in remapped_rels)
    unused_l_nodes = set(remap[k] for k in remap if remap[k] not in used_ids)

    return remapped_nodes, remapped_rels, unused_l_nodes


def explain_unused_l_nodes(unused_l_ids, nodes, edges):
    """
    Additional function for debugging locution node errors.

    :param unused_l_ids:
    :param nodes:
    :param edges:
    :return:
    """
    id_to_node = {node["nodeID"]: node for node in nodes}
    reasons = {}

    def get_type(nid):
        return id_to_node[nid]["type"]

    def get_text(nid):
        return id_to_node[nid]["text"]

    def get_children(nid):
        return [e["toID"] for e in edges if e["fromID"] == nid]

    def get_parents(nid):
        return [e["fromID"] for e in edges if e["toID"] == nid]

    for l_id in unused_l_ids:
        explanation = {"node_id": l_id, "text": get_text(l_id), "issues": []}
        ya_children = [cid for cid in get_children(l_id) if get_type(cid) == "YA"]
        if not ya_children:
            explanation["issues"].append("No YA child found")
            reasons[l_id] = explanation
            continue
        ya_id = ya_children[0]
        i_children = [cid for cid in get_children(ya_id) if get_type(cid) == "I"]
        if not i_children:
            explanation["issues"].append("YA exists but no I child found")
            reasons[l_id] = explanation
            continue
        i_id = i_children[0]
        ra_ca_children = [cid for cid in get_children(i_id) if get_type(cid) in ("RA", "MA", "CA")]
        if not ra_ca_children:
            explanation["issues"].append("I exists but does not connect to RA MA, or CA")
        else:
            explanation["issues"].append("I connects to RA/CA/MA, but the relation may be incomplete or unmatched")
        reasons[l_id] = explanation

    return reasons


def develop_argument_map_from_corpus(input_dir: str, output_dir: str):
    """

    :param input_dir:
    :param output_dir:
    :return: save the argument maps json values
    """

    folder_name = os.path.basename(input_dir)
    new_directory_path = os.path.join(output_dir, folder_name)

    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)
        print(f"New directory '{folder_name}' created at {new_directory_path}")
    else:
        print(f"Directory '{folder_name}' already exists.")

    stats = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(input_dir, filename)
            filename_base = os.path.splitext(filename)[0]
            with open(filepath, 'r') as f:
                data = json.load(f)

                raw_nodes, raw_rels = process_argument_graph_recursive(data)
                remapped_nodes, remapped_rels, unused_l_nodes = validate_and_remap(raw_nodes, raw_rels)

                final_outputs = {
                    "nodes": remapped_nodes,
                    "relations": remapped_rels
                    }

            with open(os.path.join(new_directory_path, filename), 'w') as out_f:
                json.dump(final_outputs, out_f, indent=2)


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'\s+', ' ', text)
    text = text.lower()

    # normalize apostrophes
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')

    # remove speaker tags
    text = re.sub(r"^[^:]{1,30} ?: ?", "", text, flags=re.MULTILINE)

    # remove punctuation (KEEP WORDS)
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


def clean_node_text(text):
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()

    text = re.sub(r"^[^:]{1,30} ?: ?", "", text)
    text = re.sub(r"\[deleted\]", "", text, flags=re.IGNORECASE)

    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')

    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def find_node_position(node_text, conversation_text, cursor):
    if not node_text:
        return None, cursor

    pattern = re.escape(node_text.lower())
    match = re.search(pattern, conversation_text.lower()[cursor:])
    if match:
        pos = cursor + match.start()
        return pos, pos + len(node_text)

    return None, cursor


def develop_data_files(
        text_dir, json_dirs,
        min_nodes=4, max_nodes=100,
        min_relations=4, max_relations=100,
        max_conv_size=1500,
        min_conv_size=50,
        max_context_examples_size=1200,
        do_rephrasing: bool = False,
        as_explained_structure: bool = True,
        extract_context_examples: bool = True,
        ):
    """
    Converts the Bipolar argument structures object to
    :param text_dir:
    :param json_dirs:
    :param min_nodes:
    :param max_nodes:
    :param min_relations:
    :param max_relations:
    :param max_conv_size:
    :param min_conv_size:
    :param max_context_examples_size:
    :param do_rephrasing:
    :param as_explained_structure:
    :return:
    """
    folder_name = os.path.basename(text_dir)
    new_json_dir = os.path.join(json_dirs, folder_name)
    if not os.path.exists(new_json_dir):
        raise ValueError(f"The directory {new_json_dir} does not exist.")

    records = []
    total_dropped_units = 0
    total_conversations = 0
    total_conv_tokens = 0
    total_argument_units = 0
    total_supports = 0
    total_attacks = 0

    for filename in os.listdir(text_dir):
        if not filename.endswith(".json"):
            continue

        match = re.search(r"nodeset(\d+)\.json", filename)
        if not match:
            continue
        conversation_id = match.group(1)

        json_path = os.path.join(new_json_dir, filename)
        text_path = os.path.join(text_dir, filename.replace(".json", ".txt"))

        with open(json_path, "r") as f:
            graph = json.load(f)

        # check node conditions
        if not (min_nodes <= len(graph["nodes"]) < max_nodes):
            continue
        # check relation conditions
        if not (min_relations <= len(graph["relations"]) < max_relations):
            continue
        with open(text_path, 'r', encoding='utf-8') as f:
            conversation_text = f.read()

        # check length conditions
        if not (min_conv_size <= count_tokens(conversation_text) <= max_conv_size):
            continue

        # ---- Arrange nodes chronologically with REPAIR ----
        conv_original = conversation_text  # keep original (already read from file)
        conv_norm, norm_to_orig = normalize_with_map(conv_original)

        cursor_norm = 0
        valid_nodes = []
        dropped_nodes = []
        repaired_nodes = []

        for n in graph["nodes"]:
            old_id = n["id"]
            raw_text = n.get("text", "")

            node_norm = normalize_node(raw_text)

            # at this step, we clean up the text of the nodes to ensure they exist in the conversation.
            # If the node is not identified correctly in conversation, it is repaired or dropped.
            start_norm, end_norm, score = best_span_match(
                node_norm=node_norm,
                conv_norm=conv_norm,
                cursor_norm=cursor_norm,
                min_ratio=0.62,
                max_scan_tokens=220,
                len_fuzz=3
                )
            if start_norm is None:
                dropped_nodes.append({"conversation_id": conversation_id, "node_id": old_id, "text": raw_text})
                continue

            # Map normalized span back to original conversation substring
            # norm_to_orig maps each norm char index -> original char index
            orig_start = norm_to_orig[start_norm] if start_norm < len(norm_to_orig) else 0
            orig_end = norm_to_orig[end_norm - 1] + 1 if (end_norm - 1) < len(norm_to_orig) else len(conv_original)

            conv_span = conv_original[orig_start:orig_end].strip()

            # If it wasn't an exact match (score < 1.0), we repaired it
            if score < 0.999:
                repaired_nodes.append({
                    "conversation_id": conversation_id,
                    "node_id": old_id,
                    "old_text": raw_text,
                    "new_text": conv_span,
                    "score": round(score, 3)
                    }
                    )

            # Replace node text with conversational value
            n["text"] = conv_span
            n["__pos"] = orig_start  # chronological position in original conversation
            cursor_norm = end_norm  # advance in normalized space
            valid_nodes.append(n)

        # Report repairs and drops
        if repaired_nodes:
            print(f"\n[Repaired nodes] Conversation {conversation_id}")
            for r in repaired_nodes:
                print(f"  - Node {r['node_id']} (score={r['score']}): '{r['old_text']}' -> '{r['new_text']}'")

        # if dropped_nodes:
        #     print(f"\n[Dropped nodes] Conversation {conversation_id}")
        #     for d in dropped_nodes:
        #         print(f"  - Node {d['node_id']}: {d['text']}")

        graph["nodes"] = valid_nodes

        # Enforce min_nodes after dropping
        if len(graph["nodes"]) < min_nodes:
            continue

        # Sort by chronological position
        graph["nodes"].sort(key=lambda x: x["__pos"])
        for node in graph["nodes"]:
            node.pop("__pos", None)

        old_to_new_id = {}
        for idx, node in enumerate(graph["nodes"]):
            old_to_new_id[node["id"]] = str(idx)
            node["id"] = str(idx)

            illocution = node.get("justification", "").strip()
            node["explanation"] = illocution if illocution else ""

        clean_relations = []
        for rel in graph["relations"]:
            s = rel["source_id"]
            t = rel["target_id"]
            if s in old_to_new_id and t in old_to_new_id:
                rel["source_id"] = old_to_new_id[s]
                rel["target_id"] = old_to_new_id[t]
                clean_relations.append(rel)

        graph["relations"] = clean_relations

        if len(graph["relations"]) < min_relations:
            continue

        # schema
        if as_explained_structure:
            explained = {
                "argument_units": [],
                "relations": []
                }

            # develop the argument units
            for node in graph["nodes"]:
                explained["argument_units"].append({
                    "reason": node["explanation"],
                    "id": int(node["id"]),
                    "text": node["text"],
                    }
                    )

            # develop the argument relations

            # for rel in graph["relations"]:
            #     src = int(rel["source_id"])
            #     tgt = int(rel["target_id"])
            #     reason = rel.get("reason", "")
            #     if rel["relation_type"] == "supporting":
            #         explained["support_relationships"].append({
            #             "reason": reason,
            #             "source_id": src,
            #             "target_id": tgt
            #         })
            #     elif rel["relation_type"] == "attacking":
            #         explained["attack_relationships"].append({
            #             "reason": reason,
            #             "source_id": src,
            #             "target_id": tgt
            #         })
            #     else:
            #         raise ValueError(f"Unknown relation type: {rel['relation_type']}")

            # argument_data = explained

            for rel in graph["relations"]:
                src = int(rel["source_id"])
                tgt = int(rel["target_id"])

                # Skip self-referencing relations (source_id == target_id) to prevent errors
                if src == tgt:
                    print(f"Warning: Skipping self-referencing relation (source_id == target_id) in conversation"
                          f" {conversation_id}. Relation: {rel}"
                          )
                    continue

                # ensure directionality in data points. Attack is always uni-directional with target always reported
                # first in the text followed by its source. For support bidirectionality is valid, with also cases
                # that target (claim) is reported before the source (premise). To simplify the setup, we maintain
                # unidirectionality across the relations, and switch any target-source for support

                if rel["relation_type"] == "support" and src < tgt:
                    explained["relations"].append({
                        "source_id": tgt,
                        "target_id": src,
                        "relation_type": "support"
                        }
                        )

                elif rel["relation_type"] == "support" and src > tgt:
                    explained["relations"].append({
                        "source_id": src,
                        "target_id": tgt,
                        "relation_type": "support"
                        }
                        )
                elif rel["relation_type"] == "attack":
                    explained["relations"].append({
                        "source_id": src,
                        "target_id": tgt,
                        "relation_type": "attack"
                        }
                        )
                # Explicitly handle 'rephrasing' relations as they are valid types but not processed into 'relations' list.
                elif rel["relation_type"] == "rephrasing":
                    print(f"Warning: Skipping 'rephrasing' relation in conversation {conversation_id}. Relation: {rel}")
                    continue
                else:
                    raise ValueError(f"Unknown relation type: {rel['relation_type']}")

            argument_data = explained

        # ALTERNATIVE proposed schema (IGNORE: Not important)
        else:
            if do_rephrasing:
                id_to_unit = {node["id"]: {
                    "id": node["id"],
                    "text": node["text"],
                    "explanation": node["explanation"],
                    "supports": [],
                    "attacks": [],
                    "rephrases": []
                    } for node in graph["nodes"]}
            else:
                id_to_unit = {node["id"]: {
                    "id": node["id"],
                    "text": node["text"],
                    "explanation": node["explanation"],
                    "supports": [],
                    "attacks": []
                    } for node in graph["nodes"]}

            for rel in graph["relations"]:
                src = rel["source_id"]
                tgt = rel["target_id"]
                if rel["relation_type"] == "supporting" and tgt not in id_to_unit[src]["supports"]:
                    id_to_unit[src]["supports"].append(tgt)
                elif rel["relation_type"] == "attacking" and tgt not in id_to_unit[src]["attacks"]:
                    id_to_unit[src]["attacks"].append(tgt)
                elif rel["relation_type"] == "rephrasing" and do_rephrasing:
                    if tgt not in id_to_unit[src]["rephrases"]:
                        id_to_unit[src]["rephrases"].append(tgt)

            argument_data = list(id_to_unit.values())

        # check for ambiguitites and fix any errors such as duplicate relations, nested arguments, etc.
        # Function to check if a unit is a subspan of another
        def is_subspan(smaller, larger):
            return smaller.strip() in larger.strip() and smaller.strip() != larger.strip()

        def clean_argument_structure(arg_obj):

            units = arg_obj["argument_units"]
            # supports = arg_obj["support_relationships"]
            # attacks = arg_obj["attack_relationships"]
            relations = arg_obj["relations"]

            id_text_map = {unit["id"]: unit["text"] for unit in units}
            unit_items = list(id_text_map.items())

            # Identify sub-span units (keep longer)
            subspan_ids = set()
            for i, (id_i, text_i) in enumerate(unit_items):
                for j, (id_j, text_j) in enumerate(unit_items):
                    if i == j:
                        continue
                    if is_subspan(text_i, text_j):
                        if len(text_i.strip()) < len(text_j.strip()):
                            subspan_ids.add(id_i)
                        else:
                            subspan_ids.add(id_j)

            cleaned_units = [u for u in units if u["id"] not in subspan_ids]

            # Fallback if too aggressive
            if len(cleaned_units) < 2:
                return arg_obj, 0

            drop_count = len(units) - len(cleaned_units)

            # Re-index IDs
            old_to_new_id = {u["id"]: i for i, u in enumerate(cleaned_units)}
            for i, u in enumerate(cleaned_units):
                u["id"] = i

            def update_and_filter_rels(rels):
                """
                maintain chronology, directionality and remove duplicates and nested relations.
                :param rels:
                :return:
                """
                seen = set()
                updated = []
                for rel in rels:
                    s = rel["source_id"]
                    t = rel["target_id"]
                    if s in subspan_ids or t in subspan_ids:
                        continue
                    pair = tuple(sorted([s, t]))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    updated.append({
                        "source_id": old_to_new_id[s],
                        "target_id": old_to_new_id[t],
                        "type": rel["relation_type"]
                        }
                        )
                return updated

            cleaned_relations = update_and_filter_rels(relations)


            return {
                "argument_units": cleaned_units,
                "relations": cleaned_relations
                }, drop_count

        # clean up the single conversation record.
        # print(argument_data)
        cleaned_structure, dropped_count = clean_argument_structure(argument_data)

        total_dropped_units += dropped_count

        total_conversations += 1
        total_conv_tokens += count_tokens(conversation_text)

        if as_explained_structure:
            total_argument_units += len(cleaned_structure["argument_units"])

            total_supports += len([r for r in cleaned_structure["relations"] if r["type"] == "support"])
            total_attacks += len([r for r in cleaned_structure["relations"] if r["type"] == "attack"])

        else:
            total_argument_units += len(cleaned_structure)
            for unit in cleaned_structure:
                total_supports += len(unit.get("supports", []))
                total_attacks += len(unit.get("attacks", []))

        rel_counts = {"support": total_supports, "attack": total_attacks}
        if do_rephrasing:
            rel_counts["rephrasing"] = 0

        def assign_weight(rtype):
            return {"supporting": 1, "attacking": 5, "rephrasing": 0}.get(rtype, 0)

        record = {
            "conversation_id": conversation_id,
            "conversation_text": conversation_text,
            "argument_objects": cleaned_structure,
            "dropped_units": dropped_count,
            "relation_counts": rel_counts,
            "diversity_score": len([v for v in rel_counts.values() if v > 0]),
            "weighted_relations": sum(v * assign_weight(k) for k, v in rel_counts.items())
            }

        # count tokens for the record including the conv text and unit text.
        if as_explained_structure:
            record["token_count"] = count_tokens(conversation_text) + sum(
                count_tokens(n["text"]) for n in argument_data["argument_units"]
                )
        else:
            record["token_count"] = count_tokens(conversation_text) + sum(
                count_tokens(n["text"]) for n in argument_data
                )

        # save conversation and process next one
        records.append(record)

    # develop context examples
    if extract_context_examples:

        # sorted first by having both support and attack, and weighed by prioritizing attack relation.
        sorted_contexts = sorted(
            [r for r in records if r["diversity_score"] >= 2],
            key=lambda x: (-x["diversity_score"], -x["weighted_relations"])
            )

        num_desired_contexts = 5
        context_examples = []
        total_tokens = 0
        seen_ids = set()

        # Pass 1: Try to select the most diverse/weighted examples that fit all token constraints
        for candidate in sorted_contexts:
            if len(context_examples) >= num_desired_contexts:
                break
            if candidate["conversation_id"] in seen_ids:
                continue
            if candidate["token_count"] > 0.4 * max_context_examples_size:  # Skip if individual example is too large
                continue
            if total_tokens + candidate["token_count"] <= max_context_examples_size:  # Skip if total size exceeded
                context_examples.append(candidate)
                total_tokens += candidate["token_count"]
                seen_ids.add(candidate["conversation_id"])

        # Pass 2: If not enough examples, try to fill remaining slots with smaller ones from remaining candidates,
        # relaxing the individual 0.4*max_context_examples_size constraint, but still respecting total_tokens constraint.
        if len(context_examples) < num_desired_contexts:
            # Get remaining candidates, sorted by token count (smallest first)
            remaining_candidates_by_size = sorted([c for c in sorted_contexts if c["conversation_id"] not in seen_ids],
                                                  key=lambda x: x["token_count"]
                                                  )
            for candidate in remaining_candidates_by_size:
                if len(context_examples) >= num_desired_contexts:
                    break
                # Now, individual example size (0.4 * max_context_examples_size) is ignored, but total still respected
                if total_tokens + candidate["token_count"] <= max_context_examples_size:
                    context_examples.append(candidate)
                    total_tokens += candidate["token_count"]
                    seen_ids.add(candidate["conversation_id"])

        while len(context_examples) < num_desired_contexts and len(seen_ids) < len(sorted_contexts):
            smallest_unseen_candidate = None
            # Find the smallest available candidate not already added
            for candidate in sorted_contexts:  # Iterate original sorted_contexts for original sorting priority for ties
                if candidate["conversation_id"] not in seen_ids:
                    if smallest_unseen_candidate is None or candidate["token_count"] < smallest_unseen_candidate[
                        "token_count"]:
                        smallest_unseen_candidate = candidate

            if smallest_unseen_candidate:
                context_examples.append(smallest_unseen_candidate)
                total_tokens += smallest_unseen_candidate["token_count"]  # Update total, may exceed budget
                seen_ids.add(smallest_unseen_candidate["conversation_id"])
            else:
                break  # No more candidates to add, cannot reach num_desired_contexts

        context_examples = context_examples[:num_desired_contexts]  # Ensure exactly 5 if more were added (unlikely)

        context_ids = {c["conversation_id"] for c in context_examples}
        # final_records = [r for r in records if r["conversation_id"] not in context_ids]

    # temporary to not remove the context records
    final_records = records

    total_relations = total_supports + total_attacks

    support_pct = 0.0
    attack_pct = 0.0
    if total_relations > 0:
        support_pct = 100 * total_supports / total_relations
        attack_pct = 100 * total_attacks / total_relations

    stats = {
        "total_conversations": total_conversations,
        "avg_conv_length": total_conv_tokens / total_conversations,
        "total_argument_units": total_argument_units,
        "avg_argument_units_per_conversation": total_argument_units / total_conversations,
        "total_supports": total_supports,
        "avg_supports_per_conversation": total_supports / total_conversations,
        "total_attacks": total_attacks,
        "avg_attacks_per_conversation": total_attacks / total_conversations,
        "support_percentage": support_pct,
        "attack_percentage": attack_pct,
        "total_dropped_units": total_dropped_units,
        }

    for r in final_records + context_examples:
        r.pop("relation_counts", None)
        r.pop("diversity_score", None)
        r.pop("weighted_relations", None)
        r.pop("token_count", None)

    df = pd.DataFrame(final_records)
    return df, context_examples, stats
