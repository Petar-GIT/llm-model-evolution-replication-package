import sys
import re
import pandas as pd

from eval_completions import (
    get_last_graph,
    get_anchored_completion_graph,
    V_JSON,
    IsoGraph,
    only_keep_change_types,
    remove_all_labels,
)

def looks_like_raw_edge(s: str) -> bool:
    if s is None or pd.isna(s):
        return False

    s = str(s).strip()

    if "```" in s:
        return False

    if "$$" in s:
        return False

    # SINGLE_EDGE: use first non-empty line
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if len(lines) != 1:
        return False

    line = lines[0]

    # Accept both:
    # e 3 2 ...
    # 3 2 ...
    if line.startswith("e "):
        return re.match(r"^e\s+\d+\s+\d+\s+", line) is not None

    return re.match(r"^\d+\s+\d+\s+", line) is not None


def set_false(data, idx):
    data.loc[idx, "paper_format"] = False
    data.loc[idx, "paper_structure"] = False
    data.loc[idx, "paper_change_structure"] = False
    data.loc[idx, "paper_type_structure"] = False
    data.loc[idx, "type_isomorphic_completion"] = False
    data.loc[idx, "type_subgraph_isomorphic_generated_in_gt"] = False
    data.loc[idx, "type_subgraph_isomorphic_gt_in_generated"] = False
    data.loc[idx, "at_least_one_correct_edge"] = False


def main(input_path, output_path, synthetic_dataset):
    synthetic_dataset = synthetic_dataset == "True"

    data = pd.read_json(input_path, lines=True)

    data["paper_format"] = False
    data["paper_structure"] = False
    data["paper_change_structure"] = False
    data["paper_type_structure"] = False
    data["type_isomorphic_completion"] = False
    data["type_subgraph_isomorphic_generated_in_gt"] = False
    data["type_subgraph_isomorphic_gt_in_generated"] = False
    data["at_least_one_correct_edge"] = False

    for idx, example in data.iterrows():
        completion_string = example.get("completion_string", None)

        # strict raw format check BEFORE normalization
        if not looks_like_raw_edge(completion_string):
            set_false(data, idx)
            continue

        try:
            prompt, _, _ = get_last_graph(example, synthetic_dataset)

            completion_gt = str(example["completion"])
            if not completion_gt.startswith("e"):
                completion_gt = "e" + completion_gt

            graph_gt, _ = get_anchored_completion_graph(
                prompt, completion_gt, synthetic_dataset, version=V_JSON
            )

            completion_string = str(completion_string).strip()

            if completion_string.startswith("e "):
                generated_completion = completion_string + "\n\n$$"
            else:
                generated_completion = "e " + completion_string.lstrip() + "\n\n$$"

            graph_generated, _ = get_anchored_completion_graph(
                prompt, generated_completion, synthetic_dataset, version=V_JSON
            )

        except Exception as e:
            print(f"WARN: sample {idx} could not be parsed: {e}")
            set_false(data, idx)
            continue

        data.loc[idx, "paper_format"] = True

        graph_gt = IsoGraph(graph_gt)
        graph_generated = IsoGraph(graph_generated)

        # Paper-style metrics:
        # Structure = graph structure without labels
        graph_generated_no_labels = IsoGraph(remove_all_labels(graph_generated))
        graph_gt_no_labels = IsoGraph(remove_all_labels(graph_gt))
        structure = graph_gt_no_labels == graph_generated_no_labels

        # Change structure = only change labels
        graph_generated_change = IsoGraph(only_keep_change_types(graph_generated))
        graph_gt_change = IsoGraph(only_keep_change_types(graph_gt))
        change_structure = graph_generated_change == graph_gt_change

        # Type structure = full type-isomorphic comparison
        type_structure = graph_gt == graph_generated

        # Original synthetic metrics
        type_isomorphic_completion = graph_gt == graph_generated

        from networkx.algorithms.isomorphism import DiGraphMatcher

        graph_matcher_generated_in_gt = DiGraphMatcher(graph_gt, graph_generated)
        graph_matcher_gt_in_generated = DiGraphMatcher(graph_generated, graph_gt)

        subgraph_generated_in_gt = (
            False if graph_generated.size() == 0
            else graph_matcher_generated_in_gt.subgraph_is_isomorphic()
        )

        subgraph_gt_in_generated = (
            graph_matcher_gt_in_generated.subgraph_is_isomorphic()
        )

        # At least one edge
        at_least_one_edge = False
        for e1 in graph_gt.edges(data=True):
            for e2 in graph_generated.edges(data=True):
                if e1[2] == e2[2]:
                    at_least_one_edge = True
                    break
            if at_least_one_edge:
                break   
            if at_least_one_edge:
                break

        data.loc[idx, "type_isomorphic_completion"] = type_isomorphic_completion
        data.loc[idx, "type_subgraph_isomorphic_generated_in_gt"] = subgraph_generated_in_gt
        data.loc[idx, "type_subgraph_isomorphic_gt_in_generated"] = subgraph_gt_in_generated
        data.loc[idx, "at_least_one_correct_edge"] = at_least_one_edge
        data.loc[idx, "paper_structure"] = structure
        data.loc[idx, "paper_change_structure"] = change_structure
        data.loc[idx, "paper_type_structure"] = type_structure

    out_csv = output_path + ".csv"
    data.to_csv(out_csv, index=False)

    print("\n--- PAPER METRICS ---")
    print("format:", (data["paper_format"] == True).mean() * 100)
    print("structure:", (data["paper_structure"] == True).mean() * 100)
    print("change_structure:", (data["paper_change_structure"] == True).mean() * 100)
    print("type_structure:", (data["paper_type_structure"] == True).mean() * 100)

    print("\n--- SYNTHETIC METRICS ---")
    print("type_iso:", (data["type_isomorphic_completion"] == True).mean() * 100)
    print("gen_in_gt:", (data["type_subgraph_isomorphic_generated_in_gt"] == True).mean() * 100)
    print("gt_in_gen:", (data["type_subgraph_isomorphic_gt_in_generated"] == True).mean() * 100)
    print(">=1 edge:", (data["at_least_one_correct_edge"] == True).mean() * 100)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python paper_style_eval.py input.jsonl output_path True/False")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2], sys.argv[3])