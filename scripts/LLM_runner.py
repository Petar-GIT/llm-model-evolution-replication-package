import sys
import os
import time
import logging
import pandas as pd

from common import ask_for_proceed
from config import model_prices, load_openai_key
from llm_clients import OpenAIClient, GeminiClient, DeepSeekAzureClient, OpenRouterClient, QwenClient

LOGGER = logging.getLogger("LLM-Runner")

SLEEP_PERIOD = 5

CHAT_MODEL_INSTRUCTION = """
You are an assistant that is given a list of change graphs in an edge format. That is, the graph is given edge by edge. The graphs are directed, labeled graphs. An edge is serialized as
"e src_id tgt_id edge_label src_label tgt_label"

Labels are dictionaries. If a node appears in more than one edge, the second time it appears it is replaced by "_" to avoid repetition. 

E.g.:
e 0 1 a b bar
e 1 2 bla _ foo

The second edge here would be equivalent to:
"e 1 2 bla bar foo"

There are some change graphs given as examples. Graphs are separated by "\n\n$$\n---\n".

The last graph in this list of graphs is not yet complete. Exactly one edge is missing. 
Your task is it, to complete the last graph by guessing the last edge. You can guess this typically by looking at the examples and trying to deduce the patterns in the examples. Give exactly ONE missing edge in the format
"e src_id tgt_id edge_label src_label tgt_label"

Do NOT explain anything.
Do NOT repeat existing edges.
Only output the missing edge.
"e src_id tgt_id edge_label src_label tgt_label". Note that the beginning "e" is already part of the prompt.
"""

CHAT_MODEL_INSTRUCTION_MULTI_EDGE = """
You are an assistant that is given a list of change graphs in an edge format. That is, the graph is given edge by edge. The graphs are directed, labeled graphs. An edge is serialized as
"e src_id tgt_id edge_label src_label tgt_label"

Labels are dictionaries or concatenations of change type and node/edge type. If a node appears in more than one edge, the second time it appears it can be replaced by "_" to avoid repetition. 

E.g.:
e 0 1 a b bar
e 1 2 bla _ foo

The second edge here would be equivalent to:
"e 1 2 bla bar foo"

There are some change graphs given as examples. Graphs are separated by "\n\n$$\n---\n".

The last graph in this list of graphs is not yet complete. Some edges are missing. 
Your task is it, to complete the last graph by guessing the missing edges. You can guess this typically by looking at the examples and trying to deduce the patterns in the examples. Give the missing edges in the format
"e src_id tgt_id edge_label src_label tgt_label". Note that the beginning "e" is already part of the prompt. After the last edge of the change graph, add two new lines.
"""


def create_client(provider: str, model_id: str):
    if provider == "openai":
        load_openai_key(False)
        return OpenAIClient(model_id)

    if provider == "gemini":
        return GeminiClient(model_id)

    if provider == "deepseek":
        return DeepSeekAzureClient(model_id)

    if provider == "openrouter":
        return OpenRouterClient(model_id)

    if provider == "qwen":
        return QwenClient(model_id)

    raise ValueError(f"Unsupported provider: {provider}")


def estimate_price(input_df: pd.DataFrame, model_id: str) -> float:
    if "total_token_count" not in input_df.columns:
        return 0.0

    total_tokens = sum(input_df["total_token_count"])
    return total_tokens * model_prices.get(model_id, 0)


def main(provider: str, model_id: str, path_input_file: str, path_output_file: str, mode: str):
    multi_edge = mode == "MULTI_EDGE"

    input_df = pd.read_json(path_input_file, lines=True)

    expected_price = estimate_price(input_df, model_id)
    LOGGER.info(f"The total cost for this experiment are expected to be: {expected_price}USD")

    ask_for_proceed()

    client = create_client(provider, model_id)

    if os.path.dirname(path_output_file):
        os.makedirs(os.path.dirname(path_output_file), exist_ok=True)

    output_df = input_df.copy()
    instruction = CHAT_MODEL_INSTRUCTION_MULTI_EDGE if multi_edge else CHAT_MODEL_INSTRUCTION

    for counter, (idx, row) in enumerate(input_df.iterrows(), start=1):
        prompt = row["prompt"]

        total_tokens, completion_tokens, completion_string = client.generate(
            prompt=prompt,
            instruction=instruction,
            multi_edge=multi_edge,
        )

        output_df.at[idx, "total_tokens"] = total_tokens
        output_df.at[idx, "completion_tokens"] = completion_tokens
        output_df.at[idx, "completion_string"] = completion_string
        output_df.at[idx, "provider"] = provider
        output_df.at[idx, "model_name"] = model_id

        output_df.to_json(path_output_file + "_snapshot_" + str(counter), orient="records", lines=True)

        LOGGER.info(f"Finished sample {counter}/{len(input_df)}")
        time.sleep(SLEEP_PERIOD)

    output_df.to_json(path_output_file, orient="records", lines=True)
    LOGGER.info(f"Saved results to {path_output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) == 6:
        main(
            provider=sys.argv[1],
            model_id=sys.argv[2],
            path_input_file=sys.argv[3],
            path_output_file=sys.argv[4],
            mode=sys.argv[5],
        )
    else:
        LOGGER.info(
            "Call like: python LLM_runner.py [provider] [model_id] [input_path] [output_path] [SINGLE_EDGE/MULTI_EDGE]"
        )