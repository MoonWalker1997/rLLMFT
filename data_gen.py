import argparse
import csv
import json
import os
import random

from config import cs
from formal_reasoning import (_inheritance_templates, _similarity_templates, _truth_categories,
                              _inheritance_templates_q, _similarity_templates_q, Generator)


def generate_raw_prompt(premises_str, question_str, results):
    return [
        "A conversation between User and Assistant. The user provides the background knowledge and asks a question. "
        "The Assistant needs to solve it showing logical reasoning steps. "
        "**Only** derive the asked conclusion. Output the valid JSON string **directly**. ",
        f"Suppose we have the premises: {' '.join(each for each in premises_str)} ",
        f"Could you please answer the question: {question_str}? ",
        json.dumps({"step 1": results[0], "step 2": results[1]})
    ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--num_data", type=int, default=100, help="Number of data samples to generate")
    parser.add_argument("--num_templates", type=int, default=10, help="Number of templates to use")
    parser.add_argument("--num_categories", type=int, default=5, help="Number of categories to use")
    parser.add_argument("--num_models", type=int, default=3, help="Number of LLM models used")
    parser.add_argument("--random_seed", type=int, default=39, help="Random seed")

    args = parser.parse_args()

    num_data = args.num_data
    num_templates = args.num_templates
    num_categories = args.num_categories
    num_models = args.num_models
    random_seed = args.random_seed

    random.seed(random_seed)
    _inh_template_indices = random.sample(range(len(_inheritance_templates)), num_templates)
    _sim_template_indices = random.sample(range(len(_similarity_templates)), num_templates)

    inheritance_templates = [_inheritance_templates[each] for each in _inh_template_indices]
    inheritance_templates_q = [_inheritance_templates_q[each] for each in _inh_template_indices]

    similarity_templates = [_similarity_templates[each] for each in _sim_template_indices]
    similarity_templates_q = [_similarity_templates_q[each] for each in _sim_template_indices]

    truth_categories = [[each[0], each[1], each[2][:num_categories]] for each in _truth_categories]

    G = Generator(random_seed)
    os.makedirs("data", exist_ok=True)
    os.makedirs("test_record", exist_ok=True)

    for model_index in range(num_models):

        raw_data = G.gen_random_reasoning(cs,
                                          num_data,
                                          inheritance_templates, inheritance_templates_q,
                                          similarity_templates, similarity_templates_q,
                                          truth_categories,
                                          model_index, num_models, False)
        table = [generate_raw_prompt(*each) for each in raw_data]
        with open(f"data/data_table_{model_index}_{num_models}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Introduction", "Premise", "Question", "Answer"])
            for row in table:
                writer.writerow(row)
            print(f">- data/data_table_{model_index}_{num_models}.csv generated")

    raw_data = G.gen_random_reasoning(cs,
                                      num_data,
                                      inheritance_templates, inheritance_templates_q,
                                      similarity_templates, similarity_templates_q,
                                      truth_categories,
                                      -1, -1, True)
    table = [generate_raw_prompt(*each) for each in raw_data]
    with open("data/data_table_test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Introduction", "Premise", "Question", "Answer"])
        for row in table:
            writer.writerow(row)
        print(">- data/data_table_test.csv generated")

    print(">- finished")
