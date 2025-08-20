import json

import numpy as np
from scipy.optimize import linear_sum_assignment

from formal_reasoning import Task, Truth, reasoning, TFS


def parsing_json(json_output):
    if "premise_1" in json_output:
        premise_1 = json_output["premise_1"]
        if ("s" not in premise_1
                or "o" not in premise_1
                or "cp" not in premise_1
                or "f" not in premise_1
                or "c" not in premise_1
                or "eb" not in premise_1):
            return None, None, []
        else:
            if "premise_2" in json_output:
                premise_2 = json_output["premise_2"]
                if ("s" not in premise_2
                        or "o" not in premise_2
                        or "cp" not in premise_2
                        or "f" not in premise_2
                        or "c" not in premise_2
                        or "eb" not in premise_2):
                    return None, None, []
                else:
                    results = []
                    if "results" in json_output:
                        for each in json_output["results"]:
                            if ("s" not in each
                                    or "o" not in each
                                    or "cp" not in each
                                    or "f" not in each
                                    or "c" not in each
                                    or "eb" not in each
                                    or "r" not in each):
                                return None, None, []
                            results.append(each)
                        return premise_1, premise_2, results
            else:
                return None, None, []
    else:
        return None, None, []


def similarity_score(a, b):
    diff = abs(a - b)

    if diff <= 0.05:
        return 1.0

    elif diff <= 0.2:
        norm_diff = (diff - 0.05) / 0.15
        return 0.1 + 0.9 * (1 - norm_diff) ** 2

    else:
        return 0.1


def grade_0(json_output):
    premise_1, premise_2, results = parsing_json(json_output)

    if premise_1 is None or premise_2 is None or not results:
        return 0.1

    J1 = Task(premise_1["s"],
              premise_1["o"],
              premise_1["cp"],
              Truth(premise_1["f"], premise_1["c"]),
              set(premise_1["eb"]))
    J2 = Task(premise_2["s"],
              premise_2["o"],
              premise_2["cp"],
              Truth(premise_2["f"], premise_2["c"]),
              set(premise_2["eb"]))
    Rs = reasoning(J1, J2)

    A, B = np.zeros((len(results), len(Rs))), np.zeros((len(results), len(Rs)))

    for i, each_llm_output in enumerate(results):
        for j, each_label in enumerate(Rs):
            tmp_A, tmp_B = 0, 0
            for each_key in ["s", "o", "cp", "eb", "r"]:
                if each_llm_output is not None and each_key in each_llm_output and each_llm_output[each_key] == \
                        each_label.to_json()[each_key]:
                    tmp_A += 5
                tmp_B += 5

            # grade on the truth-value
            if "f" in each_llm_output and "c" in each_llm_output:
                tmp_A += similarity_score(each_llm_output["f"], each_label.to_json()["f"]) * 25
                tmp_A += similarity_score(each_llm_output["c"], each_label.to_json()["c"]) * 25
            tmp_B += 50
            A[i, j] = tmp_A
            B[i, j] = tmp_B

    a, b = 0, 0
    scores = A / B
    row_idx, col_idx = linear_sum_assignment(-scores)
    for row_i, col_i in zip(row_idx, col_idx):
        a += A[row_i, col_i]
        b += B[row_i, col_i]

    return max(0.1, a / (b + 1e-5))


def grade_1(llm_json_output, label_json_output):
    premise_1_llm, premise_2_llm, _ = parsing_json(llm_json_output)
    premise_1_label, premise_2_label, _ = parsing_json(label_json_output)

    a, b = 0, 0

    for i, premises in enumerate([[premise_1_label, premise_1_llm], [premise_2_label, premise_2_llm]]):
        for each_key in ["s", "o", "cp", "eb"]:
            if premises[1] is not None and each_key in premises[1] and premises[0][each_key] == premises[1][each_key]:
                a += 5
            b += 5

        for each_key in ["f", "c"]:
            if premises[1] is not None and each_key in premises[1] and abs(
                    premises[0][each_key] - premises[1][each_key]) <= 0.2:
                a += 5
            b += 5

    return max(0.1, a / (b + 1e-5))


if __name__ == "__main__":
    from_LLM = '{"premise_1": {"s": "ID_33382", "o": "ID_88697", "cp": "-->", "f": 0.009, "c": 0.9, "eb": [5698, 6916]}, "premise_2": {"s": "ID_33382", "o": "ID_84647", "cp": "-->", "f": 0.79, "c": 0.9, "eb": [6416]}, "results": [{"s": "ID_84647", "o": "ID_88697", "cp": "<->", "f": 0.090, "c": 0.931, "eb": [6416, 6916], "r": "com"}]}'
    label = '{"premise_1": {"s": "ID_33382", "o": "ID_88697", "cp": "-->", "f": 0.009, "c": 0.9, "eb": [5698, 6916]}, "premise_2": {"s": "ID_33382", "o": "ID_84647", "cp": "-->", "f": 0.79, "c": 0.9, "eb": [6416]}, "results": [{"s": "ID_84647", "o": "ID_88697", "cp": "<->", "f": 0.009, "c": 0.391, "eb": [5698, 6416, 6916], "r": "com"}]}'
    print(grade_0(json.loads(from_LLM)))
    print(grade_1(json.loads(from_LLM), json.loads(label)))
