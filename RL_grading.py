import json

import json_repair
import numpy as np
from scipy.optimize import linear_sum_assignment

from formal_reasoning import Truth, TFS


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


def grade(jsons_LLM, jsons_Label):
    if "Assistant:" in jsons_LLM:
        jsons_LLM = jsons_LLM.split("Assistant:")[1].strip()
    if "Assistant:" in jsons_Label:
        jsons_Label = jsons_Label.split("Assistant:")[1].strip()

    summarization = set()

    try:
        try:
            premise_1_LLM, premise_2_LLM, results_LLM = parsing_json(json.loads(json_repair.repair_json(jsons_LLM)))
        except ValueError as e:
            print("JSON parsing/repairing failed!")
            # print("Error:", e)
            premise_1_LLM, premise_2_LLM, results_LLM = None, None, []

        # the label json is always valid
        premise_1_label, premise_2_label, results_label = parsing_json(json.loads(jsons_Label))

        # grades
        A, B = 0, 0

        # premise grading
        for i, premises in enumerate([[premise_1_label, premise_1_LLM], [premise_2_label, premise_2_LLM]]):
            for each_key in ["s", "o", "cp", "eb"]:
                if premises[1] is not None:
                    if each_key in premises[1]:
                        if premises[0][each_key] == premises[1][each_key]:
                            A += 5
                        else:
                            summarization.add(f"LLM premise {i + 1}'s key {each_key} inconsistent. "
                                              f"LLM {each_key}: {premises[1][each_key]}, "
                                              f"label {each_key}: {premises[0][each_key]}")
                    else:
                        summarization.add(f"LLM premise {i + 1}'s key {each_key} does not exist")
                else:
                    summarization.add(f"LLM premise {i + 1} does not exist")
                B += 5

            for each_key in ["f", "c"]:
                if premises[1] is not None:
                    if each_key in premises[1]:
                        if abs(premises[0][each_key] - premises[1][each_key]) <= 0.2:
                            A += 5
                        else:
                            summarization.add(f"LLM premise {i + 1}'s key {each_key} different too much. "
                                              f"LLM {each_key}: {premises[1][each_key]}, "
                                              f"label {each_key}: {premises[0][each_key]}")
                    else:
                        summarization.add(f"LLM premise {i + 1}'s key {each_key} does not exist")
                else:
                    summarization.add(f"LLM premise {i + 1} does not exist")
                B += 5

        # results grading

        num_labels = len(results_label)
        num_LLMs = len(results_LLM)
        As_matrix = np.zeros((num_labels, num_LLMs))
        Bs_matrix = np.zeros((num_labels, num_LLMs))
        summarization_matrix = [[[] for _ in range(num_LLMs)] for _ in range(num_labels)]

        for i, each_result_label in enumerate(results_label):
            for j, each_result_LLM in enumerate(results_LLM):
                tmp_as, tmp_bs = 0, 0
                # grade on the consistency
                for each_key in ["s", "o", "cp", "eb", "r"]:
                    if each_result_LLM is not None:
                        if each_key in each_result_LLM:
                            if each_result_label[each_key] == each_result_LLM[each_key]:
                                tmp_as += 5
                            else:
                                summarization_matrix[i][j].append(
                                    f"LLM result's key {each_key} inconsistent. "
                                    f"LLM result: {each_result_LLM}, "
                                    f"LLM {each_key}: {each_result_LLM[each_key]}, "
                                    f"label {each_key}: {each_result_label[each_key]}")
                        else:
                            summarization_matrix[i][j].append(f"LLM result does not have key {each_key}. "
                                                              f"LLM result: {each_result_LLM}")
                    else:
                        summarization_matrix[i][j].append(f"LLM does not have any results")
                    tmp_bs += 5

                # grade on the rule usage
                truth_1 = Truth(premise_1_LLM["f"], premise_1_LLM["c"])
                truth_2 = Truth(premise_2_LLM["f"], premise_2_LLM["c"])
                otb_truth = TFS.tf[each_result_LLM["r"]](truth_1, truth_2)
                tmp_as += (1 - similarity_score(each_result_LLM["f"], otb_truth.f)) * 25
                tmp_as += (1 - similarity_score(each_result_LLM["c"], otb_truth.c)) * 25
                tmp_bs += 50
                summarization_matrix[i][j].append(f"Ought-to-be truth-value: [{otb_truth.f}, {otb_truth.c}]. "
                                                  f"LLM truth-value: [{each_result_LLM['f']}, {each_result_LLM['c']}]")

                As_matrix[i, j] = tmp_as
                Bs_matrix[i, j] = tmp_bs

        cost_matrix = -As_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i in range(min(num_labels, num_LLMs)):  # Handle if sizes differ, but ideally they are equal
            matched_as = As_matrix[row_ind[i], col_ind[i]]
            matched_bs = Bs_matrix[row_ind[i], col_ind[i]]
            A += matched_as
            B += matched_bs
            for msg in summarization_matrix[row_ind[i]][col_ind[i]]:
                summarization.add(msg)

        return A / (B + 1e-5), summarization
    except Exception as e:
        print("An error occurred:", e)
        summarization.add("Fatal error, cannot parsing")
        return 0, summarization


def similarity_score(a, b):
    diff = abs(a - b)

    if diff <= 0.01:
        return 1.0

    elif diff <= 0.2:
        norm_diff = (diff - 0.01) / 0.19
        return 0.1 + 0.9 * (1 - norm_diff) ** 5

    else:
        return 0.1


if __name__ == "__main__":
    from_LLM = ('{'
                '"premise_1": {"s": "ID_59470", "o": "ID_425", "cp": "-->", "f": 0.802, "c": 0.9, "eb": [2927, 8201]}, '
                '"premise_2": {"s": "ID_20730", "o": "ID_59470", "cp": "<->", "f": 0.471, "c": 0.9, "eb": [4889]}, '
                '"results": ['
                '{"s": "ID_20730", "o": "ID_425", "cp": "-->", "f": 0.394, "c": 0.382, "eb": [2927, 4889, 8201], "r": "ana"}'
                '{"s": "ID_425", "o": "ID_20730", "cp": "-->", "f": 0.394, "c": 0.382, "eb": [2927, 4889, 8201], "r": "ana_p"}'
                ']}')
    label = ('{'
             '"premise_1": {"s": "ID_59470", "o": "ID_425", "cp": "-->", "f": 0.954, "c": 0.9, "eb": [2927, 8201]}, '
             '"premise_2": {"s": "ID_20730", "o": "ID_59470", "cp": "<->", "f": 0.508, "c": 0.9, "eb": [4889]}, '
             '"results": ['
             '{"s": "ID_20730", "o": "ID_425", "cp": "-->", "f": 0.484, "c": 0.411, "eb": [2927, 4889, 8201], "r": "ana"}]}')
    print(grade(from_LLM, label))
