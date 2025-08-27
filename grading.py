import csv
import os

import json

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from formal_reasoning import Task, Truth, reasoning


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


def fc_similarity_score(fc1, fc2):
    diff = abs(fc1 - fc2)

    # if the diff is smaller than 0.05, assumed the same
    if diff <= 0.05:
        return 1.0

    # project the value (from 0.05 to 0.2) to a range [0, 1]
    elif diff <= 0.2:
        norm_diff = (diff - 0.05) / 0.15
        return 0.1 + 0.9 * (1 - norm_diff) ** 2  # apply polynomial effects

    else:
        return 0.1


def grade_0(json_outputs):
    # grade 2 steps in reasoning
    # grade each individual step
    # then grade the connection between 2 steps

    ret = grade_0_util(json_outputs["step 1"]) * grade_0_util(json_outputs["step 2"])

    step_1_result = json_outputs["step 1"]["results"][0]
    A, B = np.zeros((1, 2)), np.zeros((1, 2))
    for i, each_step_2_premise in enumerate([json_outputs["step 2"]["premise_1"], json_outputs["step 2"]["premise_2"]]):
        if "r" not in each_step_2_premise:
            continue
        tmp_A, tmp_B = 0, 0
        for each_key in ["s", "o", "cp", "eb", "r"]:
            if step_1_result[each_key] == each_step_2_premise[each_key]:
                tmp_A += 5
            tmp_B += 5

        # grade on the truth-value
        tmp_A += fc_similarity_score(step_1_result["f"], each_step_2_premise["f"]) * 25
        tmp_A += fc_similarity_score(step_1_result["c"], each_step_2_premise["c"]) * 25
        tmp_B += 50
        A[0, i] = tmp_A
        B[0, i] = tmp_B

    a, b = 0, 0
    scores = A / (B + 1e-5)
    row_idx, col_idx = linear_sum_assignment(-scores)
    for row_i, col_i in zip(row_idx, col_idx):
        a += A[row_i, col_i]
        b += B[row_i, col_i]

    ret *= float(max(0.1, a / (b + 1e-5)))

    return ret


def grade_0_util(json_output):
    # grade individual single step reasoning

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
        for j, each_label_task in enumerate(Rs):
            tmp_A, tmp_B = 0, 0
            for each_key in ["s", "o", "cp", "eb", "r"]:
                if each_llm_output is not None and each_key in each_llm_output and each_llm_output[each_key] == \
                        each_label_task.to_json()[each_key]:
                    tmp_A += 5
                tmp_B += 5

            # grade on the truth-value
            if "f" in each_llm_output and "c" in each_llm_output:
                tmp_A += fc_similarity_score(each_llm_output["f"], each_label_task.to_json()["f"]) * 25
                tmp_A += fc_similarity_score(each_llm_output["c"], each_label_task.to_json()["c"]) * 25
            tmp_B += 50
            A[i, j] = tmp_A
            B[i, j] = tmp_B

    a, b = 0, 0
    scores = A / B
    row_idx, col_idx = linear_sum_assignment(-scores)
    for row_i, col_i in zip(row_idx, col_idx):
        a += A[row_i, col_i]
        b += B[row_i, col_i]

    return float(max(0.1, a / (b + 1e-5)))


def grade_1(llm_json_output_step2, label_json_output_step2):
    # grade whether the llm response solves the asked problem

    llm_json_output_step2 = llm_json_output_step2["step 2"]
    label_json_output_step2 = label_json_output_step2["step 2"]

    _, _, results_llm = parsing_json(llm_json_output_step2)

    if not results_llm:
        return 0.1

    _, _, results_label = parsing_json(label_json_output_step2)

    A, B = np.zeros((len(results_llm), len(results_label))), np.zeros((len(results_llm), len(results_label)))
    for i, each_llm_output in enumerate(results_llm):
        for j, each_label in enumerate(results_label):
            tmp_A, tmp_B = 0, 0
            for each_key in ["s", "o", "cp", "eb", "r"]:
                if (each_llm_output is not None and each_key in each_llm_output and each_llm_output[each_key]
                        == each_label[each_key]):
                    tmp_A += 5
                tmp_B += 5

            # grade on the truth-value
            if "f" in each_llm_output and "c" in each_llm_output:
                tmp_A += fc_similarity_score(each_llm_output["f"], each_label["f"]) * 25
                tmp_A += fc_similarity_score(each_llm_output["c"], each_label["c"]) * 25
            tmp_B += 50
            A[i, j] = tmp_A
            B[i, j] = tmp_B

    a, b = 0, 0
    scores = A / B
    row_idx, col_idx = linear_sum_assignment(-scores)
    for row_i, col_i in zip(row_idx, col_idx):
        a += A[row_i, col_i]
        b += B[row_i, col_i]

    return float(max(0.1, a / (b + 1e-5)))


if __name__ == "__main__":

    record = []
    for col, each_record in enumerate(os.listdir("./test_record/")):
        with open("./test_record/" + each_record) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
            for row, each_line in enumerate(reader):
                if row == 0:  # skip the title
                    continue
                if col == 0:  # create an additional colum
                    record.append([json.loads(each_line[0].split("Assistant: ")[-1].split("<|endoftext|>")[0].strip())])
                each_line[1] = each_line[1].replace("Human:", "")
                try:
                    record[row - 1].append(json.loads(each_line[1].split("Assistant: ")[-1].strip()))
                except:
                    record[row - 1].append("err")

    for i in range(len(record)):
        rebuilt_responses = []
        tmp_1 = []
        tmp_2 = []
        for j in range(len(record[i])):
            if j == 0:
                continue
            if record[i][j] == "err":
                tmp_1.append("err")
                tmp_2.append("err")
            else:
                tmp_1.append(record[i][j]["step 1"])
                tmp_2.append(record[i][j]["step 2"])
        for each_1 in tmp_1:
            for each_2 in tmp_2:
                if each_1 == "err" or each_2 == "err":
                    rebuilt_responses.append("err")
                else:
                    rebuilt_responses.append({"step 1": each_1, "step 2": each_2})
        record[i].extend(rebuilt_responses)

    scores = []
    for i in range(len(record)):
        tmp = []
        for j in range(len(record[i])):

            if j == 0:
                continue
            try:
                tmp.append(grade_0(record[i][j]) * grade_1(record[i][j], record[i][0]))
            except:
                tmp.append(-1)
        scores.append(tmp)

    scores = np.array(scores)

    c_0, c_1, c_2, c_h3, c_h = [], [], [], [], []

    for threshold in np.linspace(0.1, 0.9, 10):
        c_0.append(sum(scores[:, 0] > threshold) / scores.shape[0])
        c_1.append(sum(scores[:, 1] > threshold) / scores.shape[0])
        c_2.append(sum(scores[:, 2] > threshold) / scores.shape[0])
        c_h3.append(sum(np.max(scores[:, :3], axis=1) > threshold) / scores.shape[0])
        c_h.append(sum(np.max(scores, axis=1) > threshold) / scores.shape[0])

    plt.figure(figsize=(10, 5))
    plt.plot(c_0, label="c0")
    plt.plot(c_1, label="c1")
    plt.plot(c_2, label="c2")
    plt.plot(c_h3, label="ch3")
    plt.plot(c_h, label="ch")

    plt.grid()
    plt.legend()
    plt.show()

    plt.savefig("curve.png")
