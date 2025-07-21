import json

import json_repair

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
                premise_2 = json_output["premise_1"]
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
        for each_result_label in results_label:
            As, Bs = 0, 0
            for each_result_LLM in results_LLM:
                tmp_As, tmp_Bs = 0, 0
                # grade on the consistency
                for each_key in ["s", "o", "cp", "eb", "r"]:
                    if each_result_LLM is not None:
                        if each_key in each_result_LLM:
                            if each_result_label[each_key] == each_result_LLM[each_key]:
                                tmp_As += 5
                            else:
                                summarization.add(f"LLM result's key {each_key} inconsistent. "
                                                  f"LLM result: {each_result_LLM}, "
                                                  f"LLM {each_key}: {each_result_LLM[each_key]}, "
                                                  f"label {each_key}: {each_result_label[each_key]}")
                        else:
                            summarization.add(f"LLM result does not have key {each_key}. "
                                              f"LLM result: {each_result_LLM}")
                    else:
                        summarization.add(f"LLM does not have any results")
                    tmp_Bs += 5

                # grade on the rule usage
                truth_1 = Truth(premise_1_LLM["f"], premise_1_LLM["c"])
                truth_2 = Truth(premise_2_LLM["f"], premise_2_LLM["c"])
                otb_truth = TFS.tf[each_result_LLM["r"]](truth_1, truth_2)
                tmp_As += (1 - min(1, abs(each_result_LLM["f"] - otb_truth.f))) * 5
                tmp_As += (1 - min(1, abs(each_result_LLM["c"] - otb_truth.c))) * 5
                tmp_Bs += 10
                summarization.add(f"Ought-to-be truth-value: [{otb_truth.f}, {otb_truth.c}]. "
                                  f"LLM truth-value: [{each_result_LLM['f']}, {each_result_LLM['c']}]. ")

                if tmp_As > As:
                    As = tmp_As
                if tmp_Bs > Bs:
                    Bs = tmp_Bs
            A += As
            B += Bs

        return A / (B + 1e-5), summarization
    except:
        summarization.add("Fatal error, cannot parsing")
        return 0, summarization


if __name__ == "__main__":
    # with (open("data/data_meta.csv", newline="", encoding="utf-8") as file):
    #     reader = csv.reader(file, quoting=csv.QUOTE_NONE, escapechar='\\')
    #     next(reader, None)
    #     for each in reader:
    #         print(each)
    #         break

    from_LLM = ('{"premise_1": {"s": "ID_55142", "o": "ID_68017", "cp": "-->", "f": 0.474, "c": 0.9, '
                '"eb": [970, 2032]}, "premise_2": {"s": "ID_68017", "o": "ID_77508", "cp": "-->", "f": 0.193, '
                '"c": 0.9, "eb": [9512]}, "results": [{"s": "ID_77508", "o": "ID_55142", "cp": "-->", '
                '"f": 1.0, "c": 0.069, "eb": [970, 2032, 9512], "r": "exe"}, {"s": "ID_55142", '
                '"o": "ID_77508", "cp": "-->", "f": 0.092, "c": 0.074, "eb": [970, 2032, 9512], '
                '"r": "ded_p"}]}')
    label = ('{"premise_1": {"s": "ID_55142", "o": "ID_68017", "cp": "-->", "f": 0.474, "c": 0.9, "eb": [970, '
             '2032]}, "premise_2": {"s": "ID_68017", "o": "ID_77508", "cp": "-->", "f": 0.193, "c": 0.9, '
             '"eb": [9512]}, "results": [{"s": "ID_77508", "o": "ID_55142", "cp": "-->", "f": 1.0, '
             '"c": 0.069, "eb": [970, 2032, 9512], "r": "exe"}, {"s": "ID_55142", "o": "ID_77508", '
             '"cp": "-->", "f": 0.092, "c": 0.074, "eb": [970, 2032, 9512], "r": "ded_p"}]}')
    print(grade(from_LLM, label))
