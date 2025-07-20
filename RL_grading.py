import csv
import json
import json_repair


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
                                    or "eb" not in each):
                                return None, None, []
                            results.append(each)
                        return premise_1, premise_2, results
            else:
                return None, None, []
    else:
        return None, None, []


def grade(jsons_LLM, jsons_Label):
    
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
    
        for premises in [[premise_1_label, premise_1_LLM], [premise_2_label, premise_2_LLM]]:
            for each_key in ["s", "o", "cp", "eb"]:
                if premises[1] is not None:
                    if each_key in premises[1] and premises[0][each_key] == premises[1][each_key]:
                        A += 5
                        B += 5
                else:
                    B += 5
            for each_key in ["f", "c"]:
                if premises[1] is not None:
                    if each_key in premises[1]:
                        if abs(premises[0][each_key] - premises[1][each_key]) <= 0.2:
                            A += 5
                B += 5
    
        for each_result_label in results_label:
            As, Bs = 0, 0
            for each_result_LLM in results_LLM:
                tmp_As, tmp_Bs = 0, 0
                for each_key in ["s", "o", "cp", "eb"]:
                    if each_result_LLM is not None:
                        if each_key in each_result_LLM and each_result_label[each_key] == each_result_LLM[each_key]:
                            tmp_As += 5
                            tmp_Bs += 5
                    else:
                        tmp_Bs += 5
                for each_key in ["f", "c"]:
                    if each_result_LLM is not None:
                        if each_key in each_result_LLM:
                            tmp_As += (1 - min(1, abs(each_result_label[each_key] - each_result_LLM[each_key]))) * 5
                    tmp_Bs += 5
                if tmp_As > As:
                    As = tmp_As
                if tmp_Bs > Bs:
                    Bs = tmp_Bs
            A += As
            B += Bs
    
        return A / (B + 1e-5)
    except:
        return 0


if __name__ == "__main__":
    with (open("data/data_meta.csv", newline="", encoding="utf-8") as file):
        reader = csv.reader(file, quoting=csv.QUOTE_NONE, escapechar='\\')
        next(reader, None)
        for each in reader:
            from_LLM = "1. Identify the different types of information (statements, judgments, facts) provided by the user."
            # label = each[2]
            label = '{"premise_1": {"s": "ID_93045", "o": "ID_6664", "cp": "-->", "f": 0.045, "c": 0.9, "eb": [1270, 6980]}, "premise_2": {"s": "ID_56961", "o": "ID_6664", "cp": "<->", "f": 0.921, "c": 0.9, "eb": [1029, 2539]}, "results": [{"s": "ID_93045", "o": "ID_56961", "cp": "-->", "f": 0.042, "c": 0.746, "eb": [1029, 1270, 2539, 6980]}]}'
            print(grade(from_LLM, label))
            break
