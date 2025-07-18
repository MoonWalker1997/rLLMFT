import csv
import json

from formal_reasoning import gen_random_reasoning

num_data = 10
raw_data = gen_random_reasoning(num_data)
raw_data = [each[0] for each in raw_data]


def prompt(task_1_str, task_2_str, results: dict):
    return [
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
        "Derive all valid conclusions. ",
        f"User: Suppose we have the first premise: {task_1_str} The second premise: {task_2_str} Assistant: ",
        json.dumps(results)]


meta_data = []
for each in raw_data:
    meta_data.append(prompt(*each))

with open("data/data_meta.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar="\\")
    writer.writerow(["Introduction", "Premise", "Answer"])
    for i in range(len(meta_data)):
        writer.writerow(meta_data[i])

chunking_data = []
max_length = 0
for each in meta_data:
    tmp = each[0] + each[1]
    if len(tmp) > max_length:
        max_length = len(tmp)
    chunking_data.append(tmp)
    tmp = each[1] + each[2]
    if len(tmp) > max_length:
        max_length = len(tmp)
    chunking_data.append(tmp)

with open("data/data_chunk.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar="\\")
    writer.writerow(["chunk"])
    for i in range(len(chunking_data)):
        writer.writerow([chunking_data[i]])

print(max_length)
