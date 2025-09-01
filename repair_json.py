import argparse
import csv
import json
import os

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from grading import grade_0

parser = argparse.ArgumentParser()
parser.add_argument("--API-key", type=str)
parser.add_argument("--show-grade", type=bool, default=True)

if __name__ == "__main__":

    args = parser.parse_args()

    client = OpenAI(
        api_key=args.API_key,
        base_url="https://api.deepseek.com"
    )

    for col, each_record in enumerate(os.listdir("./test_record/")):
        if "csv" not in each_record:
            continue
        tmp = []
        with open("./test_record/" + each_record) as f:
            reader = csv.reader(f, quoting=csv.QUOTE_MINIMAL)
            for row, each_line in tqdm(enumerate(reader), desc="Processing Each Test Record"):
                if row == 0:  # skip the title
                    tmp.append(each_line)
                    continue
                try:
                    prefix, pending_json_r = each_line[1].split("\nAssistant")
                    prompt = f"""The following string is a broken json, you need to repair it. 
                    It uses "step 1" and "step 2" as two keys, in which the corresponding values are also json strings.
                    To repair: {pending_json_r}
                    Please output only the repaired json string. Do not explain anything.
                    """
                    count = 0
                    while True:
                        response = client.chat.completions.create(
                            model="deepseek-chat",
                            messages=[
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.,
                            max_tokens=700,
                            stream=False
                        )
                        repaired_jsons = response.choices[0].message.content.strip()
                        try:
                            json.loads(repaired_jsons)
                            if args.show_grade:
                                print(grade_0(json.loads(repaired_jsons)))
                            tmp.append([each_line[0], prefix + "\nAssistant: " + repaired_jsons])
                            break
                        except:
                            count += 1
                            if count > 5:
                                tmp.append(each_line)
                                break
                except:
                    tmp.append(each_line)

        df = pd.DataFrame(tmp[1:], columns=tmp[0])
        df.to_csv(f"./test_record/repaired_{each_record}.csv", index=False)
