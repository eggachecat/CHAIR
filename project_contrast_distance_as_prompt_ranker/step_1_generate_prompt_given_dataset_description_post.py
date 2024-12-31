import pandas as pd
import os
wsFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
output_file = f"{wsFolder}/saves/project_contrast_distance_as_prompt_ranker/baseline/step_1.txt"

rows = []

PROMPT_ID_PREFIX = "**Prompt "
PROMPT_ID_SUFFIX = "**\n"
current_row = None
STATE = "WAITING_PROMPT"
with open(output_file, "r") as fp:
    for line in fp.readlines():
        if line.startswith(PROMPT_ID_PREFIX) and line.endswith(PROMPT_ID_SUFFIX):
            if current_row is not None:
                current_row['content'] = current_row['content'].strip()
                rows.append(current_row)
            current_row = {
                "prompt_id":line[len(PROMPT_ID_PREFIX):-len(PROMPT_ID_SUFFIX)],
                "content": ""
            } 
            continue
        current_row["content"] += line
if current_row is not None:
    current_row['content'] = current_row['content'].strip()
    rows.append(current_row)

df = pd.DataFrame(rows)
df.to_csv(output_file.replace(".txt", ".csv"), index=False)