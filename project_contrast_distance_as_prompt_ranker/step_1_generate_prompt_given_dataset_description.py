import transformers
import torch
import os
torch.manual_seed(42)

wsFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# 切换为你下载的模型文件目录, 这里的demo是Llama-3-8B-Instruct
# 如果是其他模型，比如qwen，chatglm，请使用其对应的官方demo
output_file = f"{wsFolder}/saves/project_contrast_distance_as_prompt_ranker/baseline/step_1.txt"
model_id = f"{wsFolder}/downloaded_models/Meta-Llama-3-8B-Instruct"
dataset_description = """
Dataset Card for truthful_qa
Dataset Summary
TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. 
The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. 
Questions are crafted so that some humans would answer falsely due to a false belief or misconception. 
To perform well, models must avoid generating false answers learned from imitating human texts.
""".strip()

SYSTEM_PROMPT = """
You are an expert at creating prompts for a large language model. 
Please write a highly effective prompt that can guide another model in generating useful and relevant content. 
The prompt should have the following characteristics:

Clear Objective: The task or goal should be explicitly stated, such as writing a creative story, drafting a technical document, or answering complex questions.
Context Guidance: Provide the necessary context to ensure that the model’s response is accurate and relevant to the task.
Tone and Style Control: Include instructions to control the tone or style of the output (e.g., formal, humorous, concise).
Constraints: Specify any particular requirements, such as length limits, the use of certain keywords, or avoiding specific types of content.
Please create a general-purpose prompt that fits a wide range of use cases.
"""

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Below is the description for this task\n```\n{dataset_description}\n```\nPlease write 10 prompts to solve this task."},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


max_new_tokens = 10 * 256
outputs = pipeline(
    prompt,
    max_new_tokens=max_new_tokens,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
with open(output_file, "w") as fp:
    fp.write(outputs[0]["generated_text"][len(prompt):])

