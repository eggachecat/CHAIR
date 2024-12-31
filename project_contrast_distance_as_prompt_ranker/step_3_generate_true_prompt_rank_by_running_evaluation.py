import sys
import os
from llamafactory.eval.evaluator import Evaluator
from dataclasses import dataclass, field
from llamafactory.hparams.parser import EvaluationArguments,  HfArgumentParser, _EVAL_ARGS
from llamafactory.eval.template import get_eval_template, _register_eval_template
from llamafactory.extras.misc import torch_gc

def main():
    parser = HfArgumentParser(_EVAL_ARGS)
    parser.add_argument('--cdapr_output_folder', type=str)
    parser.add_argument('--cdapr_exp_id', type=str)
    args = parser.parse_args()
    task_id = args.task

    base_argv = sys.argv
    for prompt_id, prompt in [
        # ["wtf", "wtf"],
        ["shit", "shit"],
        ["you_are_good", "You are a good answering machine."],
        ["COT", "You are a highly knowledgeable assistant. Please think step by step"],
        ["gpt-4", "You are a highly knowledgeable assistant, proficient in answering questions across a wide range of subjects, including mathematics, history, science, and humanities, at high school to university levels. You are given a multiple-choice question with several options. Your task is to provide the correct answer, explain your reasoning, and ensure your explanation is clear and concise."]
    ]:
        torch_gc()
        prompt_key = f"prompt_key_{prompt_id}"
        save_dir = os.path.join(args.cdapr_output_folder, args.cdapr_exp_id, task_id, prompt_key)
        _register_eval_template(
            name=prompt_key,
            system=prompt + "\n" + "The following are multiple choice questions (with answers) about {subject}.\n\n",
            choice="\n{choice}. {content}",
            answer="\nAnswer:",
        )
        sys.argv = base_argv
        sys.argv.extend(['--lang', prompt_key])
        sys.argv.extend(['--save_dir', save_dir])
        print("sys.argv>>>>>>>>>>>>>>", sys.argv)
        Evaluator().eval()
        torch_gc()


if __name__ == '__main__':
    sys.exit(main())