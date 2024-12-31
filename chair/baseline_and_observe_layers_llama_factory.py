# Copyright 2024 the LlamaFactory team.
#
# This code is inspired by the Dan's test library.
# https://github.com/hendrycks/test/blob/master/evaluate_flan.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file


from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import CHOICES, SUBJECTS
from llamafactory.hparams import get_eval_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.eval.template import get_eval_template
from chair_common import DataCollectorFactory, get_intervention_config
import pyvene as pv

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Evaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        assert self.eval_args.save_dir is not None
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.loaded_model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]
        self._debug_max_question = self.eval_args.n_shot + 500000000000
        self.data_collection_factory = DataCollectorFactory()

        for name, module in self.loaded_model.named_modules():
            print(name)

        intervention_config, intervented_keys = get_intervention_config(
            self.data_collection_factory, self.loaded_model.config.name_or_path
        )
        self.intervented_keys = intervented_keys

        # intervention_config = [{
        #     "component":  f"model.layers.0.input",
        #     "intervention": self.data_collection_factory.create(0)
        # }] + [
        #     {
        #         "component":  f"model.layers.{layer-1}.output",
        #         "intervention": self.data_collection_factory.create(layer)
        #     } for layer in list(range(1,33))
        # ]
        self.model = pv.IntervenableModel(intervention_config, model=self.loaded_model)
        self.device = self.loaded_model.device
        self.oberservation = {
            'subject': [],
            'content': [],
            'truth': [],
            'layer_history': []
        }


    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, "torch.Tensor"]) -> List[str]:
        baseline_outputs, _ = self.model({ 
            **batch_input,  
            "output_hidden_states": True, 
            "return_dict":  True,
            "output_attentions": True
        }, output_original_output=True)
        baseline_logits = baseline_outputs.logits
        baseline_lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        baseline_word_probs = torch.stack([baseline_logits[i, baseline_lengths[i] - 1] for i in range(len(baseline_lengths))], dim=0)

        baseline_choice_probs = torch.nn.functional.softmax(baseline_word_probs[:, self.choice_inputs], dim=-1).detach()
        baseline_result = [chr(ord("A") + offset.item()) for offset in torch.argmax(baseline_choice_probs, dim=-1)]

        dict_outputs = self.data_collection_factory.data_collection
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        def _extract_history_for_choice_at_last_token(_choice, _batch_index, _ans_index):
            _choice_history = []
            for _layer in self.intervented_keys:
                tokens_prob = dict_outputs[_layer][_batch_index][_ans_index].squeeze(0)
                tokens_prob = tokens_prob.log_softmax(-1)  # logits to log probs
                _choice_history.append(tokens_prob[_choice].cpu().float().item())
            return _choice_history

        log_baseline_word_probs = baseline_word_probs.log_softmax(-1)
        choice_history_list_batch = [
            [
                _extract_history_for_choice_at_last_token(c, i, lengths[i] - 1) + [
                    log_baseline_word_probs[i][c].cpu().float().item()
                ]
                for c in self.choice_inputs
            ]
            for i in range(len(lengths)) 
        ]

        self.oberservation['layer_history'].extend(choice_history_list_batch)

        return baseline_result

    def eval(self) -> None:
        eval_task = self.eval_args.task.split("_")[0]
        eval_split = self.eval_args.task.split("_")[1]

        print("eval_task", eval_task)
        print("eval_split", eval_split)
        print("self.model_args.cache_dir", self.model_args.cache_dir)
        print("self.model_args.hf_hub_token", self.model_args.hf_hub_token)

        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, eval_task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )
        print("mapping", mapping)
        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        # pbar = tqdm([
        #     x
        #     for x in categorys.keys()
        #     if x in [
        #         "high_school_european_history"
        #     ]
        #     # "business_ethics",
        # ], desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, eval_task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                trust_remote_code=True,
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(min(len(dataset[eval_split]), self._debug_max_question), desc="Formatting batches", position=1, leave=False):
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset[eval_split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                )

                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])
                self.oberservation['subject'].append(subject)
                self.oberservation['content'].append(messages[-2]["content"])

            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.device)
                self.oberservation['truth'].extend(labels[i : i + self.eval_args.batch_size])
                preds = self.batch_inference(batch_input)
                outputs += preds
                torch.cuda.empty_cache()

            corrects = np.array(outputs) == np.array(labels)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}
        
       
        output_folder = self.eval_args.save_dir + "_mmlu_layer_observation"
        os.makedirs(output_folder, exist_ok=True)

        with open(f"{output_folder}/layer-observation-{self.eval_args.n_shot}.json", "w") as fp:
            json.dump(self.oberservation, fp)
        pbar.close()
        self._save_results(category_corrects, results)

    def _save_results(self, category_corrects: Dict[str, "NDArray"], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join(
            [
                "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)


def run_eval() -> None:
    Evaluator().eval()

def main():
    run_eval()


if __name__ == "__main__":
    main()