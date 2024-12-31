import torch
import pyvene as pv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import os
wsFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def kl_divergence(p, q):
    p = p + 1e-8
    q = q + 1e-8
    return torch.sum(p * torch.log(p / q), dim=-1)


def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


# the HF model you want to intervene on
model_name = f"{wsFolder}/downloaded_models/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda")

for name, module in model.named_modules():
    print(name)


def make_data_collector_factory():

    data_collection = {}

    class DataCollector:
        is_source_constant = True
        keep_last_dim = True
        def __init__(self, component, **kwargs):
            super().__init__(**kwargs)
            self.component = component

        def __call__(self, base, source=None, subspaces=None, **kwargs):
            print(self.component)
            print("base.shape", base.shape)
            print("source", source)
            print("subspaces", subspaces)
            if self.component not in data_collection:
                data_collection[self.component] = []
            return base
        
    def _f(component):
        return DataCollector(component)
    
    return _f, data_collection

factory, data_collection = make_data_collector_factory()

# def wrapper(intervener):
#     def wrapped(*args, **kwargs):
#         return intervener(*args, **kwargs)
#     return wrapped

# class Collector():
#     collect_state = True
#     collect_action = False  
#     is_source_constant = True
#     keep_last_dim = True
#     def __init__(self, multiplier, head):
#         self.head = head
#         self.states = []
#         self.actions = []
#     def reset(self):
#         self.states = []
#         self.actions = []
#     def __call__(self, b, s, *args, **kwargs): 
#         print("b", b.shape)
#         if self.head == -1:
#             self.states.append(b[0, -1].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
#         else:
#             self.states.append(b[0, -1].reshape(32, -1)[self.head].detach().clone())  # original b is (batch_size, seq_len, #key_value_heads x D_head)
#         return b

# 2. Wrap the model with an intervention config
num_layers = model.config.num_hidden_layers
attention_layers = [
    f"model.layers[{i}].self_attn.o_proj.input" for i in range(num_layers)]
print(attention_layers)
pv_model = pv.IntervenableModel([
    {
        "component": component,
        "intervention": factory(component)
        # "intervention": wrapper(Collector(component, head=-1))
        # "intervention": Collector(component, head=-1)
    } for component in attention_layers
], model=model)


# # 3. Run the intervened model
encoded = tokenizer("The capital of Spain is XXX ? No ? Yes!", return_tensors="pt")
print(encoded['input_ids'].shape)
orig_outputs, intervened_outputs = pv_model(
    {"input_ids": encoded.input_ids.to("cuda"), "output_hidden_states": True},
    output_original_output=True
)
# print(orig_outputs.logits.shape)
# print([v.shape for v in intervened_outputs])
print("hello world")
# print(intervened_outputs)
# # 4. Compare outputs
# print(intervened_outputs.logits - orig_outputs.logits)
