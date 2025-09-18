from typing import Optional, Dict, Sequence, Union, List
from dataclasses import dataclass, field
import transformers
import typing

@dataclass
class LoraArguments:
    transform_layers: str = field(metadata={"help": "Layers for Representation. Layers are seperate by `,` eg: `10,12,14,16,18,20` "})
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Union[List[str], str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    adapter_name_or_path: str = field (
        default=None, metadata={"help": "Adapater name"}
    )
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")

    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    grouped_to_max_length: bool = field (
        default=False, metadata={"help": "Group to chunks of max length for pretraining"}
    )
    
    num_train_epochs: int = field(default=1)

    use_refusal_retain : Optional[bool] = field(default = True, 
                                                metadata={"help":"whether to train on refusal retain set for Llama models"})
    sc_train_subset : Optional[List[str]] = field(default=None,
                                                  metadata={"help":"subset of the sc train set to train on"})
    log_every : Optional[int] = field(default = 10,
                                      metadata = {"help" : "log loss every log_every steps"})
    sc_train_seq_type : Optional[str] = field(default = 'all_text',
                                              metadata = {"help" : "what portion of the sequence to train on. can be all_text or assistant_response"})
    coeff_schedule : Optional[str] = field(default = 'linear_converge',
                                           metadata = {'help' : 'schedule for the coefficients. can be linear_converge or constant'})
    sc_loss_type : Optional[str] = field(default = 'orig_act_dotprod',
                                         metadata = {'help' : 'type of loss function for shortcircuiting. can be orig_act_dotprod, rand_vec_norm, pos_constant_rmu_{coeff}, center_constant_rmu_{coeff}'})
    
    save_steps:   int = field(default=100)
    save_total_limit: int = field(default=1)
    bf16: bool = field(default=True)
    per_device_train_batch_size: int = field(default=8, metadata={"help": "Batch size per device during training."})
    
    ablation_prob: float = field(default=0.05)
    hybrid_response: bool = field(default=True)
    only_malicious: bool = field(default=False)
    only_response: bool = field(default=False)
    direction_path: str = "<Your Refusal Direction Path>.pt"
    # eval_steps: int = field(default=100) 