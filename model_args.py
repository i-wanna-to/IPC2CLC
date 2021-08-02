import json
import os
import sys
import torch
from dataclasses import dataclass, field, asdict
from multiprocessing import cpu_count

def get_default_process_count():
    process_count = cpu_count() - 2 if cpu_count() > 2 else 1
    if sys.platform == "win32":
        process_count = min(process_count, 61)

    return process_count

@dataclass
class ModelArgs():
    seed: int = 42
    model_name: str = "bert-base-chinese"
    custom_model: bool = True
    batch_size: int = 64
    num_epochs: int = 20
    weight_decay: float = 0.01
    loss_type: str = "ContrastiveLoss"
    optimizer_class: str = "AdamW"
    kflod: int = 5
    device: str = "cuda" if torch.cuda.is_available() else 'cpu'
    n_gpu: int = 1
    train_dataset_path: str = "dataset/"
    test_dataset_path: str = "dataset/"
    warmup_steps: float = 0.15
    lr: float = 2e-05
    optimizer_eps: float = 1e-06
    scheduler: str = "WarmupLinear"
    evaluate_during_training: bool = True
    evaluation_steps: int = 10
    save_best_model: bool = True
    model_save_path: str = "output/"
    max_grad_norm: int = 1
    top_k: int = 3

    def update_from_dict(self, new_values):
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                setattr(self, key, value)
        else:
            raise (TypeError(f"{new_values} is not a Python dict."))

    def load(self, model_args_file):
        if model_args_file:
            #model_args_file = os.path.join(input_dir, "model_args.json")
            if os.path.isfile(model_args_file):
                with open(model_args_file, "r") as f:
                    model_args = json.load(f)
                self.update_from_dict(model_args)

@dataclass
class MappingArgs(ModelArgs):
    """
    Model args for a MappingModel
    """

    classification: bool = True