"""
Checkpointing for model parameters.
"""

import os
import torch


def save_model(model, args):
    output_dir = args.output_dir
    name = args.name

    os.makedirs(output_dir, exist_ok=True)
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]

    torch.save(state_dict, output_dir + "/" + name + ".pt")


def load_model(model, args):
    output_dir = args.output_dir
    name = args.name

    if args.device != "cuda":
        model.load_state_dict(torch.load(output_dir + "/" + name + ".pt", map_location=torch.device('cpu')), strict=False)
    else:
        model.load_state_dict(torch.load(output_dir + "/" + name + ".pt"), strict=False)
    return model
