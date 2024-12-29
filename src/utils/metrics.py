"""Metrics for seqio tasks over graph data.

This module contains definitions of metric_fns to be used for scoring
graph tasks from nlgraph and graphqa.
"""
import torch
from typing import Mapping, Sequence


def yes_no_accuracy(targets: Sequence[str], predictions: Sequence[str]) -> Mapping[str, float]:
    # Adapted from https://github.com/google-research/talk-like-a-graph/blob/main/talk_like_a_graph/graph_metrics.py
    print(predictions)
    num_correct = 0
    num_ambiguous = 0
    num_indeterminate = 0
    num_correct_yes_preds = 0
    num_wrong_yes_preds = 0
    num_correct_no_preds = 0
    num_wrong_no_preds = 0
    num_yes_targets = 0
    num_no_targets = 0
    for target, prediction in zip(targets, predictions):
        normalized_target = target.lower()
        binarized_target = 'yes' in normalized_target
        if binarized_target and 'no' in normalized_target:
            raise ValueError(f'Ambiguous target string, {target}')
        if not binarized_target and 'no' not in normalized_target:
            raise ValueError(f'Indeterminate target string, {target}')
        if 'yes' in normalized_target:
            num_yes_targets += 1
        else:
            num_no_targets += 1
        normalized_prediction = prediction.lower()
        if not normalized_prediction:
            normalized_prediction = ''
        #else:
            #normalized_prediction = normalized_prediction[0]
        #normalized_prediction = normalized_prediction.lower()

        if 'yes' in normalized_prediction and 'no' in normalized_prediction:
            num_ambiguous += 1
            continue
        if 'yes' not in normalized_prediction and 'no' not in normalized_prediction:
            num_indeterminate += 1
            continue
        if 'yes' in normalized_prediction and binarized_target:
            num_correct += 1
            num_correct_yes_preds += 1
            continue
        if 'yes' in normalized_prediction and not binarized_target:
            num_wrong_yes_preds += 1
        if 'no' in normalized_prediction and not binarized_target:
            num_correct += 1
            num_correct_no_preds += 1
            continue
        if 'no' in normalized_prediction and binarized_target:
            num_wrong_no_preds += 1
    return {
        'num_correct': num_correct,
        'num_ambiguous': num_ambiguous,
        'num_indeterminate': num_indeterminate,
        'num_items': len(targets),
        'num_correct_yes_preds': num_correct_yes_preds,
        'num_correct_no_preds': num_correct_no_preds,
        'num_wrong_yes_preds': num_wrong_yes_preds,
        'num_wrong_no_preds': num_wrong_no_preds,
        'num_yes_targets': num_yes_targets,
        'num_no_targets': num_no_targets,
    }


def age_accuracy(targets: Sequence[str], predictions: Sequence[str]) -> Mapping[str, float]:
    total_correct = 0
    total_items = 0
    age_category_counts = {"Young": 0, "Adult": 0, "Old": 0}
    correct_predictions = {"Young": 0, "Adult": 0, "Old": 0}

    for label, prediction in zip(targets, predictions):
        total_items += 1
        age_category_counts[label] += 1
        if label == prediction:
            total_correct += 1
            correct_predictions[label] += 1

    return {
        "total_correct": total_correct,
        "total_items": total_items,
        "age_category_counts": age_category_counts,
        "correct_predictions": correct_predictions,
    }


def gender_accuracy(targets: Sequence[str], predictions: Sequence[str]) -> Mapping[str, float]:
    print(targets)
    print(predictions)
    total_correct = 0
    total_items = 0

    gender_category_counts = {"Male": 0, "Female": 0}
    correct_predictions = {"Male": 0, "Female": 0}

    for label, prediction in zip(targets, predictions):
        total_items += 1
        gender_category_counts[label] += 1
        if label == prediction:
            total_correct += 1
            correct_predictions[label] += 1

    return {
        "total_correct": total_correct,
        "total_items": total_items,
        "gender_category_counts": gender_category_counts,
        "correct_predictions": correct_predictions,
    }


def get_accuracy_gnnllm(batch_labels, logits, target_mask, tokenizer, task='yes_no'):
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions[target_mask == 1]
    labels = batch_labels[target_mask == 1]
    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions)
    target_tokens = tokenizer.convert_ids_to_tokens(labels)
    if task == 'yes_no':
        return yes_no_accuracy(target_tokens, predicted_tokens)
    elif task == 'age':
        return age_accuracy(target_tokens, predicted_tokens)
    elif task == 'gender':
        return gender_accuracy(target_tokens, predicted_tokens)
    else:
        raise ValueError(f"Task {task} not supported")


def get_accuracy_gnn_binary_task(probs, labels):
    """
    Computes the accuracy for a binary classification task.
    """

    preds = (probs > 0.5).long()

    num_items = labels.size(0)

    num_yes_targets = (labels == 1).sum().item()
    num_no_targets = (labels == 0).sum().item()

    num_correct = (preds == labels).sum().item()

    num_correct_yes_preds = ((preds == 1) & (labels == 1)).sum().item()
    num_correct_no_preds = ((preds == 0) & (labels == 0)).sum().item()
    num_wrong_yes_preds = ((preds == 1) & (labels == 0)).sum().item()
    num_wrong_no_preds = ((preds == 0) & (labels == 1)).sum().item()

    return {
        'acc': num_correct/num_items,
        'num_correct_yes_preds': num_correct_yes_preds,
        'num_correct_no_preds': num_correct_no_preds,
        'num_yes_targets': num_yes_targets,
        'num_no_targets': num_no_targets,
    }


def get_rmse_gnn_regression_task(preds, labels):
    """
    Computes the RMSE for a regression task.
    """
    preds = preds.float()
    labels = labels.float()

    squared_errors = (preds - labels) ** 2

    rmse = torch.sqrt(torch.mean(squared_errors)).item()

    mean_pred = preds.mean().item()
    mean_label = labels.mean().item()
    num_samples = labels.size(0)

    return {
        'rmse': rmse,
        'mean_pred': mean_pred,
        'mean_label': mean_label,
        'num_samples': num_samples,
    }

