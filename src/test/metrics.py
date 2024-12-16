# Adapted from https://github.com/google-research/talk-like-a-graph/blob/main/talk_like_a_graph/graph_metrics.py
"""Metrics for seqio tasks over graph data.

This module contains definitions of metric_fns to be used for scoring
graph tasks from nlgraph and graphqa.
"""

from typing import Mapping, Sequence


def yes_no_accuracy(targets: Sequence[str], predictions: Sequence[str]) -> Mapping[str, float]:
    """Assesses the accuracy of LLM outputs on Yes/No tasks.

    Targets must contain either the word 'yes' or the word 'no' but not both.

    Predictions are binarized by checking for 'yes' or 'no' in the first line.

    Args:
      targets: The expected output strings.
      predictions: The LLM outputs.

    Returns:
       Returns a dict of the following metrics:
        yes_no_accuracy: The % where the target and prediction match.
        yes_no_ambiguous: The % where the prediction contained yes and no
        yes_no_indeterminate: The % where the prediction contained neither yes nor
        no

    Raises:
      ValueError: If a target string contains 'yes' and 'no'
    """
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
        normalized_prediction = prediction.splitlines()
        if not normalized_prediction:
            normalized_prediction = ''
        else:
            normalized_prediction = normalized_prediction[0]
        normalized_prediction = normalized_prediction.lower()

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
