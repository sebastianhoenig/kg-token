"""
Logging to Weights and Biases
"""

import wandb


def log_train_to_wandb(res, epoch, loss, optimizer, example_ct=None):
    if example_ct is None:
        step = epoch
    else:
        step = example_ct

    correct_yes_preds = res['num_correct_yes_preds']
    correct_no_preds = res['num_correct_no_preds']
    wrong_yes_preds = res['num_wrong_yes_preds']
    wrong_no_preds = res['num_wrong_no_preds']
    yes_targets = res['num_yes_targets']
    no_targets = res['num_no_targets']
    num_items = res['num_items']
    num_correct = res['num_correct']
    accuracy = num_correct / num_items
    yes_preds = correct_yes_preds+wrong_yes_preds
    no_preds = correct_no_preds+wrong_no_preds

    wandb.log({"epoch": epoch, "loss": loss}, step=step)
    wandb.log({"epoch": epoch, "accuracy": accuracy}, step=step)
    wandb.log({"epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']}, step=step)
    wandb.log({"epoch": epoch, "yes_preds": yes_preds, "wrong_yes_preds": wrong_yes_preds}, step=step)
    wandb.log({"epoch": epoch, "no_preds": no_preds, "wrong_no_preds": wrong_no_preds}, step=step)
    wandb.log({"epoch": epoch, "yes_targets": yes_targets, "no_targets": no_targets}, step=step)


def log_test_to_wandb(res):
    correct_yes_preds = res['num_correct_yes_preds']
    correct_no_preds = res['num_correct_no_preds']
    wrong_yes_preds = res['num_wrong_yes_preds']
    wrong_no_preds = res['num_wrong_no_preds']
    yes_targets = res['num_yes_targets']
    no_targets = res['num_no_targets']
    num_items = res['num_items']
    num_correct = res['num_correct']

    accuracy = num_correct/num_items if num_items > 0 else 0
    yes_preds = correct_yes_preds + wrong_yes_preds
    no_preds = correct_no_preds + wrong_no_preds

    data = [
        ["accuracy", accuracy],
        ["yes_preds", yes_preds],
        ["wrong_yes_preds", wrong_yes_preds],
        ["yes_targets", yes_targets],
        ["no_preds", no_preds],
        ["wrong_no_preds", wrong_no_preds],
        ["no_targets", no_targets],
        ["num_items", num_items],
        ["num_correct", num_correct],
    ]

    table = wandb.Table(columns=["Metric", "Value"], data=data)

    wandb.log({"evaluation_table": table})


def log_age_downstream_task_to_wandb(res):
    total_correct = res['total_correct']
    total_items = res['total_items']
    young_category = res['young_category']
    adult_category = res['adult_category']
    old_category = res['old_category']
    correct_young = res['correct_young']
    correct_adult = res['correct_adult']
    correct_old = res['correct_old']

    accuracy = total_correct/total_items if total_items > 0 else 0

    data = [
        ["accuracy", accuracy],
        ["total_correct", total_correct],
        ["total_items", total_items],
        ["young_category", young_category],
        ["adult_category", adult_category],
        ["old_category", old_category],
        ["correct_young", correct_young],
        ["correct_adult", correct_adult],
        ["correct_old", correct_old],
    ]

    table = wandb.Table(columns=["Metric", "Value"], data=data)

    wandb.log({"age_evaluation_table": table})


def log_age_train_task_to_wandb(total_correct, total_items, correct_young, correct_adult, correct_old, epoch, loss):
    accuracy = total_correct/total_items if total_items > 0 else 0

    wandb.log({"loss": loss}, step=epoch)
    wandb.log({"accuracy": accuracy}, step=epoch)
    wandb.log({"correct_young": correct_young}, step=epoch)
    wandb.log({"correct_adult": correct_adult}, step=epoch)
    wandb.log({"correct_old": correct_old}, step=epoch)



def log_gender_downstream_task_to_wandb(res):
    """
            "total_correct": total_correct,
        "total_items": total_items,
        "male_category": gender_category_counts["Male"],
        "female_category": gender_category_counts["Female"],
        "correct_male": correct_predictions["Male"],
        "correct_female": correct_predictions["Female"]
    """
    total_correct = res['total_correct']
    total_items = res['total_items']
    male_category = res['male_category']
    female_category = res['female_category']
    correct_male = res['correct_male']
    correct_female = res['correct_female']

    accuracy = total_correct/total_items if total_items > 0 else 0

    data = [
        ["accuracy", accuracy],
        ["total_correct", total_correct],
        ["total_items", total_items],
        ["male_category", male_category],
        ["female_category", female_category],
        ["correct_male", correct_male],
        ["correct_female", correct_female]
    ]

    table = wandb.Table(columns=["Metric", "Value"], data=data)

    wandb.log({"gender_evaluation_table": table})
