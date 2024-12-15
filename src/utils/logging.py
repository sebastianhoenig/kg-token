import wandb


def log_to_wandb(res, epoch, example_ct, loss, optimizer):
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

    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    wandb.log({"epoch": epoch, "accuracy": accuracy}, step=example_ct)
    wandb.log({"epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']}, step=example_ct)
    wandb.log({"epoch": epoch, "yes_preds": yes_preds, "wrong_yes_preds": wrong_yes_preds}, step=example_ct)
    wandb.log({"epoch": epoch, "no_preds": no_preds, "wrong_no_preds": wrong_no_preds}, step=example_ct)
    wandb.log({"epoch": epoch, "yes_targets": yes_targets, "no_targets": no_targets}, step=example_ct)


def log_evaluation_to_wandb(res):
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
