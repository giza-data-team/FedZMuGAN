import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.nn import functional as F


def entropy(p, dim=-1, keepdim=False):
    """
    Compute the entropy of the probability distribution `p` along the specified dimension.
    """

    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model, device):
    """
    Compute and collect the softmax probabilities from the model outputs.
    """
    prob = []
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for batch in data_loader:
            data, target = batch
            # Skip batches with only one sample
            if data.size(0) == 1:
                continue
            # Move data to the same device as the model
            data = data.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            # Compute softmax and move to CPU
            prob_batch = F.softmax(output, dim=-1).cpu()
            prob.append(prob_batch.data)
    return torch.cat(prob) if prob else torch.tensor([])


def get_membership_attack_data(
    retain_loader, forget_loader, test_loader, model, device
):
    """
    Generate entropy-based features and corresponding labels for membership attack.
    Returns features and labels for forget data (X_f, Y_f) and for retain/test data (X_r, Y_r).
    """

    retain_prob = collect_prob(retain_loader, model, device)
    forget_prob = collect_prob(forget_loader, model, device)
    test_prob = collect_prob(test_loader, model, device)

    X_r = (
        torch.cat([entropy(retain_prob), entropy(test_prob)])
        .cpu()
        .numpy()
        .reshape(-1, 1)
    )
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])

    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])
    return X_f, Y_f, X_r, Y_r


def get_membership_attack_prob(
    retain_loader, forget_loader, test_loader, model, device
):
    """
    Train a logistic regression classifier on entropy features derived from retain and test data and evaluate it on forget data.
    Returns the mean predicted membership probability for the forget data.
    """
    model = model.to(device)
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(
        retain_loader, forget_loader, test_loader, model, device
    )
    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs", multi_class="multinomial"
    )
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()
