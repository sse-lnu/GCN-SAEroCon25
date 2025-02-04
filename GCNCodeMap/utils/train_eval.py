import torch

def train(model, data, train_indices, optimizer, loss_fn, max_norm=5):
    """
    Perform a single training step for the model.

    Args:
        model: The model being trained.
        data: The dataset used for training.
        train_indices: Indices of the training set.
        optimizer: The optimizer used for gradient descent.
        loss_fn: The loss function used for training.
        max_norm: Maximum norm for gradient clipping.

    Returns:
        The loss value for the current training step.
    """
    model.train()
    optimizer.zero_grad()

    device = next(model.parameters()).device
    train_indices = train_indices.to(device)

    if hasattr(data, "x_dict") and hasattr(data, "edge_index_dict"):  # RGCN
        data['entity'].y = data['entity'].y.to(device)
        out = model(data.x_dict, data.edge_index_dict)
        loss = loss_fn(out[train_indices], data['entity'].y[train_indices])
    else:  # GCN
        data.y = data.y.to(device)
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[train_indices], data.y[train_indices])

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()

    return loss.item()


def evaluate(model, data, orphans, lambda_t=None, iteration=None):
    """
    Evaluate the model on the given data to calculate metrics.

    Args:
        model: The trained model.
        data: The dataset for evaluation.
        orphans: Indices of the orphan nodes to evaluate.
        lambda_t: Threshold scaling factor for confidence score.
        iteration: The current iteration in iterative learning.

    Returns:
        high_conf_indices: Indices of the high confidence orphans.
        high_conf_labels: Predicted labels for the high confidence orphans.
        low_conf_indices: Indices of the low confidence orphans.
        confidence_threshold: The calculated confidence threshold.
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        orphans = orphans.to(device)

        if hasattr(data, "x_dict") and hasattr(data, "edge_index_dict"):  
            data['entity'].y = data['entity'].y.to(device)
            out = model(data.x_dict, data.edge_index_dict).to(device)
            out_test = out[orphans]
        else: 
            data.y = data.y.to(device)
            out = model(data.x, data.edge_index).to(device)
            out_test = out[orphans]

        probs = torch.softmax(out_test, dim=1)  
        confidence_scores, predicted_classes = torch.max(probs, dim=1)  

        if lambda_t is not None:
            mean_confidence = confidence_scores.mean()
            std_confidence = confidence_scores.std() if confidence_scores.numel() > 1 else 0.0
            if callable(lambda_t):
                threshold = lambda_t(iteration)
                confidence_threshold = (mean_confidence + std_confidence) * threshold
            elif isinstance(lambda_t, (float, int)):
                confidence_threshold = (mean_confidence + std_confidence) * lambda_t
            else:
                raise ValueError("lambda_t must be a callable, float, or integer.")
        else:
            confidence_threshold = 0.9  

        confidence_threshold = min(confidence_threshold, 0.98)  

        # Identify high-confidence and low-confidence orphans
        high_conf_mask = confidence_scores >= confidence_threshold
        high_conf_indices = orphans[high_conf_mask]
        high_conf_labels = predicted_classes[high_conf_mask]
        low_conf_indices = orphans[~high_conf_mask]

        return high_conf_indices, high_conf_labels, low_conf_indices, confidence_threshold
