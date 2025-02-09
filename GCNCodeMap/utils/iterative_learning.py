import torch
from train_eval import train, evaluate
import numpy as np
from graphs.data import HeterogeneousData
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, precision_score, recall_score

def gnn_learning(data, model, 
                   initial_mapping, 
                   orphans, 
                   loss_fn, 
                   lr, 
                   max_norm, 
                   lambda_t,
                   epochs=50, 
                   verbose=True):
    
    is_heterogeneous = isinstance(data, HeterogeneousData)
    device = next(model.parameters()).device
    mapped_entities = initial_mapping.to(device)
    orphans = orphans.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    predicted_labels = {}
    updated_entities = set()
    iteration = 0
    total_nodes = 0
    node_embeddings = 0
    metrics_history = metrics_history = []

    while True:
        for _ in range(epochs):
            # training
            loss = train(model, data, mapped_entities, optimizer, loss_fn, max_norm)

        # evaluating
        high_conf_indices, high_conf_labels, new_orphans, threshold = evaluate(
            model, data, orphans, lambda_t=lambda_t, iteration=iteration
        )

        for idx, label in zip(high_conf_indices.tolist(), high_conf_labels.tolist()):
            predicted_labels[idx] = label

        mapped_entities = torch.cat([mapped_entities, high_conf_indices]).to(device)
        orphans = new_orphans.to(device)

        if len(predicted_labels) > 0:
            mapped_entity_indices = torch.tensor(list(predicted_labels.keys()), device=device)

            if is_heterogeneous:
                out = model(data.x_dict, data.edge_index_dict).to(device)
            else:
                out = model(data.x, data.edge_index).to(device)
        
            probs = torch.softmax(out[mapped_entity_indices], dim=1)
            confidence_scores, new_predictions = torch.max(probs, dim=1)

            node_embeddings = out

            for idx, score, new_label in zip(mapped_entity_indices.tolist(), confidence_scores.tolist(), new_predictions.tolist()):
                if score >= 0.9 and predicted_labels[idx] != new_label:
                    predicted_labels[idx] = new_label
                    updated_entities.add(idx)

        if predicted_labels:
            mapped_entity_indices = list(predicted_labels.keys())
            mapped_predicted_labels = list(predicted_labels.values())
            if is_heterogeneous:
                true_labels = data['entity'].y[mapped_entity_indices].cpu()
                total_nodes = len(data['entity'].x)
            else:
                true_labels = data.y[mapped_entity_indices].cpu()
                total_nodes = len(data.x)

            # Calculate metrics
            current_f1_micro = f1_score(true_labels, mapped_predicted_labels, average='micro', zero_division=1)
            current_f1_macro = f1_score(true_labels, mapped_predicted_labels, average='macro',zero_division=1)
            current_precision_micro = precision_score(true_labels, mapped_predicted_labels, average='micro',zero_division=1)
            current_precision_macro = precision_score(true_labels, mapped_predicted_labels, average='macro',zero_division=1)
            current_recall_micro = recall_score(true_labels, mapped_predicted_labels, average='micro',zero_division=1)
            current_recall_macro = recall_score(true_labels, mapped_predicted_labels, average='macro',zero_division=1)

            metrics_history.append({
                "iteration": iteration + 1,
                "f1_micro": current_f1_micro,
                "f1_macro": current_f1_macro,
                "precision_micro": current_precision_micro,
                "precision_macro": current_precision_macro,
                "recall_micro": current_recall_micro,
                "recall_macro": current_recall_macro,
                "mapped_ratio": (len(mapped_entities) + len(initial_mapping)) / total_nodes,
                "remaining_orphans_ratio": len(orphans) / total_nodes,
            })

            if verbose:
                print(f"Iteration {iteration + 1} - F1_micro: {current_f1_micro:.3f}, F1_macro: {current_f1_macro:.3f}, "
                    f"Confidence Threshold: {threshold:.3f}, Mapped Entities: {len(mapped_entities)}, "
                    f"Updated Entities: {len(updated_entities)}, Remaining Orphans: {len(orphans)}")
        
        if len(high_conf_indices) == 0 or len(orphans) == 0:
            if verbose:
                print("No new entities mapped or no orphans left. Stopping iterations.")
            break
        iteration += 1
        
    results = {
        "initial_set_size": len(initial_mapping),
        "final_mapped_size": len(mapped_entities),
        "final_unmapped_size": len(orphans),
        "mapped_entities_indices": mapped_entities.cpu().numpy(),
        "unmapped_entities_indices": orphans.cpu().numpy(),
        "initial_set_indices": initial_mapping.cpu().numpy(),
        "metrics_history": metrics_history,
        "final_f1_macro": current_f1_macro,
        "final_f1_micro": current_f1_micro,
        "final_precision_macro": current_precision_macro,
        "final_precision_micro": current_precision_micro,
        "final_recall_macro": current_recall_macro,
        "final_recall_micro": current_recall_micro,
        'node_embeddings': node_embeddings.detach().cpu().numpy()
    }

    return results

################### Niave BAYES lEARNING ###################### 

def nb_learning(X, Y, initial_mapping_indices, orphans_indices, lambda_t=None, test_mapped_entities=True, verbose=True):
    """
    Perform iterative learning for Naive Bayes model.

    Args:
        X: Features (input data).
        Y: Labels (true classes).
        initial_mapping_indices: The initial set of mapped entities.
        orphans_indices: The orphans that need to be mapped.
        lambda_t: The scaling factor for threshold adjustment.
        test_mapped_entities: Whether to test mapped entities (default: True).
        verbose: Whether to print detailed iteration logs (default: True).

    Returns:
        metrics: A dictionary with evaluation metrics after the iterative learning.
    """
    mapped_entities = initial_mapping_indices
    orphans = orphans_indices
    predicted_labels = {}
    updated_entities = set()
    iteration = 0

    model = MultinomialNB()

    metrics_history = []

    while True:
        # Train the model on currently mapped entities
        X_train = X[mapped_entities]
        Y_train = Y[mapped_entities]
        model.fit(X_train, Y_train)

        # Predict labels and confidence scores for orphans
        X_orphans = X[orphans]
        probs = model.predict_proba(X_orphans)
        confidence_scores = np.max(probs, axis=1)
        predicted_classes = model.classes_[np.argmax(probs, axis=1)]

        # Determine the confidence threshold
        if lambda_t is not None:
            mean_confidence = confidence_scores.mean()
            std_confidence = confidence_scores.std() if confidence_scores.size > 1 else 0.0
            threshold = lambda_t if isinstance(lambda_t, float) else lambda_t(iteration)
            confidence_threshold = mean_confidence + std_confidence * threshold
            confidence_threshold = min(confidence_threshold, 0.98)
        else:
            confidence_threshold = 0.9

        # Identify high-confidence and low-confidence orphans
        high_conf_mask = confidence_scores >= confidence_threshold
        high_conf_indices = orphans[high_conf_mask]
        high_conf_labels = predicted_classes[high_conf_mask]
        low_conf_indices = orphans[~high_conf_mask]

        # Update mappings and labels for high-confidence orphans
        newly_mapped_count = 0
        for idx, label in zip(high_conf_indices, high_conf_labels):
            if idx not in predicted_labels:  # Check if it's a newly mapped entity
                predicted_labels[idx] = label
                newly_mapped_count += 1

        mapped_entities = np.concatenate([mapped_entities, high_conf_indices])
        orphans = low_conf_indices

        # Reevaluate predictions for mapped entities if enabled
        if test_mapped_entities and len(predicted_labels) > 0:
            X_mapped = X[list(predicted_labels.keys())]
            probs_mapped = model.predict_proba(X_mapped)
            confidence_scores_mapped = np.max(probs_mapped, axis=1)
            new_predictions = model.classes_[np.argmax(probs_mapped, axis=1)]

            for idx, score, new_label in zip(predicted_labels.keys(), confidence_scores_mapped, new_predictions):
                if score >= 0.9 and predicted_labels[idx] != new_label:
                    predicted_labels[idx] = new_label
                    updated_entities.add(idx)

        # Verbose iteration details
        if verbose and predicted_labels:
            mapped_entity_indices = list(predicted_labels.keys())
            mapped_predicted_labels = list(predicted_labels.values())
            true_labels = Y[mapped_entity_indices]

            # Calculate metrics
            current_f1_micro = f1_score(true_labels, mapped_predicted_labels, average='micro', zero_division=1)
            current_f1_macro = f1_score(true_labels, mapped_predicted_labels, average='macro',zero_division=1)
            current_precision_micro = precision_score(true_labels, mapped_predicted_labels, average='micro',zero_division=1)
            current_precision_macro = precision_score(true_labels, mapped_predicted_labels, average='macro',zero_division=1)
            current_recall_micro = recall_score(true_labels, mapped_predicted_labels, average='micro',zero_division=1)
            current_recall_macro = recall_score(true_labels, mapped_predicted_labels, average='macro',zero_division=1)

            metrics_history.append({
                "iteration": iteration + 1,
                "initial_set_size": len(initial_mapping_indices),
                "f1_micro": current_f1_micro,
                "f1_macro": current_f1_macro,
                "precision_micro": current_precision_micro,
                "precision_macro": current_precision_macro,
                "recall_micro": current_recall_micro,
                "recall_macro": current_recall_macro,
                "mapped_ratio": (len(mapped_entities) + len(initial_mapping_indices)) / len(X),
                "remaining_orphans_ratio": len(orphans) / len(X),
            })

            print(f"Iteration {iteration + 1} - F1_micro: {current_f1_micro:.3f}, F1_macro: {current_f1_macro:.3f}, "
                  f"Confidence Threshold: {threshold:.3f}, Mapped Entities: {len(mapped_entities)}, "
                  f"Updated Entities: {len(updated_entities)}, Remaining Orphans: {len(orphans)}")

        # Stopping condition
        if len(high_conf_indices) == 0 or len(orphans) == 0:
            if verbose:
                print("No new entities mapped or no orphans left. Stopping iterations.")
            break

        iteration += 1

    return {
        "initial_set_size": len(initial_mapping_indices),
        "final_mapped_size": len(mapped_entities),
        "final_unmapped_size": len(orphans),
        "final_f1_macro": current_f1_macro,
        "final_f1_micro": current_f1_micro,
        "final_precision_macro": current_precision_macro,
        "final_precision_micro": current_precision_micro,
        "final_recall_macro": current_recall_macro,
        "final_recall_micro": current_recall_micro,
        "metrics_history": metrics_history,
    }

