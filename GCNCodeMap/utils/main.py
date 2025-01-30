import argparse
import torch
import pandas as pd
import os
from models.gcn import GCN, RGCN
from iterative_learning import gnn_learning, nb_learning
from graphs.data import HeterogeneousData, HomogeneousData, DataNB
import pickle


def main():

    parser = argparse.ArgumentParser(description="Train and evaluate models based on user-specified parameters.")

    # Required Arguments
    parser.add_argument("--model_type", type=str, choices=["gcn", "rgcn", "nb"], required=True,
                        help="Type of model: 'gcn' for Graph Convolutional Network, 'rgcn' for Relational GCN, 'nb' for Naïve Bayes.")
    parser.add_argument("--df_path", type=str, required=True, help="Path to the dataframe CSV file.")
    parser.add_argument("--df_dep_path", type=str, required=True, help="Path to the dependency dataframe CSV file.")
    
    # **Step 2: Training Parameters**
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer (default: 0.001).")
    parser.add_argument("--max_norm", type=float, default=5, help="Max norm for gradient clipping (default: 5).")
    
    # **Step 3: Iterative Learning Parameters**
    parser.add_argument("--lambda_t", type=float, default=0.8, help="Threshold scaling factor for iterative learning.")
    parser.add_argument("--split_ratio", type=float, default=0.05, help="Train-test split ratio (default: 5%).")
    parser.add_argument("--q_threshold", type=float, default=0.3, help="Quantile threshold for centrality filtering (default: 0.3).")

    # **Step 4: Model-Specific Parameters**
    parser.add_argument("--hidden_channels", type=int, default=16, help="Number of hidden channels (default: 16).")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.1).")
    parser.add_argument("--gcn_embed_dim", type=int, default=128, help="Embedding dimension for GCN (default: 128).")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden GCN layers (default: 1)")

    # **Step 5: Miscellaneous**
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times to run the model for the dataset.")
    parser.add_argument("--verbose", type=bool, default=True, help="Set verbosity (default: True)")

    args = parser.parse_args()

    # Create directory for saving results
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    print("\n✅ **Loading Data...**")
    df = pd.read_csv(args.df_path)
    df_dep = pd.read_csv(args.df_dep_path)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset based on the model type
    if args.model_type == "rgcn":
        print("✅ Creating Heterogeneous Dataset...")
        data = HeterogeneousData(df, df_dep)
        model = RGCN(hidden_channels=args.hidden_channels,
                     out_channels=data.num_classes,
                     num_layers=args.num_layers,
                     dropout=args.dropout,
                     embed_dim=args.gcn_embed_dim,
                     input_dim_dict={'entity': data['entity'].x.shape[1]},
                     relations=data.relations)
        model = model.to(device)
        data = data.to(device)

    elif args.model_type == "gcn":
        print("✅ Creating Homogeneous Dataset...")
        data = HomogeneousData(df, df_dep)
        model = GCN(input_dim=data.x.shape[1],
                    hidden_channels=args.hidden_channels,
                    out_channels=data.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout)
        model = model.to(device)
        data = data.to(device)

    elif args.model_type == "nb":
        print("✅ Creating Naïve Bayes Dataset...")
        data = DataNB(df, df_dep)

    else:
        raise ValueError("❌ Invalid model type!")

    train_indices, test_indices = data.generate_split(q_threshold=args.q_threshold, split_ratio=args.split_ratio)
    all_metrics = [] 
    for run in range(args.num_runs):
        print(f"\n🔄 **Run {run + 1}/{args.num_runs}**")

        if args.model_type in ['gcn', 'rgcn']:
            if args.model_type == "rgcn":
                model = RGCN(hidden_channels=args.hidden_channels,
                             out_channels=data.num_classes,
                             num_layers=args.num_layers,
                             dropout=args.dropout,
                             embed_dim=args.gcn_embed_dim,
                             input_dim_dict={'entity': data['entity'].x.shape[1]},
                             relations=data.relations)
            else:  # gcn
                model = GCN(input_dim=data.x.shape[1],
                            hidden_channels=args.hidden_channels,
                            out_channels=data.num_classes,
                            num_layers=args.num_layers,
                            dropout=args.dropout)

            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            loss_fn = torch.nn.CrossEntropyLoss()

            embeddings, metrics = gnn_learning(
                data, model, train_indices, test_indices,
                optimizer=optimizer, loss_fn=loss_fn,
                epochs=args.epochs, max_norm=args.max_norm, lambda_t=args.lambda_t,
                verbose=args.verbose
            )

            # Store embeddings and metrics for this run
            run_metrics = {f"run_{run + 1}": metrics}
            all_metrics.append(run_metrics)

            # Save embeddings for each run
            embeddings_path = os.path.join(results_dir, f"embeddings_run_{run + 1}.pkl")
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Saved embeddings for run {run + 1} at {embeddings_path}")

            # Print metrics for each run
            print("\n🎯 **Final GNN Metrics:**")
            print(metrics)

        elif args.model_type == "nb":
            metrics = nb_learning(
                X=data.X,
                Y=data.Y,
                initial_mapping_indices=train_indices,
                orphans_indices=test_indices,
                lambda_t=args.lambda_t,
                verbose=args.verbose
            )

            run_metrics = {f"run_{run + 1}": metrics}
            all_metrics.append(run_metrics)

            print("\n🎯 **Final Naïve Bayes Metrics:**")
            print(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(results_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved all metrics at {metrics_path}")

    print("\n✅ **Training & Evaluation Completed!")

if __name__ == "__main__":
    main()
