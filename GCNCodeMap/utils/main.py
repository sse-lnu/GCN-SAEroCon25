import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import pandas as pd
import os
from models.gnn import GCN, RGCN
from iterative_learning import gnn_learning, nb_learning
from graphs.data import  DataNB, HomogeneousData, HeterogeneousData



def main():

    parser = argparse.ArgumentParser(description="Train and evaluate models based on user-specified parameters.")

    # Required Arguments
    parser.add_argument("--model_type", type=str, choices=["gcn", "rgcn", "nb"], required=True,
                        help="Type of model: 'gcn' for Graph Convolutional Network, 'rgcn' for Relational GCN, 'nb' for Naïve Bayes.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataframe CSV file.")
    parser.add_argument("--dependencies_path", type=str, required=True, help="Path to the dependency dataframe CSV file.")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset to use when saving results.")
    
    # Optional Arguments
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer (default: 0.001).")
    parser.add_argument("--max_norm", type=float, default=5, help="Max norm for gradient clipping (default: 5).")
    parser.add_argument("--lambda_t", type=lambda x: eval(x, {"__builtins__": None}, {}) if 'lambda' in x else (float(x) if '.' in x else int(x)), 
    default=0.8, help="Threshold scaling factor for iterative learning. Can be an int, float, or a lambda function.")
    parser.add_argument("--split_ratio", type=float, default=0.05, help="Train-test split ratio (default: 5%).")
    parser.add_argument("--q_threshold", type=float, default=0.3, help="Quantile threshold for centrality filtering (default: 0.3).")
    parser.add_argument("--hidden_channels", type=int, default=16, help="Number of hidden channels (default: 16).")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2).")
    parser.add_argument("--gcn_embed_dim", type=int, default=128, help="Embedding dimension for GCN (default: 128).")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden GCN layers (default: 1).")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of times to run the model for the dataset.")
    parser.add_argument("--verbose", type=bool, default=True, help="Set verbosity (default: True)")

    args = parser.parse_args()


    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    print("\n **Loading Data...**")
    df = pd.read_csv(args.data_path)
    df_dep = pd.read_csv(args.dependencies_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.model_type == "rgcn":
        print(f" Creating Heterogeneous Dataset for ...")
        data = HeterogeneousData(df, df_dep)
        data = data.to(device)

    elif args.model_type == "gcn":
        print(f" Creating Homogeneous Dataset for ...")
        data = HomogeneousData(df, df_dep)
        data = data.to(device)

    elif args.model_type == "nb":
        print(f" Creating Naïve Bayes Dataset for...")
        data = DataNB(df, df_dep)

    else:
        raise ValueError("Invalid model type!")

    results_dir = os.path.join('results', args.model_type, args.data_name)
    os.makedirs(results_dir, exist_ok=True)
    all_metrics = []  

    for run in range(args.num_runs):
        print(f"\n**Run {run + 1}/{args.num_runs}**")

        train_indices, test_indices = data.generate_split(q_threshold = args.q_threshold, split_ratio=args.split_ratio)
        
        if args.model_type in ['gcn', 'rgcn']:
            if args.model_type == "rgcn":
                model = RGCN(hidden_channels=args.hidden_channels,
                    out_channels=data.num_classes,
                    num_layers=args.num_layers,
                    dropout=args.dropout,
                    embed_dim=args.gcn_embed_dim,
                    input_dim_dict={'entity': data['entity'].x.shape[1]},
                    relations=data.relations)
            else:
                model = GCN(input_dim=data.x.shape[1],
                        hidden_channels=args.hidden_channels,
                        embed_dim=args.gcn_embed_dim,
                        out_channels=data.num_classes,
                        num_layers=args.num_layers,
                        dropout=args.dropout)
                
            model.apply(lambda layer: layer.reset_parameters() if hasattr(layer, 'reset_parameters') else None)
            model = model.to(device)

            metrics = gnn_learning(data,
                                   model,
                                   train_indices,
                                   test_indices,
                                   loss_fn=torch.nn.CrossEntropyLoss(),
                                   epochs=args.epochs,
                                   lr = args.lr,
                                   max_norm=args.max_norm,
                                   lambda_t=args.lambda_t,
                                   verbose=args.verbose
            )

            all_metrics.append({**metrics, 'run': run + 1})

        elif args.model_type == "nb":
            metrics = nb_learning(
                X=data.X,
                Y=data.Y,
                initial_mapping_indices=train_indices,
                orphans_indices=test_indices,
                lambda_t=args.lambda_t,
                verbose=args.verbose
            )
            all_metrics.append({**metrics, 'run': run + 1})
           

    metrics_df = pd.DataFrame(all_metrics)

    # Save combined metrics and embeddings in a single CSV file
    final_path = os.path.join(results_dir, f"results_{args.data_name}.csv")
    metrics_df.to_csv(final_path, index=False)
    print(f"Saved combined results at {final_path}")

    print("\n**Training & Evaluation Completed!")


print("\n**Training & Evaluation Completed!")

if __name__ == "__main__":
    main()
