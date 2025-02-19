#!/usr/bin/env python3
import argparse
import os
import sys
import datetime
import json
import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv
import pandas as pd
from PIL import Image
from tqdm import tqdm
from conch.open_clip_custom import create_model_from_pretrained
from torchvision import transforms
from info_nce import InfoNCE
import h5py
import scanpy as sc
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class ImageExpressionDataset(Dataset):
    def __init__(self, cases, image_dir, expression_dir, selected_genes=None, transform=None, args=None):
        self.cases = cases
        self.image_dir = image_dir
        self.expression_dir = expression_dir
        self.transform = transform
        self.selected_genes = selected_genes
        self.args = args
        
        self.image_size = 224
        
        # TODO: This is not the optimal way to load the data, 
        # as it loads all the data in memory, and is very slow at start up

        # From dataset, read what cases to load
        # Load cases as anndata
        # Filter genes that are included from list
        self.data_df = self.load_data()
        print(f'Loaded {len(self.data_df)} patches with expression data')


    def load_data(self):
        # For each case, load the expression data and the patches
        dfs = []
        for case in self.cases:
            # Load the patches
            df_patches = self.load_patches(case)
            # Load the expression
            adata = self.load_expressions(case)
            # Merge the data
            df_patches.loc[:, 'expression'] = list(adata[df_patches.barcode, :].X)
            
            dfs.append(df_patches)

        return pd.concat(dfs)

    def load_patches(self, case):
        # Open the file in read mode
        with h5py.File(f'{self.image_dir}/{case}.h5', 'r') as file:
            # Get the data
            imgs = list(file['img'])
            barcodes = list(file['barcode'])
            coords = list(file['coords'])

        barcodes = [b[0].decode('utf-8') for b in barcodes]
        df = pd.DataFrame(
            {
                'barcode': barcodes,
                'coord': coords,
                'img': [Image.fromarray(im) for im in imgs]
            }
        )
        return df

    def load_expressions(self, case):       
        adata = sc.read_h5ad(f'{self.expression_dir}/{case}.h5ad')
        adata.obs['batch'] = case  # Add filename as batch key

        adata.X = adata.X.astype('float32') # Convert to float32 before turning into a view

        # Filter to include only selected genes
        if self.selected_genes is not None:
            adata = adata[:, self.selected_genes]

        # Normalize and log transform
        # TODO: Should this be done for the whole dataset or per case?
        if self.args.normalize_total:
            target_sum = 1e6 if self.args.normalize_CPM else None
            exclude_highly_expressed = self.args.exclude_highly_expressed
            sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=exclude_highly_expressed)
        if self.args.log1p:
            sc.pp.log1p(adata)
        

        if self.args.smoothing:
            raise NotImplementedError("Smoothing not implemented yet")
        
        if self.args.st_PCA:
            raise NotImplementedError("PCA not implemented yet")
        
        return adata


    def __len__(self):
        return len(self.data_df.index)
   

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        
        image = row.img
        expression = row.expression

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, expression
    

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune and test CONCH models")
    # BCNB configuration
    parser.add_argument('--base_dir', type=str, default='/scratch/local/Data/BCNB') # /proj/hugek/BCNB
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                        help="Base folder for experiments")
    parser.add_argument('--ids_file', type=str,
                        default='patient-clinical-data.xlsx',
                        help="Excel file containing patient IDs and target values")
    parser.add_argument('--BCNB_data_folder', type=str,
                        help="Folder with patient image subfolders")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Batch size for processing images during testing")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="Number of workers for DataLoader")
    
    # Model and checkpoint
    parser.add_argument('--local_weights', action='store_true',
                        help="Use local weights instead of remote checkpoint")
    parser.add_argument('--model_cfg', type=str, default='conch_ViT-B-16',
                        help="Model configuration")
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help="Checkpoint path if using local weights (default: ./checkpoints/CONCH/pytorch_model.bin)")
    
    # Training parameters
    parser.add_argument('--train_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--train_batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Learning rate decay factor')
    parser.add_argument('--lr_layer_3', type=float, default=1e-4, help='Learning rate for the last block')
    parser.add_argument('--lr_mlp', type=float, default=0.003, help='Learning rate for the MLP')
    parser.add_argument('--mlp_input_dim', type=int, default=280, help='Input dimension for the MLP')
    parser.add_argument('--mlp_output_dim', type=int, default=512, help='Output dimension for the MLP')
    
    # InfoNCE and finetuning dataset
    parser.add_argument('--temperature', type=float, default=0.02, help='Temperature for InfoNCE loss')
    parser.add_argument('--finetuning_cases', type=str, default='TENX99,TENX96,TENX95,NCBI783,NCBI785',
                        help='Comma-separated list of cases for finetuning')
    parser.add_argument('--HEST_dir', type=str, default='/scratch/local/Data/HEST', # /proj/hugek/HEST
                        help='Directory with HEST data')
    parser.add_argument('--expression_dir', type=str, default='st',
                        help='Directory with expression data')
    parser.add_argument('--image_dir', type=str,
                        help='Directory with patient images')
    
    # Experiment parameters
    parser.add_argument('--transform', type=str, default='preprocess',
                        help='Transform to apply to images. Default is preprocess from CONCH with color jitter and random flips')
    parser.add_argument('--filtered_genes', type=str, default='filtered_genes.json',
                        help='JSON file with list of genes to include in the expression data')
    parser.add_argument('--brightness', type=float, default=0.1, help="Brightness factor for color jitter")
    parser.add_argument('--contrast', type=float, default=0.1, help="Contrast factor for color jitter")  
    parser.add_argument('--saturation', type=float, default=0.1, help="Saturation factor for color jitter")
    parser.add_argument('--hue', type=float, default=0.1, help="Hue factor for color jitter")
    parser.add_argument('--stain_normalize_finetune', action='store_true',
                        help="Use stain normalized patches instead of original patches in training")
    parser.add_argument('--stain_normalize_inference', action='store_true',
                        help="Use stain normalized patches instead of original patches in inference")
    parser.add_argument('--normalize_total', action='store_true',
                        help="Normalize the total counts across samples")
    parser.add_argument('--normalize_CPM', action='store_true',
                        help="Normalize to counts per million")
    parser.add_argument('--exclude_highly_expressed', action='store_true',
                        help="Exclude highly expressed genes from normalization")
    parser.add_argument('--log1p', action='store_true',
                        help="Apply log1p transformation to expression data")
    parser.add_argument('--smoothing', action='store_true',
                        help="Apply smoothing to the expression data")
    parser.add_argument('--st_PCA', action='store_true',
                        help="Perform PCA on spatial transcriptomics data")
    
    args, _ = parser.parse_known_args()
    
    # Set paths based on whether stain normalization is used
    if args.BCNB_data_folder is None:
        subfolder = 'patches_normalized' if args.stain_normalize_inference else 'patches'
        args.BCNB_data_folder = os.path.join(args.base_dir, f'paper_patches/{subfolder}')
    if args.image_dir is None:
        args.image_dir = 'patches_normalized' if args.stain_normalize_finetune else 'patches'
                        
    # Get list of arguments that were set from command line
    manual_args = {action.dest: getattr(args, action.dest)
                   for action in parser._actions
                   if any(opt in sys.argv for opt in action.option_strings)}
    
    return args, manual_args

def create_experiment_folder(experiment_dir):
    os.makedirs(experiment_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(experiment_dir, timestamp)
    os.makedirs(run_folder, exist_ok=True)
    print(f"Created experiment folder at {run_folder}")
    return run_folder

def save_args(args, folder):
    args_file = os.path.join(folder, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4)

def init_conch(args):
    load_dotenv()
    # Set checkpoint path based on local_weights flag
    if args.local_weights:
        ckpt_path = args.checkpoint_path if args.checkpoint_path else './checkpoints/CONCH/pytorch_model.bin'
        hf_auth_token = None
    else:
        ckpt_path = 'hf_hub:MahmoodLab/conch'
        hf_auth_token = os.getenv("HF_AUTH_TOKEN")
        if hf_auth_token is None:
            raise ValueError("HF_AUTH_TOKEN environment variable not set")

    model, preprocess = create_model_from_pretrained(args.model_cfg, ckpt_path, hf_auth_token=hf_auth_token)
    model_vit = model.visual
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_vit = model_vit.to(device)

    return model_vit, preprocess

def init_models(args):
    model_vit, preprocess = init_conch(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Freeze all layers
    for param in model_vit.parameters():
        param.requires_grad = False
    # Unfreeze the last 3 blocks
    for param in model_vit.trunk.blocks[-3:].parameters():
        param.requires_grad = True

    # Define a 3-layer MLP for additional finetuning
    model_mlp = nn.Sequential(
        nn.Linear(args.mlp_input_dim, args.mlp_output_dim * 2),
        nn.ReLU(),
        nn.Linear(args.mlp_output_dim * 2, args.mlp_output_dim * 2),
        nn.ReLU(),
        nn.Linear(args.mlp_output_dim * 2, args.mlp_output_dim)
    )
    model_mlp = model_mlp.to(device)

    # Training configuration
    epochs = args.train_epochs
    train_batch_size = args.train_batch_size
    lr_layer_3 = args.lr_layer_3
    lr_layer_2 = lr_layer_3 * args.lr_decay
    lr_layer_1 = lr_layer_2 * args.lr_decay

    # Define optimizers with layerwise learning rate decay
    param_groups = [
        {'params': model_vit.trunk.blocks[-3].parameters(), 'lr': lr_layer_1},
        {'params': model_vit.trunk.blocks[-2].parameters(), 'lr': lr_layer_2},
        {'params': model_vit.trunk.blocks[-1].parameters(), 'lr': lr_layer_3}
    ]

    optimizer_vit = torch.optim.Adam(param_groups)
    optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=args.lr_mlp)

    scheduler_vit = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vit, T_max=epochs)
    scheduler_mlp = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mlp, T_max=epochs)


    if args.transform == 'preprocess':
        aug = transforms.Compose([
            transforms.ColorJitter(brightness=args.brightness, contrast=args.contrast, saturation=args.saturation, hue=args.hue),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        # Extend augmentation with the model's preprocess transforms
        aug.transforms.extend(preprocess.transforms)

    # Define finetuning dataset
    cases = args.finetuning_cases.split(',')
    expression_dir = os.path.join(args.HEST_dir, args.expression_dir)
    image_dir = os.path.join(args.HEST_dir, args.image_dir)
    with open(args.filtered_genes, 'r') as f:
        selected_genes = json.load(f)

    dataset = ImageExpressionDataset(cases=cases, image_dir=image_dir, expression_dir=expression_dir,
                                     selected_genes=selected_genes, transform=aug, args=args)
    data_loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=args.num_workers)

    return model_vit, preprocess, model_mlp, data_loader, optimizer_vit, optimizer_mlp, scheduler_vit, scheduler_mlp

def finetune_model(args, run_folder):
    model_vit, preprocess, model_mlp, data_loader, optimizer_vit, optimizer_mlp, scheduler_vit, scheduler_mlp = init_models(args)
    epochs = args.train_epochs

    # Set up InfoNCE loss
    infoloss = InfoNCE(temperature=args.temperature, negative_mode='unpaired')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_folder = os.path.join(run_folder, 'checkpoints')
    os.makedirs(checkpoint_folder, exist_ok=True)

    # Training loop
    
    training_metrics = []  # List to store loss per epoch

    for epoch in range(1, epochs + 1):
        model_vit.train()
        model_mlp.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
        for img, expr in train_bar:
            img, expr = img.to(device, non_blocking=True), expr.to(device, non_blocking=True)
            out_1, _ = model_vit(img)
            out_2 = model_mlp(expr)

            loss = infoloss(out_1, out_2)
            optimizer_vit.zero_grad()
            optimizer_mlp.zero_grad()
            loss.backward()
            optimizer_vit.step()
            optimizer_mlp.step()

            total_num += args.train_batch_size
            total_loss += loss.item() * args.train_batch_size
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

        epoch_loss = total_loss / total_num
        scheduler_vit.step()
        scheduler_mlp.step()
        training_metrics.append({'epoch': epoch, 'loss': epoch_loss})

        if epoch % 10 == 0:
            finetuned_model_path = os.path.join(checkpoint_folder, f'epoch_{epoch}.pth')
            torch.save(model_vit.state_dict(), finetuned_model_path)
            print(f"Checkpoint saved at epoch {epoch}:", finetuned_model_path)
    
    finetuned_model_path = os.path.join(checkpoint_folder, 'finetuned_model.pth')
    torch.save(model_vit.state_dict(), finetuned_model_path)
    print("Finetuning complete. Finetuned model saved at", finetuned_model_path)

    # Save training metrics to CSV
    if training_metrics:
        metrics_df = pd.DataFrame(training_metrics)
        metrics_csv_path = os.path.join(run_folder, 'training_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print("Training metrics saved at", metrics_csv_path)

    return model_vit, preprocess

# Define a custom Dataset to load patient images
class PatientImagesDataset(Dataset):
    def __init__(self, patient_ids, data_folder, transform):
        self.samples = []
        for pid in patient_ids:
            patient_folder = os.path.join(data_folder, str(pid))
            if not os.path.exists(patient_folder):
                print(f"Warning: Patient folder {patient_folder} does not exist.")
                continue
            image_files = [f for f in os.listdir(patient_folder) if f.endswith('.jpg')]
            for img_file in image_files:
                self.samples.append((pid, os.path.join(patient_folder, img_file)))
        self.transform = transform

        print(f'Loaded {len(self.samples)} patches from BCNB data')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        pid, path = self.samples[index]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return pid, image

def extract_embeddings(args, run_folder):
    # Create folder to save embeddings
    embeddings_folder = os.path.join(run_folder, 'embeddings')
    os.makedirs(embeddings_folder, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load patient data and target values
    data_file = os.path.join(args.base_dir, args.ids_file)
    patient_data_df = pd.read_excel(data_file)
    ids = patient_data_df['Patient ID'].tolist()
    target_columns = ['ER', 'PR', 'HER2']
    target_values = patient_data_df.set_index('Patient ID')[target_columns].to_dict('index')

    base_model, preprocess = init_conch(args)

    # Create a DataLoader that processes all patient images in one go
    dataset = PatientImagesDataset(ids, args.BCNB_data_folder, preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    checkpoints_folder = os.path.join(run_folder, 'checkpoints')
    for checkpoint_file in os.listdir(checkpoints_folder):
        #if not checkpoint_file.endswith('.pth'):
        if not checkpoint_file.endswith('finetuned_model.pth'):
            continue
        
        # Skip if the CSV file already exists
        csv_path = os.path.join(embeddings_folder, checkpoint_file.replace('.pth', '.csv'))
        if os.path.exists(csv_path):
            print(f"Embeddings already extracted for {checkpoint_file}. SKipping...")
            continue

        # Load checkpoint weights and update the model
        ckpt_path = os.path.join(checkpoints_folder, checkpoint_file)
        print(f"Loading checkpoint: {ckpt_path}")
        cloned_model = copy.deepcopy(base_model)
        cloned_model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        cloned_model.to(device)
        cloned_model.eval()

        # Dictionary to aggregate embeddings per patient
        patient_embeddings = {}

        with torch.no_grad():
            for pids, images in tqdm(dataloader, desc=f'Processing {os.path.basename(ckpt_path)}'):
                images = images.to(device)
                outputs, _ = cloned_model(images)
                outputs = outputs.cpu()
                for pid, feat in zip(pids, outputs):
                    pid = pid.item() if isinstance(pid, torch.Tensor) else pid
                    patient_embeddings.setdefault(pid, []).append(feat)

        all_patients_embeddings = []
        for pid, feats in patient_embeddings.items():
            feats_tensor = torch.stack(feats, dim=0)
            mean_embedding = feats_tensor.mean(dim=0)
            df = pd.DataFrame([mean_embedding.numpy()])
            df.insert(0, 'patient_id', pid)
            # Add target values to the DataFrame
            for target in target_columns:
                df[target] = target_values.get(pid, {}).get(target, None)
            all_patients_embeddings.append(df)

        final_df = pd.concat(all_patients_embeddings, ignore_index=True)
        final_df.to_csv(csv_path, index=False)
        print(f"Testing complete. Embeddings saved at {csv_path}")

def load_embeddings(run_folder):
    embeddings = {}
    embeddings_folder = os.path.join(run_folder, 'embeddings')
    for f in os.listdir(embeddings_folder):
        if not f.endswith('.csv'):
            continue
        csv_path = os.path.join(embeddings_folder, f)
        ckpt_name = f.replace('.csv', '')
        embeddings[ckpt_name] = pd.read_csv(csv_path)
    return embeddings

def train_and_test_ridge(args, run_folder, random_state=42, max_iter=1000):
    embeddings = load_embeddings(run_folder)
    splitting_folder = os.path.join(args.base_dir, "dataset-splitting")
    train_indices = np.loadtxt(os.path.join(splitting_folder, 'train_id.txt'), dtype=int).tolist()
    test_indices = np.loadtxt(os.path.join(splitting_folder, 'test_id.txt'), dtype=int).tolist()

    results = {}
    for model_name, df in embeddings.items():
        X = df.drop(columns=['patient_id', 'ER', 'PR', 'HER2']).values
        y = df[['ER', 'PR', 'HER2']].map(lambda x: 1 if x == 'Positive' else 0).values

        X_train = df[df['patient_id'].isin(train_indices)].drop(columns=['patient_id', 'ER', 'PR', 'HER2']).values
        X_test = df[df['patient_id'].isin(test_indices)].drop(columns=['patient_id', 'ER', 'PR', 'HER2']).values
        y_train = y[df['patient_id'].isin(train_indices)]
        y_test = y[df['patient_id'].isin(test_indices)]

        print(f"Using LogisticRegression for model {model_name}")

        preds_all = np.zeros_like(y_test, dtype=float)
        for i in range(y_train.shape[1]):
            clf = LogisticRegression(random_state=random_state, max_iter=max_iter)
            clf.fit(X_train, y_train[:, i])
            preds_all[:, i] = clf.predict_proba(X_test)[:, 1]

        # Compute and store predictions and basic accuracies
        preds_bin = (preds_all > 0.5).astype(int)
        accuracies = (preds_bin == y_test).mean(axis=0)
        aucs = [roc_auc_score(y_test[:, i], preds_all[:, i]) for i in range(y_test.shape[1])]
        bal_accs = [balanced_accuracy_score(y_test[:, i], preds_bin[:, i]) for i in range(y_test.shape[1])]

        print(f"Accuracy for {model_name}: ER={accuracies[0]:.3f}, PR={accuracies[1]:.3f}, HER2={accuracies[2]:.3f}")
        print(f"AUC for {model_name}: ER={aucs[0]:.3f}, PR={aucs[1]:.3f}, HER2={aucs[2]:.3f}")
        print(f"Balanced accuracy for {model_name}: ER={bal_accs[0]:.3f}, PR={bal_accs[1]:.3f}, HER2={bal_accs[2]:.3f}")

        results[model_name] = {
            'seed': random_state,
            'predictions': preds_all,
            'accuracy': {'ER': accuracies[0], 'PR': accuracies[1], 'HER2': accuracies[2]},
            'auc': {'ER': aucs[0], 'PR': aucs[1], 'HER2': aucs[2]},
            'balanced_accuracy': {'ER': bal_accs[0], 'PR': bal_accs[1], 'HER2': bal_accs[2]}
        }

        json_file = os.path.join(run_folder, 'results.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=4, default=lambda o: o.tolist() if hasattr(o, 'tolist') else o)

        rows = []
        for model, metrics in results.items():
            for metric_type in ['accuracy', 'auc', 'balanced_accuracy']:
                row = {
                    'Model': model,
                    'Seed': random_state,
                    'Metric': metric_type,
                    'ER': metrics[metric_type]['ER'],
                    'PR': metrics[metric_type]['PR'],
                    'HER2': metrics[metric_type]['HER2']
                }
                rows.append(row)

        df_results = pd.DataFrame(rows)
        result_file = os.path.join(run_folder, 'results.csv')
        write_header = not os.path.exists(result_file)
        df_results.to_csv(result_file, index=False, mode='a', header=write_header)

    return df_results

def visualize_results(run_folder):
    # Create a directory to save plots
    plots_dir = os.path.join(run_folder, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Plot training loss over time
    training_metrics_path = os.path.join(run_folder, 'training_metrics.csv')
    if os.path.exists(training_metrics_path):
        metrics_df = pd.read_csv(training_metrics_path)
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['epoch'], metrics_df['loss'], marker='o', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        loss_plot_path = os.path.join(plots_dir, 'training_loss.png')
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Saved training loss plot at {loss_plot_path}")
    
    # Plot predictions for each model
    results_path = os.path.join(run_folder, 'results.csv')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at {results_path}")
    df_results = pd.read_csv(results_path)

    # Melt data for plotting
    df_melted = df_results.melt(
        id_vars=['Model', 'Seed', 'Metric'],
        value_vars=['ER', 'PR', 'HER2'],
        var_name='Marker',
        value_name='Score'
    )

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df_melted, x='Metric', y='Score', hue='Marker')
    ax.set_title("Metrics Distribution")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)  # Ensure y-axis is always between 0 and 1
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    plt.tight_layout()

    results_plot_path = os.path.join(plots_dir, 'results_plot.png')
    plt.savefig(results_plot_path)
    plt.close()
    print(f"Saved results plot at {results_plot_path}")
    



def main():
    assert torch.cuda.is_available(), "CUDA not available"
    args, manual_args = parse_args()
    experiment_dir = os.path.join(
        args.base_dir,
        args.experiment_dir,
        "Defaults+"+",".join(
            f"{k};{str(v).replace('/', '_')}" 
            for k, v in manual_args.items()
        )
    )
    run_folder = create_experiment_folder(experiment_dir)
    save_args(args, run_folder)
    

    model_vit, preprocess = finetune_model(args, run_folder)
    extract_embeddings(args, run_folder)
    results = train_and_test_ridge(args, run_folder)
    visualize_results(run_folder)
    
    print("Finetuning process completed successfully.")

if __name__ == "__main__":
    main()
