import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightning.pytorch as pl
from scipy.ndimage import gaussian_filter1d
import ast
import copy

class PostProcessTraining(pl.Callback):
    """
    PyTorch Lightning callback for post-processing during training.
    NOTE: Implemented for GBF-B-GUN only
    """
    
    def __init__(self, 
                 interval=100,
                 pyg_graph_dict_test=None,
                 dataset_info_path=None,
                 model_name="model",
                 output_dir="./training_snapshots/",
                 sigma=10):
        """
        Initialize the post-processing callback
        
        Args:
            interval: Run callback every 'interval' epochs
            pyg_graph_dict_test: Test data graph dictionary
            dataset_info_path: Path to directory containing dataset info files
            model_name: Name of the model being trained
            output_dir: Directory to save outputs
            sigma: Sigma for smoothing in loss plots
        """
        # Call parent constructor properly - critical for PyTorch Lightning
        super(PostProcessTraining, self).__init__()
        
        self.interval = interval
        self.pyg_graph_dict_test = pyg_graph_dict_test
        self.dataset_info_path = dataset_info_path
        self.model_name = model_name
        self.output_dir = output_dir
        self.sigma = sigma
        self.minmax_df = None
        self.test_data_df = None
        
        # Create output directories
        Path(f"{output_dir}/figures").mkdir(parents=True, exist_ok=True)
    
    def _load_data_if_needed(self):
        """Load data files if they haven't been loaded yet"""
        if self.minmax_df is None or self.test_data_df is None:
            try:
                minmax_stats_path = os.path.join(self.dataset_info_path, "minmax_normalization_stats.csv")
                test_data_path = os.path.join(self.dataset_info_path, "test_dataset.csv")
                
                if os.path.exists(minmax_stats_path) and os.path.exists(test_data_path):
                    self.minmax_df = pd.read_csv(minmax_stats_path)
                    self.test_data_df = pd.read_csv(test_data_path, dtype=str)
                    return True
                else:
                    print(f"Warning: Required files not found at {self.dataset_info_path}")
                    print(f"Minmax file exists: {os.path.exists(minmax_stats_path)}")
                    print(f"Test data file exists: {os.path.exists(test_data_path)}")
                    return False
            except Exception as e:
                print(f"Error loading data files: {e}")
                return False
        return True
    
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Run post-processing after first epoch and at specified intervals
        """
        current_epoch = trainer.current_epoch
        if current_epoch == 0 or ((current_epoch + 1) % self.interval == 0):
            epoch_num = current_epoch + 1
            
            print(f"\nRunning post-processing at epoch {epoch_num}...")
            
            # Create directories for this epoch
            snapshot_dir = f"{self.output_dir}/epoch_{epoch_num}"
            Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
            Path(f"{snapshot_dir}/figures").mkdir(parents=True, exist_ok=True)
            
            # Try to load data if not already loaded
            data_loaded = self._load_data_if_needed()
            
            # Always plot training loss curves if available
            self._plot_train_val_loss(
                trainer, 
                output_path=f"{snapshot_dir}/figures/train_val_loss_epoch_{epoch_num}.png", 
                title_name=f"Loss curve - Epoch {epoch_num}"
            )
            
            # Only run inference if data files are loaded successfully
            if data_loaded and self.pyg_graph_dict_test is not None:
                try:
                    # Create sample inference
                    sample_inference_path = f"{snapshot_dir}/sample_inference_epoch_{epoch_num}.pt"
                    self._create_sample_inference(pl_module, sample_inference_path)
                    
                    # Store min, max, mean cases
                    sample_inference = torch.load(sample_inference_path)
                    min_max_mean_cases_path = f"{snapshot_dir}/min_max_mean_cases_epoch_{epoch_num}.pt"
                    self._store_cases(sample_inference, min_max_mean_cases_path)
                    
                    # Plot cases
                    min_max_mean_cases = torch.load(min_max_mean_cases_path)
                    self._plot_cases(
                        min_max_mean_cases, 
                        sample_inference, 
                        f"{snapshot_dir}/figures/cases_epoch_{epoch_num}.png"
                    )
                except Exception as e:
                    print(f"Error during post-processing: {e}")

    # Add the remaining methods with proper error handling
    # (methods like _plot_train_val_loss, _minmax_normalize_sample, etc.)
    
    def _plot_train_val_loss(self, trainer, output_path, title_name):
        """Plot training and validation loss curves"""
        if not hasattr(trainer, 'train_losses'):
            print("No loss history found in trainer. Skipping loss plot.")
            return
        
        # Extract losses from trainer
        epoch_np = np.arange(len(trainer.train_losses))
        train_loss_np = np.array(trainer.train_losses)
        val_loss_np = np.array(trainer.val_losses) if hasattr(trainer, 'val_losses') else None
        
        # Apply smoothing
        train_loss_smth = gaussian_filter1d(train_loss_np, sigma=self.sigma)
        val_loss_smth = gaussian_filter1d(val_loss_np, sigma=self.sigma) if val_loss_np is not None else None
        
        # Plot
        plt.figure()
        plt.plot(epoch_np, train_loss_smth, label=r"$\mathcal{L}_{\mathrm{train}}$")
        if val_loss_smth is not None:
            plt.plot(epoch_np, val_loss_smth, label=r"$\mathcal{L}_{\mathrm{val}}$")
        plt.xlabel('Epoch')
        plt.ylabel('MSE (norm) Loss')
        plt.legend(loc="best")
        plt.yscale('log')
        plt.grid(True, which='both')
        plt.title(f'{title_name}')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _minmax_normalize_sample(self, sample_graph_dict):
        """Normalize sample graph using min-max normalization"""
        # Convert string to list
        min_x = torch.tensor(ast.literal_eval(self.minmax_df['min_x[pos_x,pos_y,re]'].values[0]))
        max_x = torch.tensor(ast.literal_eval(self.minmax_df['max_x[pos_x,pos_y,re]'].values[0]))
        min_y = torch.tensor(ast.literal_eval(self.minmax_df['min_y[cp]'].values[0]))
        max_y = torch.tensor(ast.literal_eval(self.minmax_df['max_y[cp]'].values[0]))
        
        # Make copies of the dictionaries
        sample_graph_dict_copy = copy.deepcopy(sample_graph_dict)
        
        # Normalize feature vector (Data.x)
        sample_graph_dict_copy.x = (sample_graph_dict_copy.x - min_x) / (max_x - min_x)
        sample_graph_dict_copy.x = sample_graph_dict_copy.x.float()
        sample_graph_dict_copy.y = (sample_graph_dict_copy.y - min_y) / (max_y - min_y)
        sample_graph_dict_copy.y = sample_graph_dict_copy.y.float()
        
        return sample_graph_dict_copy, min_x.float().numpy(), max_x.float().numpy(), min_y.float().numpy(), max_y.float().numpy()
    
    def _create_sample_inference(self, model, output_path):
        """Create sample inference for test data"""
        sample_inference = {}
        for idx, row in self.test_data_df.iterrows():
            test_data_key = (row['u_inlet'], row['aoa'], row['naca_param'])
            test_graph_sample = self.pyg_graph_dict_test[test_data_key]
            test_graph_sample_norm, min_x, max_x, min_y, max_y = self._minmax_normalize_sample(test_graph_sample)
            
            # Data
            label_real = test_graph_sample.y[:, 0].numpy().flatten()
            
            model.eval()
            out_ = model.predict_step(test_graph_sample_norm, batch_idx=0)
            pred_real = (out_.detach().numpy().flatten() * (max_y - min_y)) + min_y
            
            # Metrics
            mse_ = mean_squared_error(label_real, pred_real)
            rmse_= np.sqrt(mse_)
            mae_ = mean_absolute_error(label_real, pred_real)
            mape_ = np.mean(np.abs((label_real - pred_real) / label_real)) * 100
            
            sample_inference[test_data_key] = {
                'input_real': test_graph_sample.x.numpy(),
                'label_real': label_real,
                'pred_real': pred_real.flatten(),
                'mse': mse_,
                'rmse': rmse_,
                'mae': mae_,
                'mape': mape_
            }
        
        torch.save(sample_inference, output_path)
    
    def _store_cases(self, sample_inference, output_path):
        """Store min, max, and mean cases based on MSE"""
        mse_list = [mean_squared_error(v['label_real'], v['pred_real']) for v in sample_inference.values()]
        mse_array = np.array(mse_list)
        min_idx = np.argmin(mse_array)
        max_idx = np.argmax(mse_array)
        mean_idx = np.argmin(np.abs(mse_array - np.mean(mse_array)))
        
        cases = {
            'min_case': list(sample_inference.keys())[min_idx],
            'max_case': list(sample_inference.keys())[max_idx],
            'mean_case': list(sample_inference.keys())[mean_idx]
        }
        
        torch.save(cases, output_path)
    
    def _plot_cases(self, cases, sample_inference, output_path):
        """Plot min, max, and mean cases"""
        fig, axs = plt.subplots(3, 2, figsize=(15, 18))
        
        for idx, key in enumerate(list(cases.keys())):
            case_key = cases[key]
            input_real = sample_inference[case_key]['input_real']
            label_real = sample_inference[case_key]['label_real']
            pred_real = sample_inference[case_key]['pred_real']
            rmse_ = sample_inference[case_key]['rmse']
            
            # Case data
            U_infty = float(case_key[0])
            aoa = float(case_key[1])
            re_ = U_infty/1.56e-5
            
            # Plot the airfoil profile
            axs[idx][0].plot(input_real[:,0], input_real[:,1], 'k-', label='XFOIL', linewidth=2)
            axs[idx][0].fill_between(input_real[:,0], input_real[:,1], color='skyblue', alpha=0.5)
            axs[idx][0].set_xlabel(r"$x/c$")
            axs[idx][0].set_ylabel(r"$y/c$")
            axs[idx][0].set_xlim(-0.5,1.5)
            axs[idx][0].set_ylim(-1.5,1.5)
            axs[idx][0].set_title(r"$Re$ = " + f"{re_:.2e}, " + r"$\alpha$ = " + f"{aoa:.1f} deg, ")
            axs[idx][0].grid('both')
            
            # Plot the cp profile
            rotated_airfoil_profile = np.array([input_real[:,0], input_real[:,1]]).T
            airfoil_profile = self._rotate_airfoil_coordinates(rotated_airfoil_profile, -aoa)
            axs[idx][1].plot(airfoil_profile[:,0], label_real, 'k-', label='XFOIL', linewidth=4)
            axs[idx][1].plot(airfoil_profile[:,0], pred_real, '-', label=f'RMSE:{rmse_:.2f}')
            axs[idx][1].set_xlabel(r"$x/c$")
            axs[idx][1].set_ylabel(r"$c_p$")
            axs[idx][1].legend(bbox_to_anchor=(1,1))
            axs[idx][1].set_title(f"{key}")
            axs[idx][1].grid(True, which='both')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def _rotate_airfoil_coordinates(self, airfoil_coordinates, angle_of_attack):
        """Rotate airfoil coordinates"""
        angle_of_attack = angle_of_attack * (np.pi/180)
        rotated_airfoil_coordinates = np.zeros((airfoil_coordinates.shape[0], airfoil_coordinates.shape[1]))
        rotated_airfoil_coordinates[:,0] = airfoil_coordinates[:,0] * np.cos(angle_of_attack) + airfoil_coordinates[:,1] * np.sin(angle_of_attack)
        rotated_airfoil_coordinates[:,1] = - airfoil_coordinates[:,0] * np.sin(angle_of_attack) + airfoil_coordinates[:,1] * np.cos(angle_of_attack)
        return rotated_airfoil_coordinates