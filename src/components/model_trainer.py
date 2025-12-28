

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier

# Metrics
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Utils
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object


plt.style.use('seaborn-v0_8-darkgrid')


@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    plots_dir = os.path.join("artifacts", "plots")
    comparison_graph = os.path.join("artifacts", "plots", "model_comparison.png")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainingConfig()
        os.makedirs(self.config.plots_dir, exist_ok=True)
        logging.info("ModelTrainer initialized")

    def generate_comparison_graph(self, results):
        
        try:
            logging.info("Generating comparison graph...")
            
            models = list(results.keys())
            test_accs = [results[m]['test_acc'] for m in models]
            f1_scores = [results[m]['test_f1'] for m in models]
            
            x = np.arange(len(models))
            width = 0.35

            fig, ax = plt.subplots(figsize=(11, 6))
            
            # Bars
            rects1 = ax.bar(x - width/2, test_accs, width, 
                          label='Test Accuracy', 
                          color='#3498db', 
                          edgecolor='black', 
                          linewidth=1.2)
            rects2 = ax.bar(x + width/2, f1_scores, width, 
                          label='F1 Score', 
                          color='#e74c3c', 
                          edgecolor='black', 
                          linewidth=1.2)

           
            ax.set_ylabel('Score', fontsize=13, fontweight='bold')
            ax.set_xlabel('Models', fontsize=13, fontweight='bold')
            ax.set_title('Model Performance Comparison\nEmployee Attrition Prediction', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(models, fontsize=11, fontweight='bold')
            ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
            ax.set_ylim(0.83, 1.0)  
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.3f}',
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 4),
                              textcoords="offset points",
                              ha='center', va='bottom', 
                              fontsize=9, fontweight='bold')

            autolabel(rects1)
            autolabel(rects2)

            plt.tight_layout()
            plt.savefig(self.config.comparison_graph, dpi=300, bbox_inches='tight')
            plt.close()
            
           
            print(" (Results)")
            
            logging.info("Comparison graph ")

        except Exception as e:
            logging.warning(f"Graph generation failed: {e}")

    def train_model(self, X_train, y_train, X_test, y_test, model_name, model, params):
        """Train individual model with full metrics"""
        try:
            logging.info(f"Training {model_name}...")
            
            # === TRAINING ===
            if model_name == "TabPFN_Improved":
                # Calibrated ensemble (no tuning)
                base_model = TabPFNClassifier(device="cpu")
                best_model = CalibratedClassifierCV(
                    base_model, 
                    method='sigmoid', 
                    cv=3, 
                    ensemble=True
                )
                best_model.fit(X_train, y_train)
                
            else:
                # GridSearchCV for others
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid = GridSearchCV(
                    model, 
                    params, 
                    cv=cv, 
                    scoring="f1_macro", 
                    n_jobs=-1, 
                    verbose=0
                )
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_

           
            y_test_pred = best_model.predict(X_test)
            
            # Basic metrics
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred, average="macro")
            
            # AUC-ROC
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            else:
                auc = 0.0

            logging.info(f"{model_name} - Acc: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {auc:.4f}")

            return {
                'model': best_model,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'auc_roc': auc
            }

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_array, test_array):
        """Main training pipeline"""
        try:
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

           
            models = {
                "TabPFN_Improved": None,
                
                "XGBoost": XGBClassifier(
                    random_state=42, 
                    eval_metric="logloss",
                    use_label_encoder=False
                ),
                
                
                "RandomForest": RandomForestClassifier(
                    random_state=42,
                    class_weight='balanced' 
                ),
                
                "LogisticRegression": LogisticRegression(
                    random_state=42, 
                    max_iter=1000
                )
            }

           
            
            params = {
                "TabPFN_Improved": {},  
                "XGBoost": {
                    "n_estimators": [50],      
                    "max_depth": [3],          
                    "learning_rate": [0.1]     
                },
                
                # RandomForest: Basic configuration
                "RandomForest": {
                    "n_estimators": [100],     
                    "max_depth": [None],       
                    "min_samples_split": [2]   
                },
                
               
                "LogisticRegression": {
                    "C": [1.0],               
                    "solver": ['lbfgs']
                }
            }

           
            results = {}
            fitted_models = {}

           
            print(f"{'Model':<20} | {'Test Acc':<12} | {'F1 Score':<12} | {'AUC-ROC':<10} | Status")
            print("-"*90)

            for name, model in models.items():
                res = self.train_model(
                    X_train, y_train, X_test, y_test, 
                    name, model, params[name]
                )

                # Status
                if name == "TabPFN_Improved":
                    status = " CHAMPION"
                elif res['test_f1'] > 0.90:
                    status = " EXCELLENT"
                elif res['test_f1'] > 0.85:
                    status = " GOOD"
                else:
                    status = " MODERATE"

                print(
                    f"{name:<20} | "
                    f"{res['test_acc']:.4f}      | "
                    f"{res['test_f1']:.4f}      | "
                    f"{res['auc_roc']:.4f}     | {status}"
                )

                results[name] = res
                fitted_models[name] = res['model']

            print("="*90)

          
            self.generate_comparison_graph(results)

            
            best_name = max(results, key=lambda x: results[x]['test_f1'])
            best_model = fitted_models[best_name]
            best_res = results[best_name]

            print("\n" + "="*90)
            print(f" BEST PERFORM MODEL: {best_name}")
            print("="*90)
            print(f"   Test Accuracy:  {best_res['test_acc']:.4f} ({best_res['test_acc']*100:.2f}%)")
            print(f"   F1 Score:       {best_res['test_f1']:.4f}")
            print(f"   AUC-ROC:        {best_res['auc_roc']:.4f}")
            print("="*90)

           
            save_object(self.config.trained_model_file_path, best_model)
            logging.info(f"Best model saved: {best_name}")
            
            print(f"\n💾 Model saved: {self.config.trained_model_file_path}")
            
            print("="*90)

            return {
                'best_model_name': best_name,
                'best_model': best_model,
                'all_results': results
            }

        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    print("\n" + "="*90)
    print(" THESIS MODEL TRAINING PIPELINE")
    print("="*90)

    print("\n[1/3] Loading Data...")
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    print("\n[2/3] Transforming Data...")
    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(
        train_path, test_path
    )

    print("\n[3/3] Training Models...")
    trainer = ModelTrainer()
    results = trainer.initiate_model_trainer(train_arr, test_arr)

    print("\n" + "="*90)
    print(" TRAINING COMPLETED!")
    print("="*90)
    print("\nNext Steps:")
    print("  → Run monitoring: python -m src.components.model_monitoring")
    print("="*90)