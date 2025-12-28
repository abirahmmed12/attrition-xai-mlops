

import os
import sys
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_curve,
    roc_auc_score, confusion_matrix, auc, RocCurveDisplay,
    classification_report, matthews_corrcoef, cohen_kappa_score, log_loss
)
import pickle
import warnings
warnings.filterwarnings('ignore')


plt.style.use('ggplot')

from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelMonitoringConfig:
    monitoring_dir = os.path.join("artifacts", "monitoring")
    plots_dir = os.path.join("artifacts", "monitoring", "plots")
    reports_dir = os.path.join("artifacts", "monitoring", "reports")
    excel_path = os.path.join("artifacts", "monitoring", "reports", "cv_results.csv")

class ModelMonitor:
    def __init__(self, model, X_train, y_train, X_test, y_test, model_name="TabPFN_Improved"):
        self.model = model
       
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.model_name = model_name
        self.config = ModelMonitoringConfig()
        
       
        self._clean_artifacts()
        
        os.makedirs(self.config.monitoring_dir, exist_ok=True)
        os.makedirs(self.config.plots_dir, exist_ok=True)
        os.makedirs(self.config.reports_dir, exist_ok=True)
        
        logging.info(f"Model Monitor initialized for {model_name}")

    def _clean_artifacts(self):
        if os.path.exists(self.config.plots_dir):
            shutil.rmtree(self.config.plots_dir)
        if os.path.exists(self.config.reports_dir):
            shutil.rmtree(self.config.reports_dir)

    def run_full_monitoring(self):
        try:
            print("\n" + "="*80)
            print(f" THESIS DEFENSE MONITORING: {self.model_name}")
            print("="*80)
            
            results = {}
            
            
            print("\n MODULE 1: 5-FOLD STABILITY CHECK")
            print("(Checking if model performance is consistent across folds...)")
            cv_results = self.run_advanced_cv()
            results['cv'] = cv_results
            
            
            print("\n MODULE 2: THRESHOLD OPTIMIZATION")
            self.plot_precision_recall_vs_threshold()
            
           
            print("\n MODULE 3: FINAL TEST SET EVALUATION")
            print("(Calculating MCC, Kappa, and Log Loss...)")
            test_results = self.evaluate_test_set()
            results['test'] = test_results
            
            
            print("\n MODULE 4: GENERATING FINAL REPORT")
            self.generate_report(results)
            
            print("\n" + "="*80)
            print(" MONITORING COMPLETED!")
            print(f" Check Graphs: {self.config.plots_dir}")
            print(f" Check Report: {self.config.reports_dir}")
            print("="*80)
            
        except Exception as e:
            raise CustomException(e, sys)

    def run_advanced_cv(self):
        """
        Performs Stratified K-Fold Cross-Validation and logs metrics.
        """
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)
            fold_metrics = []
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            print(f"{'Fold':<6} | {'Accuracy':<10} | {'F1 Score':<10}")
            print("-" * 35)
            
            for i, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
                X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
                y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
                
                self.model.fit(X_fold_train, y_fold_train)
                y_pred = self.model.predict(X_fold_val)
                y_proba = self.model.predict_proba(X_fold_val)[:, 1]
                
                acc = accuracy_score(y_fold_val, y_pred)
                f1 = f1_score(y_fold_val, y_pred, average='macro')
                roc_auc = roc_auc_score(y_fold_val, y_proba)
                
                fold_metrics.append({'Fold': f'Fold {i+1}', 'Accuracy': acc, 'F1_Score': f1})
                print(f"Fold {i+1:<1} | {acc:.4f}     | {f1:.4f}")
                
                
                viz = RocCurveDisplay.from_estimator(
                    self.model, X_fold_val, y_fold_val,
                    name=f"Fold {i+1}", alpha=0.3, lw=1, ax=ax
                )
                interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(viz.roc_auc)

           
            ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            
            
            ax.plot(mean_fpr, mean_tpr, color="b",
                    label=f"Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})", 
                    lw=2, alpha=0.8)
            
            ax.set(title=f"Stability Analysis (5-Fold CV)\n{self.model_name}")
            ax.legend(loc="lower right")
            
            path = os.path.join(self.config.plots_dir, "stability_roc_curve.png")
            plt.savefig(path, dpi=300)
            plt.close()
            
           
            pd.DataFrame(fold_metrics).to_csv(self.config.excel_path, index=False)
            
            
            return pd.DataFrame(fold_metrics).mean(numeric_only=True)

        except Exception as e:
            raise CustomException(e, sys)

    def plot_precision_recall_vs_threshold(self):
        try:
            y_scores = self.model.predict_proba(self.X_test)[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_scores)
            
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
            plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
            plt.plot(thresholds, f1_scores[:-1], "r-", label="F1 Score")
            
            plt.axvline(best_threshold, color='k', linestyle=':', label=f'Optimal: {best_threshold:.2f}')
            plt.title(f"Threshold Optimization\nOptimal Cutoff: {best_threshold:.2f}")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend()
            
            path = os.path.join(self.config.plots_dir, "threshold_analysis.png")
            plt.savefig(path, dpi=300)
            plt.close()

        except Exception as e:
            logging.warning(f"Threshold plot failed: {e}")

    def evaluate_test_set(self):
        try:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[:, 1]
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 14})
            plt.title('Confusion Matrix (Test Set)')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.savefig(os.path.join(self.config.plots_dir, "confusion_matrix.png"), dpi=300)
            plt.close()
            
            # Metrics
            report = classification_report(self.y_test, y_pred, output_dict=True)
            mcc = matthews_corrcoef(self.y_test, y_pred)
            kappa = cohen_kappa_score(self.y_test, y_pred)
            logloss = log_loss(self.y_test, y_proba)
            
            print(f"   ➤ MCC:       {mcc:.4f} (Quality of binary classification)")
            print(f"   ➤ Kappa:     {kappa:.4f} (Reliability)")
            print(f"   ➤ Log Loss:  {logloss:.4f} (Confidence)")
            
            return {'report': report, 'mcc': mcc, 'kappa': kappa, 'log_loss': logloss}

        except Exception as e:
            raise CustomException(e, sys)

    def generate_report(self, results):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.config.reports_dir, f'FINAL_THESIS_REPORT_{timestamp}.txt')
            
            with open(path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("  MODEL DEEP DIVE\n")
                f.write("="*60 + "\n\n")
                
                f.write("1. STABILITY ANALYSIS (5-Fold CV)\n")
                f.write("-" * 40 + "\n")
                f.write(f"Mean Accuracy: {results['cv']['Accuracy']:.4f}\n")
                f.write(f"Mean F1 Score: {results['cv']['F1_Score']:.4f}\n\n")
                
                f.write("2. OTHER METRICS (Test Set)\n")
                f.write("-" * 40 + "\n")
                f.write(f"Accuracy:      {results['test']['report']['accuracy']:.4f}\n")
                f.write(f"Macro F1:      {results['test']['report']['macro avg']['f1-score']:.4f}\n")
                f.write(f"MCC Score:     {results['test']['mcc']:.4f}\n")
                f.write(f"Kappa Score:   {results['test']['kappa']:.4f}\n")
                f.write(f"Log Loss:      {results['test']['log_loss']:.4f}\n")
                
           

        except Exception as e:
            logging.warning(f"Report generation failed: {e}")

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation
    
    print("\n STARTING MONITORING PIPELINE...")
    
    # Load Data
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    transformer = DataTransformation()
    train_arr, test_arr, _ = transformer.initiate_data_transformation(train_path, test_path)
    
    X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
    X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
    
   
    model_path = os.path.join("artifacts", "model.pkl")
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Run Monitoring
        monitor = ModelMonitor(model, X_train, y_train, X_test, y_test, "TabPFN_Improved")
        monitor.run_full_monitoring()
    else:
        print("  'artifacts/model.pkl' not found .")