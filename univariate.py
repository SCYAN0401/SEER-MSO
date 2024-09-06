import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sksurv.metrics import cumulative_dynamic_auc


def get_univariate_table(estimator_list, X_train, name_list, y_train, cv = None):
    table=[]
    annot_table=[]
    for estimator in estimator_list:
        list = []
        annot_list = []
        for feature in X_train.columns:
            c_index_scores = cross_val_score(estimator,
                                             X_train[[feature]],
                                             y_train, 
                                             cv = cv)
            mean_c_index = np.mean(c_index_scores)
            std_c_index = np.std(c_index_scores)
            list.append(mean_c_index)
            annot_list.append(f'{mean_c_index:.2f} ± {std_c_index:.2f}')    
        table.append(list)
        annot_table.append(annot_list)
    table = pd.DataFrame(table, index = name_list, columns = X_train.columns)
    annot_table = pd.DataFrame(annot_table)
    return table, annot_table
        
def plot_cumulative_dynamic_auc(estimator, X_train, y_train, times, cv = None, feature = None, color = None, mean = None):
    aucs = []
    mean_aucs = []
    for train_index, val_index in cv.split(X_train[[feature]]):
        X_train_ = np.array(X_train[feature]).reshape(-1,1)
        estimator.fit(X_train_[train_index], y_train[train_index])
        risk_scores = estimator.predict(X_train_[val_index])
        auc, mean_auc = cumulative_dynamic_auc(y_train[train_index], y_train[val_index], risk_scores, times)
        aucs.append(auc)
        mean_aucs.append(mean_auc)
    mean_mean_auc = np.mean(mean_aucs)
    std_mean_auc = np.std(mean_aucs)
    
    auc_plot = np.mean(aucs, axis = 0)
    
    plt.plot(times, auc_plot, marker="", color=color,
             label=f'{feature} (mean AUC, m ± sd: {mean_mean_auc:.3f} ± {std_mean_auc:.3f})')
    print(f"{feature} \n {mean_mean_auc:.3f} ± {std_mean_auc:.3f}")
    if mean == True:
        plt.axhline(mean_mean_auc, linestyle="--", color=color)

def plot_cumulative_dynamic_auc_together(estimator, X_train, y_train, times, cv = None, mean = False):
    print(f'{estimator} {np.min(times)} - {np.max(times)} months, mAUC, m ± sd:')    
    plt.figure(figsize=(10, 10))
    for i, col in enumerate(X_train.columns):
        plot_cumulative_dynamic_auc(estimator, X_train, y_train, times, cv = cv, feature = col, color=f"C{i}")
    plt.title(f'{estimator.__class__.__name__}')
    plt.xlabel("Time (months)")
    plt.ylabel("Time-dependent AUC")
    plt.legend(loc="lower left", bbox_to_anchor=(1, 0.5))
    plt.show()