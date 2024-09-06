import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.utils import resample
from sksurv.metrics import (
    concordance_index_censored,
    brier_score,
    integrated_brier_score, 
    cumulative_dynamic_auc
)
from sksurv.compare import compare_survival
from sksurv.nonparametric import kaplan_meier_estimator

def time_points(y, ll = 10, ul = 90, step = 1):    
    lower, upper = np.percentile(y["time"], [ll, ul])
    times = np.arange(lower, upper + 1, step, dtype=int)
    return times

def make_random_list(n_samples = 1000, seed = None):
    rng = np.random.default_rng(seed)
    random_list = list(rng.choice(np.arange(n_samples*10), size = n_samples, replace = False))
    return random_list

### metrics

def plot_c_index_ci(estimator_list, X_test_list, y, name_list, n_samples = 1000, random_list = None, ax = None):
    bin_center = np.arange(len(name_list))
    name_list_ = []
    for i, (estimator, X_test, name) in enumerate(zip(estimator_list, X_test_list, name_list)):
        prediction = estimator.predict(X_test)
        c_index_list = []        
        for j in range(n_samples):
            indices = resample(np.arange(len(prediction)), random_state = random_list[j])
            c_index = concordance_index_censored(y["event"][indices], y["time"][indices], prediction[indices])
            c_index_list.append(c_index[0])
        mean_list = np.mean(c_index_list)
        ci_list = np.percentile(c_index_list, [2.5, 97.5])
        print(f"c-index, 95% CI: {mean_list:.3f}, {ci_list[0]:.3f}-{ci_list[1]:.3f}")
        name_list_.append(f'{name}\n{mean_list:.3f}\n{ci_list[0]:.3f}-{ci_list[1]:.3f}')
        plt.bar(bin_center[i], 
                mean_list, 
                width=0.8, 
                align='center', 
                alpha=0.5,
                color=f"C{i}")        
        plt.errorbar(bin_center[i], 
                     mean_list, 
                     yerr=[[mean_list - ci_list[0]], [ci_list[1] - mean_list]], 
                     color=f"C{i}")    
    plt.ylim(0.45, 1.05)
    plt.xticks(bin_center, name_list_)
    plt.ylabel('C-index with 95% CI')
    if ax is None:
        ax = plt.gca()
    return ax
   
    
def plot_td_auc_no_ci(estimator_list, X_test_list, name_list, 
                      y_test, y_train, times, mean = False, xticks = False, ax = None):      
    for i, (estimator, X_test, name) in enumerate(zip(estimator_list, X_test_list, name_list)):
        risk_scores = estimator.predict(X_test)
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk_scores, times)
        plt.plot(times, auc, marker="", label=f"{name} \nmean AUC = {mean_auc:.3f}", color=f"C{i}")
        if mean == True:
            plt.axhline(mean_auc, linestyle="--", color=f"C{i}")
    plt.axhline(0.5, linestyle="--", color='gray', label='Chance level\nmean AUC = 0.500')
    if xticks == True:
        plt.xticks(times)
    else:
        plt.xticks(np.arange(0,np.max(times),30))
    plt.ylim(0.45, 1.05)
    plt.xlabel("Time (months)")
    plt.ylabel("Time-dependent AUC")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    if ax is None:
        ax = plt.gca()
    return ax

def plot_td_bs_no_ci(estimator_list, X_test_list, name_list, 
                     y_test, y_train, times, ibs = False, xticks = False, ax = None):       
    for i, (estimator, X_test, name) in enumerate(zip(estimator_list, X_test_list, name_list)):
        survs = estimator.predict_survival_function(X_test)
        scores = []
        for time in times:
            preds = [fn(time) for fn in survs]
            _, score = brier_score(y_train, y_test, preds, time)
            scores.append(score[0])

        # IBS
        surv_prob = np.row_stack([fn(times) for fn in estimator.predict_survival_function(X_test)])
        IBS = integrated_brier_score(y_train, y_test, surv_prob, times)
        
        scores_ = np.array(scores)    
        plt.plot(times, scores_, marker="", label=f"{name} \nIBS = {IBS:.3f}", color=f"C{i}")
        if ibs == True:
            plt.axhline(IBS, linestyle="--", color=f"C{i}")
    plt.axhline(0.25, linestyle="--", color='gray', label='Chance level\nIBS = 0.250')
    if xticks == True:
        plt.xticks(times)
    else:
        plt.xticks(np.arange(0,np.max(times),30))    
    plt.ylim(-0.025, 0.275)
    plt.xlabel("Time (months)")
    plt.ylabel("Time-dependent BS")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    if ax is None:
        ax = plt.gca()
    return ax
    
    
    
### KM

def get_best_cutoff(estimator, X_, y_):
    rs = estimator.predict(X_)
    pvals = [[],[]]
    for j in np.arange(25,76,1):
        q_rs = np.percentile(rs, j)
        high_risk_indices = []
        for i in range(len(rs)):
            if rs[i] > q_rs:
                high_risk_indices.append(True)
            else:
                high_risk_indices.append(False)
        high_risk_indices = np.array(high_risk_indices)
        stats = compare_survival(y_, high_risk_indices, return_stats = True)
        pvals[0].append(j)
        pvals[1].append(stats[1])
    pvals_ = pd.DataFrame(pvals).transpose()
    pvals_.columns = ['percentile', 'p-value']
    display(pvals_.sort_values(by = 'p-value', ascending = True).head(10))

def plot_km(estimator, X_, y_, percentile = 50, plot_ci = False):
    rs = estimator.predict(X_)
    q_rs = np.percentile(rs, percentile)

    high_risk_indices = []
    for i in range(len(rs)):
        if rs[i] > q_rs:
            high_risk_indices.append(True)
        else:
            high_risk_indices.append(False)
    high_risk_indices = np.array(high_risk_indices)

    stats = compare_survival(y_, high_risk_indices, return_stats = True)
    pval = stats[1]

    time_l, survival_prob_l, conf_int_l = kaplan_meier_estimator(
    y_["event"][~high_risk_indices], y_["time"][~high_risk_indices], conf_type="log-log"
    )
    plt.step(time_l, survival_prob_l, where="post", label=f"Low risk")
    if plot_ci == True:
        plt.fill_between(time_l, conf_int_l[0], conf_int_l[1], alpha=0.25, step="post")

    time_h, survival_prob_h, conf_int_h = kaplan_meier_estimator(
    y_["event"][high_risk_indices], y_["time"][high_risk_indices], conf_type="log-log"
    )
    plt.step(time_h, survival_prob_h, where="post", label=f"High risk")
    if plot_ci == True:
        plt.fill_between(time_h, conf_int_h[0], conf_int_h[1], alpha=0.25, step="post")

    plt.ylim(-0.05, 1.05)
    plt.xlim(-6, np.max(time_points(y_, 5, 95, 1)) + 30) # 0-95% time points
    plt.xticks(np.arange(0, np.max(time_points(y_, 5, 95, 1)) + 30, 60))
    plt.xlabel("time $t$ (months)" )
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    if pval < 0.0001:
        plt.annotate(f'p-value < 0.0001', xy = (0, 0.18), xytext = (0, 0.18))
    else:
        plt.annotate(f'p-value = {pval:.4f}', xy = (0, 0.18), xytext = (0, 0.18))
    plt.legend(loc="lower left")    

def plot_km_together(estimator_list, X_list, y, name_list, percentiles):
    for estimator, X, name, percentile in zip(estimator_list, X_list, name_list, percentiles):
        plot_km(estimator, X, y, percentile, plot_ci = True)
        plt.title(name)    
        plt.savefig(f'KM {name}.pdf', format='pdf', bbox_inches = 'tight')
        plt.show()

def plot_km_original(y_, name, show = False, ci = False):
    time, survival_prob, conf_int = kaplan_meier_estimator(
        y_["event"], y_["time"], conf_type="log-log"
    )
    plt.step(time, survival_prob, where="post", label=name)
    if ci == True:
        plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
    plt.ylim(-0.05, 1.05)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xticks(np.arange(0,np.max(time)+30,60))
    plt.xlabel("time $t$ (months)")
    if show == True:
        plt.show()
    return np.vstack((time, survival_prob)).T

### prediction KM

def plot_predictions(estimator, X, times, best_cop, show = False):
    rss = estimator.predict(X)
    pred_surv = estimator.predict_survival_function(X)
    for i, surv_func in enumerate(pred_surv):
        if rss[i] > best_cop:
            plt.step(times, surv_func(times), where="post", color='#ff7f0e', alpha = 0.1)
        else:
            plt.step(times, surv_func(times), where="post", color='#1f77b4', alpha = 0.1)
    plt.text(0, 0.175, 'High-risk group', color='#ff7f0e', weight='bold')
    plt.text(0, 0.125, 'Low-risk group', color='#1f77b4', weight='bold')
    plt.xticks(np.arange(0,np.max(times)+30,60))
    plt.ylim(-0.05, 1.05)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$ (months)")
    if show == True:
        plt.show()

### personalized

def plot_personalized_predictions(estimator, X, times, best_cop, show = False):
    rs = estimator.predict(X)
    if rs > best_cop:
        color_ = '#ff7f0e'
        plt.text(0, 0.175, 'High-risk group', color=color_, weight='bold')
    else:
        color_ = '#1f77b4'
        plt.text(0, 0.175, 'Low-risk group', color=color_, weight='bold')       
    pred_surv = estimator.predict_survival_function(X)
    for surv_func in pred_surv:    
        plt.step(times, surv_func(times), where="post", color=color_)
    plt.xticks(np.arange(0,np.max(times)+30,60))
    plt.ylim(-0.05, 1.05)
    plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
    plt.xlabel("time $t$ (months)")
    plt.text(0, 0, f'Survival probability\n5-year: {surv_func(60):.1%}\n10-year: {surv_func(120):.1%}')
    if show == True:
        plt.show()
        
### shap   
    
def plot_beeswarm_per_features(explanation, name):
    mask = [name in n for n in explanation.feature_names]
    explanation_ = shap.Explanation(explanation.values[:, mask],
                                    feature_names=list(np.array(explanation.feature_names)[mask]),
                                    data=explanation.data[:, mask],
                                    base_values=explanation.base_values,
                                    display_data=explanation.display_data,
                                    instance_names=explanation.instance_names,
                                    output_names=explanation.output_names,
                                    output_indexes=explanation.output_indexes,
                                    lower_bounds=explanation.lower_bounds,
                                    upper_bounds=explanation.upper_bounds,
                                    main_effects=explanation.main_effects,
                                    hierarchical_values=explanation.hierarchical_values,
                                    clustering=explanation.clustering,
    )
    shap.plots.beeswarm(explanation_)

def get_ylabels(explanation_patient, X_patient): 
    Surgery_PR = 'Yes' if X_patient['Surgery'].values == 'PR' else 'No'
    Surgery_USO = 'Yes' if X_patient['Surgery'].values == 'USO' else 'No'
    Radiotherapy_RAI = {'RAI': 'Yes', 'EBRT': 'No', 'No/Unknown': 'No/Unknown'}[X_patient['Radiotherapy'].values[0]]
    Radiotherapy_EBRT = {'EBRT': 'Yes', 'RAI': 'No', 'No/Unknown': 'No/Unknown'}[X_patient['Radiotherapy'].values[0]]
    Grade = 'Differentiated' if X_patient['Grade'].values == 'G1' else 'PD/UD'
    ylabels = [
        str(X_patient['Age'].values[0]) + ' years' + ' = ' + 'Age',
        str(X_patient['Extent'].values[0]) + ' = ' + 'Extent', 
        str(X_patient['N category'].values[0]) + ' = ' + 'N category', 
        str(X_patient['Hysterectomy'].values[0]) + ' = ' + 'Hysterectomy', 
        str(Surgery_PR) + ' = ' + 'Surgery_PR', 
        str(X_patient['Chemotherapy'].values[0]) + ' = ' + 'Chemotherapy', 
        str(X_patient['M category'].values[0]) + ' = ' + 'M category', 
        str(Radiotherapy_RAI) + ' = ' + 'Radiotherapy_RAI', 
        str(Surgery_USO) + ' = ' + 'Surgery_USO', 
        str(X_patient['Tumor size'].values[0].astype(int)) + ' mm' + ' = ' + 'Tumor size', 
        str(Radiotherapy_EBRT) + ' = ' + 'Radiotherapy_EBRT', 
        str(Grade) + ' = ' + 'Grade', 
        str(X_patient['AJCC stage'].values[0]) + ' = ' + 'AJCC stage'
    ]
    combine_list = list(zip(
        np.abs(explanation_patient.values),
        explanation_patient.feature_names, 
        ylabels))
    sorted_lists = sorted(combine_list, key = lambda x: x[0], reverse = False)
    sorted_ylabels = [item[2] for item in sorted_lists]
    sorted_ylabels_names = [item[1] for item in sorted_lists]
    
    return sorted_ylabels, sorted_ylabels_names