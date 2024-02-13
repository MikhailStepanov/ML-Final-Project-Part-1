#Mikhail Stepanov. IDA Prüfung. Matrikelnummer 811991

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

from sklearn.metrics import accuracy_score, mean_squared_error, roc_curve, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing(df, type1):
    features  = df.drop(['Creditworthy'], axis = 1)
    features.replace(to_replace = '?', value = np.nan, inplace = True)

    if type1:
        one_hot_columns = ['Status_of_EA', 'Credit_hist', 'Purpose', 'Savings_account', 'Personal_status', 
                               'Other_debtors', 'Property', 'Other_IP', 'Housing', 'Job', 'Telephone', 'Foreign_worker']

        #One-hot encoding of all categorical columns except Present_ES (Present employment since)
        features_oneHot  = pd.get_dummies(features, dummy_na=True, columns=one_hot_columns)

        #Drop all constant columns
        features_oneHot  = features_oneHot.loc[:, (features_oneHot != features_oneHot.iloc[0]).any()]

        return features_oneHot 

    else:
        categorical_columns = ['Credit_hist', 'Purpose', 'Personal_status', 'Other_debtors', 'Property', 'Other_IP', 
                               'Housing', 'Job', 'Telephone', 'Foreign_worker']

        #One-hot encoding of all categorical columns except Present_ES (Present employment since)
        features_oneHot  = pd.get_dummies(features, dummy_na=True, columns=categorical_columns)

        #Drop all constant columns
        features_oneHot  = features_oneHot.loc[:, (features_oneHot != features_oneHot.iloc[0]).any()]

        #Here I use a numerical transformation for the two numerical continuous features in our data 
        #(Status of existing checking account and Savings account/bonds)
        
        #Status of existing checking account
        checkAcc = [-100, 100, 300, 0]

        #Savings account/bonds
        savingAccount = [50, 250, 750, 1250, 0]

        for i in range(4):
            features_oneHot.Status_of_EA.replace(to_replace = 'A1'+str(i+1), value = checkAcc[i], inplace=True)
        
        for j in range(5):
            features_oneHot.Savings_account.replace(to_replace = 'A6'+str(j+1), value = savingAccount[i], inplace=True)

        return features_oneHot 

def decisionTreeReg(X_train, X_test, y_train, y_test):
    
    #DescisionTreeRegressor with MSE splitting criterion
    reg = DecisionTreeRegressor(criterion = 'mse')
    #I vary min_samples_split and max_leaf_nodes parameters of the tree and search 
    #for the best estimator optimized by a cross validation
    params = {'min_samples_split' : range(5, 50), 'max_leaf_nodes' : range(5, 15)}

    grid_search_cv = GridSearchCV(reg, params, cv=5)
    grid_search_cv.fit(X_train, y_train)

    best_est   = grid_search_cv.best_estimator_
    print("Best parameters of DecisionTreeRegressor", grid_search_cv.best_params_)
    y_pred = best_est.predict(X_test)
    err    = mean_squared_error(y_test, y_pred)
    
    return best_est, err 

def linReg(X_train, X_test, y_train, y_test):

    reg         = LinearRegression().fit(X_train, y_train)
    predictions = reg.predict(X_test)
    err = mean_squared_error(y_test, predictions)

    return reg, err

def feature_importance(model, X_train, y_train, n_repeats):
    #Use permutation importance with 10 repeats on all features of the train dataset  
    perm_import = permutation_importance(model, X_train, y_train, n_repeats=n_repeats, random_state=0)
    #Find mean permutation importance of 10 repeats
    perm_mean   = perm_import.importances_mean

    return perm_mean

def predictPresentES(preprocessed):
    expYears = [0, 0.5, 2.5, 5.5, 9]

    #Numerical representation of "Present employment since" feature that I am going to predict
    for i in range(5):
        preprocessed.Present_ES.replace(to_replace = 'A7'+str(i+1), value = expYears[i], inplace=True)

    #Split data set on a dataset where are no missing values in present employment since feature and on a data set,
    #where all Prese_ES features are missed 
    withPES          = preprocessed.loc[preprocessed['Present_ES'].notnull()]
    missingPES       = preprocessed.loc[preprocessed['Present_ES'].isnull()]

    #X - all samples with present values of Present_ES without Present_ES column, that will be used as labels
    X      = withPES.drop(['Present_ES'], axis = 1)  
    #data - features with missing values of Present_ES, that will be used for predictions  
    data   = missingPES.drop(['Present_ES'], axis = 1)

    #labels of Present_ES
    y = withPES.Present_ES

    #split X and y on train and test data set, to validate the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Try with decision Tree regressor
    best_tree, tree_mse = decisionTreeReg(X_train, X_test, y_train, y_test) 
    print("MSE DecisionTreeRegressor", tree_mse)

    #plot best decision tree
    fig, ax = plt.subplots(figsize=(19.2,10.8))
    plot_tree(best_tree, ax=ax, class_names=[str(i) for i in range(5)], feature_names=X_train.columns) 
    plt.show()

    #Try linear regression
    linearReg, mse_LinReg = linReg(X_train, X_test, y_train, y_test)
    print("MSE LinearRegression with all features", mse_LinReg)

    #Use permutation importance for linear regression model
    perm_mean = feature_importance(linearReg, X_train, y_train, 10)

    plt.figure(figsize=(12,8))
    plt.title('Feature importances', fontsize=20)
    perm_mean = np.array(perm_mean)
    idxSort = np.argsort(perm_mean)
    plt.plot(X_train.columns[np.flip(idxSort)[:10]], perm_mean[np.flip(idxSort)[:10]], 'bx')
    plt.xticks(X_train.columns[np.flip(idxSort)[:10]], X_train.columns[np.flip(idxSort)[:10]], fontsize=14, rotation='vertical', )
    plt.yticks(fontsize=14)
    plt.ylabel('Feature importances Preprocessing', fontsize=16)
    plt.show()

    mse_differentFI  = []
    models_differentFI = [] 
    columns_differentFI = []

    #Try to drop some features and find a best set of most important features
    #and make predictions only with theses set of features  
    for i in perm_mean[idxSort]:
  
        data_bestPermImp = X_train.iloc[:, np.where(perm_mean >= i)[0]]
        maxFI = data_bestPermImp.columns

        X_train_maxFI = X_train[maxFI]
        X_test_maxFI  = X_test[maxFI]

        LinReg_bestFI, mse_LinReg_bestFI = linReg(X_train_maxFI, X_test_maxFI, y_train, y_test)
        mse_differentFI.append(mse_LinReg_bestFI)
        models_differentFI.append(LinReg_bestFI)
        columns_differentFI.append(maxFI) 

    LinReg_bestFI = models_differentFI[np.argmin(np.array(mse_differentFI))] 
    mse_LinReg_bestFI = min(mse_differentFI)
    print("MSE of Linear Regression model with highest feature importances", mse_LinReg_bestFI)

    maxFI = columns_differentFI[np.argmin(np.array(mse_differentFI))]

    print("Shape of X_train with all features", X_train.shape)
    print("Shape of X_train with best features only", data[maxFI].shape) 

    minMSE = min([tree_mse, mse_LinReg, mse_LinReg_bestFI])
    

    if minMSE == tree_mse:
        print("Desicison Tree was used")
        predictionsPES = best_tree.predict(data)                
        preprocessed.loc[preprocessed.Present_ES.isnull(), 'Present_ES'] = predictionsPES 
        return preprocessed

    elif minMSE == mse_LinReg:
        print("Linear Regression with all features was used")
        predictionsPES = linearReg.predict(data)
        preprocessed.loc[preprocessed.Present_ES.isnull(), 'Present_ES'] = predictionsPES
        return preprocessed

    else:
        print("Linear Regression with most important features was used")
        data      = data[maxFI]
        predictionsPES  = LinReg_bestFI.predict(data)
        preprocessed.loc[preprocessed.Present_ES.isnull(), 'Present_ES'] = predictionsPES
        return preprocessed

def errorsType(y_test, predictions):
    #Array of different thresholds  
    thresholds = np.arange(0.1, 1, 0.01)

    fp = []
    fn = []

    for i in thresholds:
        #Predict class 1 if the predicted probability of class 1 is higher than threshold i
        predictOne  = predictions > i
        #Predict class 0 if the predicted probability of class 1 is less or equal threshold i
        predictZero = predictions <= i

        #Count number of False positives 
        fp.append(np.sum(predictOne & (y_test == 0)))
        #Count number of False negatives
        fn.append(np.sum(predictZero & (y_test == 1)))
    
    return thresholds, fp, fn

def best_threshold(fp, fn, thresholds): 

    #Cost function where false positives have 5 times higher importnace than false negatives
    costSensetiveLost = 5*np.array(fp) + np.array(fn)
    plt.figure(figsize=(12,8))
    plt.title('Best costSensetive threshold ' + str(round(thresholds[np.argmin(costSensetiveLost)], 2)), fontsize=20)
    plt.plot(np.array(thresholds), costSensetiveLost)
    plt.xlabel('Thresholds', fontsize=18)
    plt.ylabel('Weighted cost of FPs and FNs', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    return thresholds[np.argmin(costSensetiveLost)]

def analyseAllPreprocTypes(data, labels, withFI):

    #Labels preprocessing. Since I am going to use logistic regression, that output is the probability of 
    #a class being 0 or 1, we need to represent the labels as 0 and 1
    labels = labels.replace(to_replace = 2, value = 0)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)

    #if with FI find feature importances and use only most important features for training
    if withFI:

        model     = LogisticRegression().fit(X_train, y_train)
        perm_mean = feature_importance(model, X_train, y_train, 10)

        plt.figure(figsize=(12,8))
        plt.title('Feature importances', fontsize=20)
        perm_mean = np.array(perm_mean)
        idxSort = np.argsort(perm_mean)
        plt.plot(X_train.columns[np.flip(idxSort)[:22]], perm_mean[np.flip(idxSort)[:22]], 'bx')
        plt.xticks(X_train.columns[np.flip(idxSort)[:22]], X_train.columns[np.flip(idxSort)[:22]], rotation='vertical', fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylabel('Feature importances Logistic regression', fontsize=16)
        plt.show()

        bestPermImp = X_train.iloc[:, np.where(perm_mean > 0.0015)[0]]
        maxFI = bestPermImp.columns

        print("Shape of X_train with all features", X_train.shape)
        X_train = X_train[maxFI]
        print("Shape of X_train with most important features", X_train.shape)
        X_test  = X_test[maxFI]

    #Logistic regression model 
    logReg = LogisticRegression(max_iter=15000)
    #Try different regularization parameters C and find the best model 
    #validated by 5-fold cross valdation
    param_LogReg = {'C' : range(2, 16, 2)}

    grid_search_cv_logReg = GridSearchCV(logReg, param_LogReg, scoring='f1', cv=5)
    grid_search_cv_logReg.fit(X_train, y_train)

    best_logReg   = grid_search_cv_logReg.best_estimator_
    y_pred_default = best_logReg.predict(X_test)

    #Different validation metrics for predictions with default threshold 
    precisionDefault = precision_score(y_test, y_pred_default)
    recallDefault   = recall_score(y_test, y_pred_default)
    fpDefault       = np.sum((y_pred_default == 1) & (y_test == 0))
    fnDefault       = np.sum((y_pred_default == 0) & (y_test == 1))
    tpDefault       = np.sum((y_pred_default == 1) & (y_test == 1))
    tnDefault       = np.sum((y_pred_default == 0) & (y_test == 0))
    accuracyDefault = np.sum(y_pred_default == y_test)/len(y_test)
    costTresholdDefault = fpDefault*5 + fnDefault
    f1Default = f1_score(y_test, y_pred_default)

    metrics_default = pd.DataFrame({'Estimator' : ['Logistic Regression Default param'], 
                                    'Precision' : [precisionDefault],
                                    'Recall'    : [recallDefault],
                                    'Number FP' : [fpDefault],
                                    'Number FN' : [fnDefault],
                                    'Number TP' : [tpDefault],
                                    'Number TN' : [tnDefault],
                                    'Accuracy'  : [accuracyDefault],
                                    'Total cost with threshold' : [costTresholdDefault],
                                    'Threshold' : [0.5], 
                                    'F1 score' : [f1Default]})
    
    predict_probaLogReg = best_logReg.predict_proba(X_test)

    #predicted probabilities for class 1
    y_probaLogReg = predict_probaLogReg[:, 1]
 
    #The default classification threshold of binary classification is 0.5,
    #but since identifing of not creditworthy client as creditworthy is 5 times more expensive, than the other way around 
    #We need to find another threshold with a stricter rejection of false positives

    thresholdLogReg, fp, fn  = errorsType(y_test, y_probaLogReg)
    best_thresholdLogReg = round(best_threshold(fp, fn, thresholdLogReg), 2)

    y_pred_bestThreshold = np.where(y_probaLogReg > best_thresholdLogReg, 1, 0)
    
    metrics_differentThresholds  =  pd.DataFrame()

    minThreshold = best_thresholdLogReg - 0.1
    maxThreshold = best_thresholdLogReg + 0.1

    #Different validation metrics for predictions with different thresholds 
    for i in np.arange(minThreshold, maxThreshold, 0.01):
        y_pred_Diff_Thresholds = np.where(y_probaLogReg > i, 1, 0)
        precision = precision_score(y_test, y_pred_Diff_Thresholds)
        recall = recall_score(y_test, y_pred_Diff_Thresholds)
        f1 = f1_score(y_test, y_pred_Diff_Thresholds)
        fp = np.sum((y_pred_Diff_Thresholds == 1) & (y_test == 0))
        fn = np.sum((y_pred_Diff_Thresholds == 0) & (y_test == 1))
        tp = np.sum((y_pred_Diff_Thresholds == 1) & (y_test == 1))
        tn = np.sum((y_pred_Diff_Thresholds == 0) & (y_test == 0))
        accuracy = np.sum(y_pred_Diff_Thresholds == y_test)/len(y_test)
        cost_default  = metrics_default['Number FP'].loc[0]*5 + metrics_default['Number FN'].loc[0]
        cost_threshold = fp*5 + fn
        weightedFscore = tp / (tp + 0.5*(5*fp + fn))

        metrics =    pd.DataFrame({'Estimator' : ['Logistic Regression'], 
                                                  'Precision' : [precision],
                                                  'Recall'    : [recall],
                                                  'Number FP' : [fp],
                                                  'Number FN' : [fn],
                                                  'Number TP' : [tn],
                                                  'Number TN' : [tn],
                                                  'Accuracy'  : [accuracy],
                                                  'Total cost with current threshold' : [cost_threshold],
                                                  'Threshold' : [i], 
                                                  'F1 score' : [f1],
                                                  'Weighted F1 score' : [weightedFscore]})
        
        metrics_differentThresholds = metrics_differentThresholds.append(metrics)

    return best_logReg, y_probaLogReg, y_pred_bestThreshold, metrics_default, metrics_differentThresholds, y_test

def printROCcurve(y_test, y_proba, y_pred, metrics):
    
    #Roc curve for predicted probabilities
    fpr_default, tpr_default, threshold_default = roc_curve(y_test, y_proba)

    tn = np.sum((y_pred == 0) & (y_test == 0))

    metricsBestTreshold = metrics.loc[(metrics['Total cost with current threshold'] == metrics['Total cost with current threshold'].min())]

    #Find true positive rate and false positive rate with best threshold for our problem
    tpr_best = metricsBestTreshold['Recall'].loc[metricsBestTreshold['Accuracy'] == metricsBestTreshold['Accuracy'].max()] 
    fp_best = metricsBestTreshold['Number FP'].loc[metricsBestTreshold['Accuracy'] == metricsBestTreshold['Accuracy'].max()]

    fpr_best = fp_best / (fp_best + tn)

    plt.figure(figsize=(12,8))
    plt.title('ROC Curve', fontsize=20)
    plt.plot(fpr_default, tpr_default)
    plt.plot(fpr_best, tpr_best, marker = "x", c = 'black', linewidth=4, markersize=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=16)

    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()

#Load data set and set column names
df = pd.read_csv('kredit.dat', sep = '\t', header = None)
df.columns = ['Status_of_EA', 'Duration', 'Credit_hist', 'Purpose', 
              'Credit_amount', 'Savings_account', 'Present_ES', 
              'Installment_rate', 'Personal_status', 'Other_debtors', 
              'Present_RS', 'Property', 'Age', 'Other_IP', 'Housing', 
              'Num_exist_credits', 'Job', 'Number_of_people', 
              'Telephone', 'Foreign_worker', 'Creditworthy']

labels = df.Creditworthy
print("Labels initialy", np.array(labels[:5]))

#Initial data set
df.head()

#Data preprocessing. "Status of existing checking account" and "Savings account/bonds" features are one-hot encoded 
preprocessedData1 = preprocessing(df, True)

preprocessedData1.head()

# Present employment since
# A71: unemployed
# A72: ... < 1 year
# A73: 1 ≤ ... < 4 years
# A74: 4 ≤ ... < 7 years
# A75: .. ≥ 7 years

#Count number of clients for different employment durations
preprocessedData1.Present_ES.value_counts()

preprocessedData1.Present_ES.hist()

#Total number of clients
print(preprocessedData1.Present_ES.shape)

#Number of clients with missing values for "Present employment since feature"
preprocessedData1.Present_ES.isna().sum()

#Data preprocessing type 2. "Status of existing checking account" and "Savings account/bonds" features are represented with numerical values
preprocessedData2 = preprocessing(df, False)

preprocessedData2.head()

#Predict Present ES with 1st type of preprocessed dataset
data1 = predictPresentES(preprocessedData1)

#Predict Present ES with 2nd type of preprocessed dataset
data2 = predictPresentES(preprocessedData2)

#Use logistic regression model with 1st type preprocessed dataset using most important features
model_T1_Best, y_proba_T1_Best, y_pred_T1_Best, metricsDefault_T1_Best, metricsDiffThr_T1_Best, y_test_T1_Best = analyseAllPreprocTypes(data1, labels, True)

metricsDefault_T1_Best

metricsDiffThr_T1_Best

printROCcurve(y_test_T1_Best, y_proba_T1_Best, y_pred_T1_Best, metricsDiffThr_T1_Best)



#Use logistic regression model with 1st type preprocessed dataset using all features
model_T1_ALL, y_proba_T1_ALL, y_pred_T1_ALL, metricsDefault_T1_ALL, metricsDiffThr_T1_ALL, y_test_T1_ALL = analyseAllPreprocTypes(data1, labels, False)

metricsDefault_T1_ALL

metricsDiffThr_T1_ALL

printROCcurve(y_test_T1_ALL, y_proba_T1_ALL, y_pred_T1_ALL, metricsDiffThr_T1_ALL)



#Use logistic regression model with 2nd type preprocessed dataset using most important features
model_T2_Best, y_proba_T2_Best, y_pred_T2_Best, metricsDefault_T2_Best, metricsDiffThr_T2_Best, y_test_T2_Best = analyseAllPreprocTypes(data2, labels, True)

metricsDefault_T2_Best

metricsDiffThr_T2_Best

printROCcurve(y_test_T2_Best, y_proba_T2_Best, y_pred_T2_Best, metricsDiffThr_T2_Best)