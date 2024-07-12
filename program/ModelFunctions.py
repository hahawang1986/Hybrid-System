#### package load #####
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import math
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,cross_val_score, KFold,validation_curve,GridSearchCV
from sklearn.preprocessing import LabelEncoder
import csv
import joblib
from datetime import datetime


############# standardize_column_names ################
def standardize_column_names(df):
    df.columns = df.columns.str.replace(' ', '_')
    
    # Remove parentheses and any other special characters
    df.columns = df.columns.str.replace(r'[^\w\s]', '')

############# Split data###############################
def split_data(data, target_column, drop_column = [],test_size=0.3, valtest_available = 1, random_state=42):
    """
    Split the dataset into training, validation, and test sets, and return their features and labels.
    
    Parameters:
    - data: Input DataFrame containing the dataset.
    - target_column: Name of the target variable column.
    - drop_column: need to further delete column.
    - test_size: Proportion of the dataset to include in the test split. Default is 0.3.
    - random_state: Random seed for reproducibility. Default is 42.
    
    Returns:
    - Xtrain, Xval, Xtest: Features of training, validation, and test sets as DataFrames.
    - Ytrain, Yval, Ytest: Labels of training, validation, and test sets as Series.
    """
    # Split the dataset into training and temporary sets
    data_temp = data.drop(columns=[target_column])
    if drop_column != []:
        data_temp = data_temp.drop(columns=drop_column)
    Xtrain, Xtemp, Ytrain, Ytemp = train_test_split(data_temp, data[target_column],
                                                    test_size=test_size, random_state=random_state)
    if valtest_available == 1:
    # Split the temporary set into validation and test sets
        Xval, Xtest, Yval, Ytest = train_test_split(Xtemp, Ytemp, test_size=0.5, random_state=random_state)
    else:
        Xval,Yval = Xtemp,Ytemp
        Xtest,Ytest = None,None
    print(f"#######  {target_column} Data Set  ######")
    print(f"{target_column} Training set feature count:", Xtrain.shape)
    print(f"{target_column} Validation set feature count:", Xval.shape)
    #print(f"{target_column} Test set feature count:", Xtest.shape)
    print(f"{target_column} Training set label count:", Ytrain.shape)
    print(f"{target_column} Validation set label count:", Yval.shape)
    #print(f"{target_column} Test set label count:", Ytest.shape)
    
    return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest

################# data split1 ###############
def split_data1(data, target_column, drop_column = [],test_size=0.3, valtest_available = 0, random_state=42,synop_column = 'SYNOPCode'):
    """
    Split the dataset into training, validation, and test sets, and return their features and labels.
    
    Parameters:
    - data: Input DataFrame containing the dataset.
    - target_column: Name of the target variable column.
    - drop_column: need to further delete column.
    - test_size: Proportion of the dataset to include in the test split. Default is 0.3.
    - random_state: Random seed for reproducibility. Default is 42.
    
    Returns:
    - Xtrain, Xval, Xtest: Features of training, validation, and test sets as DataFrames.
    - Ytrain, Yval, Ytest: Labels of training, validation, and test sets as Series.
    """
    # Split the dataset into training and temporary sets
    data_temp = data.drop(columns=[target_column])
    if drop_column != []:
        data_temp = data_temp.drop(columns=drop_column)
    Xtrain, Xtemp, Ytrain, Ytemp = train_test_split(data_temp, data[target_column],
                                                    test_size=test_size, random_state=random_state,
                                                    stratify=data_temp[synop_column])
    if valtest_available == 1:
    # Split the temporary set into validation and test sets
        Xval, Xtest, Yval, Ytest = train_test_split(Xtemp, Ytemp, test_size=0.5, random_state=random_state)
    else:
        Xval,Yval = Xtemp,Ytemp
        Xtest,Ytest = None,None
    print(f"#######  {target_column} Data Set  ######")
    print(f"{target_column} Training set feature count:", Xtrain.shape)
    print(f"{target_column} Validation set feature count:", Xval.shape)
    #print(f"{target_column} Test set feature count:", Xtest.shape)
    print(f"{target_column} Training set label count:", Ytrain.shape)
    print(f"{target_column} Validation set label count:", Yval.shape)
    #print(f"{target_column} Test set label count:", Ytest.shape)
    
    return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest
############# GSCV grid #####################################################
def save_model(GS_mse, GS_r2, parameters, file_model="_RF",file_att = "FSO"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    
    def generate_file_name(prefix):
        file_name = f'{prefix}_{timestamp}'
        for param, values in parameters.items():
            file_name += f'_{param}-{min(values)}-{max(values)}-{len(values)}'
        file_name += file_model + '.pkl'
        return file_name
    
    file_name_fso_mse = generate_file_name('fso_gscv_mse')
    file_name_fso_r2 = generate_file_name('fso_gscv_r2')
    file_name_rfl_mse = generate_file_name('rfl_gscv_mse')
    file_name_rfl_r2 = generate_file_name('rfl_gscv_r2')
    
    if file_att == "FSO":
        joblib.dump(GS_mse, file_name_fso_mse)
        joblib.dump(GS_r2, file_name_fso_r2)
    else:
        joblib.dump(GS_mse, file_name_rfl_mse)
        joblib.dump(GS_r2, file_name_rfl_r2)
    
    return #file_name_fso_mse, file_name_fso_r2, file_name_rfl_mse, file_name_rfl_r2


########## don't use cv method and use validation to tune hyperparameter ###############
def save_model_results(results, parameters, file_model="_RF", file_att="FSO"):
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    
    def generate_file_name(prefix):
        file_name = f'{prefix}_{timestamp}'
        for param, values in parameters.items():
            file_name += f'_{param}-{min(values)}-{max(values)}-{len(values)}'
        file_name += file_model + file_att + '.csv'
        return file_name
    
    file_name_csv = generate_file_name('results')
 

    with open(file_name_csv, 'w', newline='') as csvfile:
        fieldnames = ['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'train_mse', 'train_r2', 'val_mse', 'val_r2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write the results
        for res in results:
            writer.writerow({
                'n_estimators': res['params']['n_estimators'],
                'max_depth': res['params']['max_depth'],
                'min_samples_leaf': res['params']['min_samples_leaf'],
                'min_samples_split': res['params']['min_samples_split'],
                'train_mse': res['train_mse'],
                'train_r2': res['train_r2'],
                'val_mse': res['val_mse'],
                'val_r2': res['val_r2']
            })

    print("Results saved to", file_name_csv)
    return file_name_csv


############# training with train set and validation set #####################
def rf_evaluate_parameter_grid(parameters, 
                            Xtrain, 
                            Ytrain, 
                            Xval, 
                            Yval,
                            n_jobs = 4):
    """
    Evaluate RandomForestRegressor with different parameter combinations.

    Parameters:
        parameters (dict): A dictionary containing parameter names as keys
                          and list of values to be tried as parameter values.
        Xtrain (array-like): Training input samples.
        Ytrain (array-like): Target values for training.
        Xval (array-like): Validation input samples.
        Yval (array-like): Target values for validation.
        n_jobs: the number of working cpu core
    Returns:
        list of dict: A list of dictionaries containing parameter combination
                      and performance metrics for each combination.
    """

    # Create parameter grid
    param_grid = ParameterGrid(parameters)

    # Initialize best parameters and best scores
    train_best_params = None
    train_best_score = float('inf')
    valid_best_params = None
    valid_best_score = float('inf')

    # Initialize list to store results for each parameter combination
    results = []
    i = 0
    # Iterate through parameter grid
    for params in param_grid:
        # Create RandomForestRegressor
        rfr = RandomForestRegressor(random_state=25, n_jobs=n_jobs, **params)

        # Fit the model
        rfr.fit(Xtrain, Ytrain)

        # Calculate predictions on training set
        y_train_pred = rfr.predict(Xtrain)

        # Calculate MSE and R2 on training set
        train_mse = mean_squared_error(Ytrain, y_train_pred)
        train_r2 = r2_score(Ytrain, y_train_pred)

        # Calculate predictions on validation set
        y_val_pred = rfr.predict(Xval)

        # Calculate MSE and R2 on validation set
        val_mse = mean_squared_error(Yval, y_val_pred)
        val_r2 = r2_score(Yval, y_val_pred)

        # Record parameter combination and performance metrics
        results.append({
            'params': params,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'val_mse': val_mse,
            'val_r2': val_r2
        })

        # Update best parameters and best scores
        if val_mse < valid_best_score:
            valid_best_score = val_mse
            valid_best_params = params
        if train_mse < train_best_score:
            train_best_score = train_mse
            train_best_params = params
        i+=1    
        if i % 10 == 0:
            print(i)
    # Print best parameters and best scores
    print("train Best parameters:", train_best_params)
    print("train Best score:", train_best_score)
    print("valid Best parameters:", valid_best_params)
    print("valid Best score:", valid_best_score)

    return results


############# functions in the part of random forest ############
# random forext model result plot #
# plot_coarse_tuning visilized coarse tunning result
# plot_fine_tuning visualized fine tunning result

def plot_coarse_tuning(df, channel = "FSO", group_column='max_depth', Xaxis='n_estimators',
                        metrics='RMSE', metrics1='R_square', legend_loc="upper right",\
                            markersize = 10,legend_fontsize = 14,xlim_s = 60,xlim_e = 500):
    # Group by group_column
    grouped = df.groupby(group_column)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    
    # Define color list
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    # Iterate over each group
    for i, (name, group) in enumerate(grouped):
        color = colors[i % len(colors)]  # Cycle through the color list
        group= group.sort_values(by=Xaxis)
        ax1.plot(group[Xaxis], np.sqrt(group['train_mse']), marker='o',markersize = markersize, color=color, label=f"train_rmse: {group_column}={name}")
        ax1.plot(group[Xaxis], np.sqrt(group['val_mse']), marker='^',markersize = markersize, linestyle='-', color=color, label=f"val_rmse: {group_column}={name}")

    # Add labels and legend to ax1
    ax1.set_xlabel(f'{Xaxis}', fontsize=18)
    ax1.set_ylabel(f'{metrics}', fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.legend(loc=legend_loc, framealpha=0.5,fontsize = legend_fontsize)
    ax1.set_title(f'{channel} Train and Validation {metrics}', fontsize=18)
    ax1.set_xlim(xlim_s, xlim_e)
    # Iterate over each group
    for i, (name, group) in enumerate(grouped):
        color = colors[i % len(colors)]  # Cycle through the color list
        group = group.sort_values(by=Xaxis)
        ax2.plot(group[Xaxis], (group['train_r2']), marker='o', markersize = markersize,color=color, label=f"train_r2: {group_column}={name}")
        ax2.plot(group[Xaxis], (group['val_r2']), marker='^',markersize = markersize, linestyle='-', color=color, label=f"val_r2: {group_column}={name}")

    # Add labels and legend to ax2
    ax2.set_xlabel(f'{Xaxis}', fontsize=18)
    ax2.set_ylabel(f'{metrics1}', fontsize=18)
    ax2.tick_params(axis='both', labelsize=16)
    ax2.legend(loc=legend_loc, framealpha=0.5,fontsize = legend_fontsize)
    ax2.set_title(f'{channel} Train and Validation {metrics1}', fontsize=18)
    ax2.set_xlim(xlim_s, xlim_e)
    fig.tight_layout()

    # Show plot
    plt.show()



def plot_fine_tuning(df, channel= 'FSO', lis=None, group_column='max_depth', Xaxis='n_estimators',\
                            metrics='RMSE', metrics1='R_square',markersize = 10,\
                            legend_loc="upper right",legend_loc1="right",
                            legend_fontsize = 14
                            ):
    # Group the dataframe by group_column
    grouped = df.groupby(group_column)

    # If lis parameter is provided, only select the interested groups
    if lis:
        grouped = {key: group for key, group in grouped if key in lis}

    # Create the figure
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 9))
    ax2 = ax1.twinx()  # Create the second y-axis, shared with the first subplot
    ax4 = ax3.twinx()
    # Define the color list
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    # Iterate over each group
    for i, (name, group) in enumerate(grouped.items()):
        color = colors[i % len(colors)]  # Cycle through the color list
        group = group.sort_values(by=Xaxis)
        ax1.plot(group[Xaxis], np.sqrt(group['train_mse']), marker='o', markersize = markersize,color=color, label=f"train_rmse: {group_column}={name}")
        ax2.plot(group[Xaxis], group['train_r2'], marker='^',markersize = markersize, linestyle='-', color=color, label=f"train_r2: {group_column}={name}")

    # Add labels and legend to the first subplot
    ax1.set_xlabel(f'{Xaxis}', fontsize=18)
    ax1.set_ylabel(f'{metrics}', fontsize=18)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.legend(loc=legend_loc, framealpha=0.5,fontsize = legend_fontsize)
    ax1.set_title(f'{channel} Train {metrics} and {metrics1}', fontsize=18)

    # Set labels and legend for the second y-axis
    ax2.set_ylabel(f'{metrics1}', fontsize=18)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.legend(loc=legend_loc1, framealpha=0.5,fontsize = legend_fontsize)

    # Iterate over each group again for the second subplot
    for i, (name, group) in enumerate(grouped.items()):
        color = colors[i % len(colors)]  # Cycle through the color list
        group = group.sort_values(by=Xaxis)
        ax3.plot(group[Xaxis], np.sqrt(group['val_mse']), marker='o', markersize = markersize,color=color, label=f"val_rmse: {group_column}={name}")
        ax4.plot(group[Xaxis], group['val_r2'], marker='^', markersize = markersize,linestyle='-', color=color, label=f"val_r2: {group_column}={name}")

    # Add labels and legend to the second subplot
    ax3.set_xlabel(f'{Xaxis}', fontsize=18)
    ax3.set_ylabel(f'{metrics}', fontsize=18)
    ax3.tick_params(axis='both', labelsize=16)
    ax3.legend(loc=legend_loc, framealpha=0.5,fontsize = legend_fontsize)
    ax3.set_title(f'{channel} Validation {metrics} and {metrics1}', fontsize=18)

    # Set labels and legend for the second y-axis
    ax4.set_ylabel(f'{metrics1}', fontsize=18)
    ax4.tick_params(axis='y', labelsize=16)
    ax4.legend(loc=legend_loc1, framealpha=0.5,fontsize = legend_fontsize)

    plt.tight_layout()

    # Display the plot
    plt.show()

############### caculate_mrtrics #########################
    """
    Calculate evaluation metrics of the model, including RMSE and R² for both training and validation sets,
    and optionally compute Out-of-Bag (OOB) R².

    Parameters:
    - model: Trained regression model.
    - Xtrain: Features of the training set.
    - Ytrain: Labels of the training set.
    - Xval: Features of the validation set.
    - Yval: Labels of the validation set.
    - oob: Whether to compute Out-of-Bag (OOB) R². Default is True.

    Returns:
    - train_rmse: RMSE of the training set.
    - train_r2: R² of the training set.
    - val_rmse: RMSE of the validation set.
    - val_r2: R² of the validation set.
    - oob_r2: Optional, Out-of-Bag (OOB) R² (if computed).
    """
def caculate_mrtrics(model,Xtrain,Ytrain,Xval,Yval,oob = True):
    y_pred_train = model.predict(Xtrain)
    y_pred_val = model.predict(Xval)
    
    train_rmse = mean_squared_error(Ytrain, y_pred_train, squared=False)
    val_rmse = mean_squared_error(Yval, y_pred_val, squared=False)
    print("Training RMSE:", train_rmse)
    print("Validation RMSE:", val_rmse)

    train_r2 = r2_score(Ytrain, y_pred_train)
    val_r2 = r2_score(Yval, y_pred_val)
    print("Train R²:", train_r2)
    print("Validation R²:", val_r2)
    
    if oob:
        oob_r2 = model.oob_score_
        print("OOB R²:", oob_r2)
    else:
        oob_r2 = None

        # oob_predictions = np.zeros_like(Ytrain, dtype=float)
        # for tree in model.estimators_:
        #     # 确定 Out-of-Bag (OOB) 数据的索引
        #     mask = ~np.isin(range(len(Xtrain)), tree.indices_)
        #     # 预测 Out-of-Bag (OOB) 数据
        #     oob_predictions[mask] += tree.predict(Xtrain[mask])

        # # 计算 Out-of-Bag (OOB) 数据的预测值
        # oob_predictions /= len(model.estimators_)

        # # 计算 Out-of-Bag (OOB) RMSE
        # oob_rmse = mean_squared_error(Ytrain, oob_predictions, squared=False)
        # print("OOB RMSE:", oob_rmse)

    return train_rmse, train_r2,val_rmse,val_r2,oob_r2

########### feature-importance ##############

def sort_features(model,Xtrain,channel = "FSO",figure = False):
    """
    Sort features based on their importance in the model.

    Parameters:
    - model: Trained model with feature_importances_ attribute.
    - Xtrain: DataFrame containing features.
    - channel: Optional, name of the channel. Default is "FSO".
    - figure: Optional, whether to plot feature importance. Default is False.

    Returns:
    - sorted_features: List of sorted feature names.
    - sorted_importance: List of corresponding feature importance values.
    """
    #model.fit(Xtrain, Ytrain)
    feature_importance = model.feature_importances_
    # print("Feature Importance:", feature_importance)
    feature_importance_dict = dict(zip(Xtrain.columns, feature_importance))
    sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=False)
    sorted_features= [x[0] for x in sorted_feature_importance]
    sorted_importance = [x[1] for x in sorted_feature_importance]

    if figure:
        plt.figure(figsize=(12, 9))
        bars = plt.barh(sorted_features, sorted_importance)
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width()*100:.1f}%', 
                    va='center', ha='left', fontsize=16, color='black')

        plt.xlabel('Feature Importance Ratio',fontsize=18)
        plt.ylabel('Features',fontsize=18)
        plt.title(f'Feature Importance of {channel} RF Model',fontsize=18)
        plt.tick_params(axis='both', labelsize=16)
        plt.show()
    return sorted_features, sorted_importance

def sort_features_1(model, Xtrain, channel="FSO", figure=False, threshold=0.025):
    """
    Sort features based on their importance in the model.

    Parameters:
    - model: Trained model with feature_importances_ attribute.
    - Xtrain: DataFrame containing features.
    - channel: Optional, name of the channel. Default is "FSO".
    - figure: Optional, whether to plot feature importance. Default is False.
    - threshold: Optional, the threshold below which features are considered unimportant. Default is 0.02 (2%).

    Returns:
    - sorted_features: List of sorted feature names.
    - sorted_importance: List of corresponding feature importance values.
    """
    feature_importance = model.feature_importances_
    feature_importance_dict = dict(zip(Xtrain.columns, feature_importance))

    # Filter out features with importance below the threshold
    important_features = {k: v for k, v in feature_importance_dict.items() if v >= threshold}
    unimportant_features = {k: v for k, v in feature_importance_dict.items() if v < threshold}

    # Sum the importance of unimportant features
    sum_unimportant_importance = sum(unimportant_features.values())

    # Combine important features and the sum of unimportant features
    combined_importance_dict = {**important_features, 'Other Features': sum_unimportant_importance}

    # Sort the combined importance dictionary
    sorted_combined_importance = sorted(combined_importance_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_features = [x[0] for x in sorted_combined_importance]
    sorted_importance = [x[1] for x in sorted_combined_importance]

    if figure:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 9))
        bars = plt.barh(sorted_features, sorted_importance)
        for bar in bars:
            plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width() * 100:.1f}%',
                     va='center', ha='left', fontsize=16, color='black')

        plt.xlabel('Feature Importance Ratio', fontsize=18)
        plt.ylabel('Features', fontsize=18)
        plt.title(f'Feature Importance of {channel} RF Model', fontsize=18)
        plt.tick_params(axis='both', labelsize=16)
        plt.show()

    return sorted_features, sorted_importance


####################### wrapper_method for exploring feature importance ##################
def wrapper_method (model,Xtrain,Ytrain,Xval,Yval,channel = "FSO"):
    """
    Perform feature selection using a wrapper method.

    Parameters:
    - model: Model to be used for training and feature selection.
    - Xtrain: Training set features.
    - Ytrain: Training set labels.
    - Xval: Validation set features.
    - Yval: Validation set labels.
    - channel: Optional, name of the channel. Default is "FSO".

    Returns:
    - performance_scores: List of tuples containing performance metrics for each feature subset.
                          Each tuple includes feature removed, training RMSE, training R², validation RMSE, validation R²,
                          OOB R² (if applicable), and the most important feature in the subset.
    """
    performance_scores = []
    for i in range(len(Xtrain.columns)+1):
        if i == 0:
            model.fit(Xtrain, Ytrain)
            sorted_features, sorted_importance = \
                sort_features(model,Xtrain,channel = "FSO",figure = False)
            train_rmse, train_r2,val_rmse,val_r2,oob_r2= \
                caculate_mrtrics(model,Xtrain,Ytrain,Xval,Yval,oob = True)

        else:

            remove_feature = sorted_features[0]
            features_list =  sorted_features[1::] # 每次把最不重要的特征剔除
            print(i,remove_feature,features_list) # the final loop is like "distance,[ ]"
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%") # so len(features_list) >= 1 to avoid Xtrain[features_list] to be none
            if remove_feature and len(features_list) >= 1:
            #     model.fit(Xtrain[remove_feature], Ytrain)
                
            #     sorted_features, sorted_importance = \
            #         sort_features(model,Xtrain[remove_feature] ,Ytrain,channel = "FSO")
            #     train_rmse, train_r2,val_rmse,val_r2,oob_r2= \
            #         caculate_mrtrics(model,Xtrain[remove_feature],Ytrain,Xval[remove_feature],Yval,oob = True)
            # else:
                model.fit(Xtrain[features_list], Ytrain)
                
                sorted_features, sorted_importance = \
                    sort_features(model,Xtrain[features_list] ,channel = "FSO",figure = False)
                train_rmse, train_r2,val_rmse,val_r2,oob_r2= \
                    caculate_mrtrics(model,Xtrain[features_list],Ytrain,Xval[features_list],Yval,oob = True)
        if i == 0:
            performance_scores.append(("All Features",train_rmse, train_r2,val_rmse,val_r2, oob_r2,sorted_features[0],sorted_importance[0]))
        else:
            performance_scores.append((remove_feature,train_rmse, train_r2,val_rmse,val_r2, oob_r2,sorted_features[0],sorted_importance[0]))
    
    
    return performance_scores

################ plot_feature_importance ######################
def plot_feature_importance(df,threshold="VisibilityMin", 
                            channel="FSO", markersize=10, legend_loc="lower left", 
                            legend_loc1="upper left",legend_x=0.5, legend_y=0.5):
    """
    Plot the feature importance metrics including RMSE and R_square, along with an indicator line at a specified threshold feature.

    Parameters:
        df (DataFrame): DataFrame containing the feature importance metrics.
        threshold (str): The name of the feature at which the indicator line will be drawn. Default is "VisibilityMin".
        channel (str): Name of the model or channel. Default is "FSO".
        markersize (int): Size of the markers in the plot. Default is 10.
        legend_loc (str): Location of the legend for the first y-axis. Default is "lower left".
        legend_loc1 (str): Location of the legend for the second y-axis. Default is "upper left".
    """
    remove_features = df["remove_feature"]
    train_rmse = df["train_rmse"]
    train_r2 = df["train_r2"]
    val_rmse = df["val_rmse"]
    val_r2 = df["val_r2"]
    oob_r2 = df["oob_r2"]
    
    fig, ax1 = plt.subplots(figsize=(12, 9))
    ax2 = ax1.twinx()  # Create the second y-axis, shared with the first subplot

    # Define the color list
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    ax1.plot(remove_features, train_rmse, marker='o', markersize=markersize, color=colors[0], label="train_rmse")
    ax1.plot(remove_features, val_rmse, marker='^', markersize=markersize, color=colors[0], label="validation_rmse")
    ax2.plot(remove_features, train_r2, marker='o', markersize=markersize, linestyle='--', color=colors[2], label="train_r2")
    ax2.plot(remove_features, val_r2, marker='^', markersize=markersize, linestyle='--', color=colors[2], label="validation_r2")
    ax2.plot(remove_features, oob_r2, marker='*', markersize=markersize, linestyle='--', color=colors[2], label="oob_r2")
    
    ax1.set_ylabel('RMSE', fontsize=18, color=colors[0])
    ax1.tick_params(axis='y', labelsize=16, colors=colors[0])
    
    ax2.set_ylabel('R_square', fontsize=18, color=colors[2])
    ax2.tick_params(axis='y', labelsize=16, colors=colors[2])
    
    ax1.legend(loc=legend_loc, framealpha=0.5, fontsize=16)
    #ax2.legend(loc=legend_loc1, framealpha=0.5, fontsize=16)
    ax2.legend(loc="center", bbox_to_anchor=(legend_x, legend_y), framealpha=0.5, fontsize=16)  # Adjust legend position
    #ax1.set_xlabel('Features', fontsize=18)
    ax1.set_title(f'Feature Pruning of {channel} Model', fontsize=18)
    ax1.grid(True)
    
    ax1.set_xticks(remove_features)
    ax1.set_xticklabels(remove_features, rotation=80, fontsize=16)
    
    index = remove_features.tolist().index(threshold)
    ax1.axvline(x=index, color=colors[1], linestyle='--',linewidth=4)

    plt.tight_layout()
    plt.show()
    return fig
#################### plot for SYNOPCode split ####################
def plot_feature_importance_code(df,code =0,threshold="VisibilityMin", channel="FSO", markersize=10, 
                                 legend_loc="lower left", legend_loc1="upper left",
                                 legend_valid = 0,legend_x1 = 0.2,legend_y1 = 0.4
                                 ,legend_x2 = 0.5,legend_y2 = 0.4):
    """
    Plot the feature importance metrics including RMSE and R_square, along with an indicator line at a specified threshold feature.

    Parameters:
        df (DataFrame): DataFrame containing the feature importance metrics.
        threshold (str): The name of the feature at which the indicator line will be drawn. Default is "VisibilityMin".
        channel (str): Name of the model or channel. Default is "FSO".
        markersize (int): Size of the markers in the plot. Default is 10.
        legend_loc (str): Location of the legend for the first y-axis. Default is "lower left".
        legend_loc1 (str): Location of the legend for the second y-axis. Default is "upper left".
    """
    abbreviation_mapping = {
    'All Features': 'AF',
    'WindDirection': 'WD',
    'WindSpeed': 'WS',
    'WindSpeedMin': 'WSmin',
    'ParticulateMin': 'PMmin',
    'WindSpeedMax': 'WSmax',
    'TemperatureMin': 'Tmin',
    'TemperatureMax': 'Tmax',
    'Particulate': 'PM',
    'VisibilityMin': 'VMmin',
    'AbsoluteHumidityMax': 'AHmax',
    'TemperatureDifference': 'Tdiff',
    'AbsoluteHumidityMin': 'AHmin',
    'Time': 'Time',
    'VisibilityMax': 'VMmax',
    'Frequency': 'Freq',
    'RelativeHumidity': 'RH',
    'ParticulateMax': 'PMmax',
    'Visibility': 'Vis',
    'SYNOPCode': 'SC',
    'Temperature': 'Temp',
    'RainIntensityMin': 'RImin',
    'RainIntensityMax': 'RImax',
    'Distance': 'Dist',
    'RainIntensity': 'RI',
    'AbsoluteHumidity': 'AH'
}

    if code == 0:
        weather = "Clear"
        train_num = 39875
        test_num = 17089
    elif code == 3:
        weather = "Dust Storm"
        train_num = 134 
        test_num = 57
    elif code == 4:
        weather = "Fog"
        train_num = 326 
        test_num = 140
    elif code == 5:
        weather = "Drizzle"
        train_num = 4623 
        test_num = 1982
    elif code == 6:
        weather = "Rain"
        train_num = 17513 
        test_num = 7505
    elif code == 7:
        weather = "Snow"
        train_num = 293 
        test_num = 126
    elif code == 8:
        weather = "Showers"
        train_num = 1201 
        test_num = 515
    else:
        weather = "All Weather"
        train_num = 63965 
        test_num = 27414

# 使用映射字典将原内容替换为简称
    df["remove_feature"] = df["remove_feature"].map(abbreviation_mapping)
    remove_features = df["remove_feature"]
    train_rmse = df["train_rmse"]
    train_r2 = df["train_r2"]
    val_rmse = df["val_rmse"]
    val_r2 = df["val_r2"]
    oob_r2 = df["oob_r2"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()  # Create the second y-axis, shared with the first subplot

    # Define the color list
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    ax1.plot(remove_features, train_rmse, marker='o', markersize=markersize, color=colors[0], label="train_rmse")
    ax1.plot(remove_features, val_rmse, marker='^', markersize=markersize, color=colors[0], label="test_rmse")
    ax2.plot(remove_features, train_r2, marker='o', markersize=markersize, linestyle='--', color=colors[2], label="train_r2")
    ax2.plot(remove_features, val_r2, marker='^', markersize=markersize, linestyle='--', color=colors[2], label="test_r2")
    ax2.plot(remove_features, oob_r2, marker='*', markersize=markersize, linestyle='--', color=colors[2], label="oob_r2")
    
    ax1.set_ylabel('RMSE', fontsize=18, color=colors[0])
    ax1.tick_params(axis='y', labelsize=16, colors=colors[0])
    
    ax2.set_ylabel('R_square', fontsize=18, color=colors[2])
    ax2.tick_params(axis='y', labelsize=16, colors=colors[2])
    
    
    #ax1.set_xlabel('Features', fontsize=18)
    ax1.set_title(f'{channel} for {weather} (train_num:{train_num},test_num:{test_num}))', fontsize=18)
    ax1.grid(True)
    
    ax1.set_xticks(remove_features)
    ax1.set_xticklabels(remove_features, rotation=80, fontsize=16)
    
    index = remove_features.tolist().index(threshold)
    ax1.axvline(x=index, color=colors[1], linestyle='--',linewidth=4)
    if legend_valid ==0 :
        ax1.legend(loc=legend_loc, framealpha=0.5, fontsize=16)
        ax2.legend(loc=legend_loc1, framealpha=0.5, fontsize=16)
    else:
        ax1.legend(loc="center", bbox_to_anchor=(legend_x1, legend_y1), framealpha=0.5, fontsize=16)
        ax2.legend(loc="center", bbox_to_anchor=(legend_x2, legend_y2), framealpha=0.5, fontsize=16)

    plt.tight_layout()
    plt.show()
    return fig

######################################################
def filter_data_by_SYNOPCode(X, Y, code = 9):
    """
    Filter data based on the SYNOPCode value.

    Parameters:
        X (DataFrame): The DataFrame containing the features data.
        Y (Series): The Series containing the target data.
        code (int): The value of SYNOPCode used for filtering.

    Returns:
        Tuple of DataFrames: (Xdata, Ydata)
            Xdata: The subset of the features data where SYNOPCode equals the given code.
            Ydata: The subset of the target data corresponding to the filtered features.
    """
    # Concatenate features and target data
    trainset = pd.concat([X, Y], axis=1)
    
    # Filter data based on SYNOPCode value
    if code == 9:  # 9 is all weather
        Xdata = X
        Ydata = Y
    else:
        filtered_data = trainset[trainset['SYNOPCode'] == code]
        
        # Separate features and target
        Xdata = filtered_data.iloc[:, :-1]  # All columns except the last one
        Ydata = filtered_data.iloc[:, -1]   # Last column
    
    return Xdata, Ydata
