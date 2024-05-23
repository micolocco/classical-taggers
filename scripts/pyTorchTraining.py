import numpy as np 
import os
import time
import torch
from torch.utils.data import DataLoader
import copy
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve
from sklearn.linear_model import LogisticRegression
import pickle
from scipy.special import expit
import json
import pipeline

# Local imports
from scripts.NNModel import EarlyStopper
from scripts.inputDataset import inputDataset
from scripts import ranges, nice_names, matplotlib_lhcb_style
matplotlib_lhcb_style(plt)
import yaml

def get_features(tagger, yaml_file):
    '''Function for assigning the input features corresponding to each tagger.
    The input features will be used for the training of the NN
    yaml_file: Configuration file for getting the input features'''
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        if tagger in config:
            return config[tagger]['features']
        else:
            print(f"Error: Tagger {tagger} not found in configuration.\n Please check {yaml_file} file ")
            return []

# def apply_log(df, features):
#     log_features = ['Column1']
#     for feature in features:
#         if 'log' in feature:
#             df.eval(f'log({feature}) = log({feature})', inplace = True)
#     return features


def splitByEvent (df, config):
    '''Function to random split by events (not by index) the dataset into training and test set
    Use random.Random(2) to reproduce same shuffling''' 
    import random
    events_list = np.unique(df.event_entry)
    random.Random(3).shuffle(events_list) # cfg.seed
    n_train_val = int(config['train_val_split']*len(events_list)) # Divide
    n_train = int(0.8 * n_train_val)
    train_df = df[df.event_entry.isin(events_list[:n_train])].copy()
    val_df = df[df.event_entry.isin(events_list[n_train:n_train_val])].copy()
    test_df = df[df.event_entry.isin(events_list[n_train_val:])].copy()
    return train_df.query('selected==1'), val_df.query('selected==1'), test_df
    

def prepare_data(train_df, features, val_df, savePlot_path, scalerPath, transformerPath, train_batch_size = 32, test_batch_size = 1024):
    # Load the dataset
    train_dataset = inputDataset(df=train_df) #scaler=PowerTransformer() 
    train_dataset.scale(test=False, scalerPath=scalerPath, transformerPath=transformerPath)
    val_dataset = inputDataset(df=val_df)
    val_dataset.scale(test=True, scalerPath=scalerPath, transformerPath=transformerPath)
    plot_features(data=train_df, features_list=features, target_path=savePlot_path, flag='label', name=f'training_inputFeatures')
    # Prepare data loaders
    train_dl = DataLoader(train_dataset, batch_size = train_batch_size, shuffle=True)
    validation_dl = DataLoader(val_dataset, batch_size = test_batch_size, shuffle=False)
    return train_dl, validation_dl 


def plot_features(data, features_list, target_path, name, flag, nbins=100):

    # Plot input features 
    plt.figure(figsize=(24,25))
    pos=0
    for i, col in enumerate(data.columns.to_list()):
        if col in features_list:
            plt.subplot(4, 4 , pos + 1) # hardcoded according to the number of features
            if col in nice_names.keys():
                plt.hist(data[col][data[flag]==0], density = True, bins=nbins, label = f"{flag} = 0",color='b', alpha=0.5, range=ranges[col])
                plt.hist(data[col][data[flag]==1], density = True, bins=nbins, label = f"{flag} = 1",color='r', alpha=0.5, range=ranges[col])
                plt.xlabel(nice_names[col])
            else:
                plt.hist(data[col][data[flag]==0], density = True, bins=nbins, label = f"{flag} = 0",color='b', alpha=0.5, )
                plt.hist(data[col][data[flag]==1], density = True, bins=nbins, label = f"{flag} = 1",color='r', alpha=0.5, )
                plt.xlabel(col)

            plt.legend()
            plt.tight_layout()
            pos+=1
    plt.savefig(f"{target_path}/{name}.pdf")
    

def train_model_EarlyStopping(model, train_dl, validation_dl, target_path, config):
        
        trainingEpoch_loss = []
        validationEpoch_loss = []
        lossValBest = 10000
        rollingAverageNew = 0
        rollingAverageOld = 10000 # just to be sure that the first rolling average value is lower than this
        stopped = False
        bestEpoch = 0

        training_start = time.time()
        early_stopper = EarlyStopper(patience=config['patience'], min_delta=config['min_delta'])
        
        i = 1
        initial_validation_loss = model.validate_model(validation_dl)
        print("Initial Validation Loss:", initial_validation_loss)

        for epoch in range(config['n_epochs']):
            epoch_start = time.time()
            print(f"--------------Epoch:{epoch+1}/{config['n_epochs']}-------------")
            stepLoss = model.train_model(train_dl, epoch, config['n_epochs'])
            # Train over mini-batches
            trainingEpoch_loss.append(np.array(stepLoss).mean())
            # Compute validation loss
            validationStep_loss = model.validate_model(validation_dl)
            validationEpoch_loss.append(np.array(validationStep_loss).mean())
            print(f"Train:{np.array(stepLoss).mean():.6f}, Validation:{np.array(validationStep_loss).mean():.6f}, Time:{round((time.time()-epoch_start) ,2)}s") 
            if early_stopper.early_stop(validationEpoch_loss[-1]): 
                stopped = True 
                break
            if early_stopper.counter == 0:
                lossValBest = validationEpoch_loss[-1]
                lossTrainBest = trainingEpoch_loss[-1]
                bestEpoch = epoch
                save_model(model, target_path)
                bestModel = copy.deepcopy(model)
            i +=1
        training_time = round((time.time()- training_start) / 60 , 2)
        print(f"Training finished in {training_time} min, {i-1} epochs, early stopping: {stopped}")
        return bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, np.array([lossTrainBest, lossValBest], dtype=float)
            #if epoch > earlyStop:  # check the termination condition
            #    rollingAverageNew = np.mean(validationEpoch_loss[-earlyStop:])
            #    if validationEpoch_loss[-1] < lossValBest:
            #        lossValBest = validationEpoch_loss[-1]
            #        lossTrainBest = trainingEpoch_loss[-1]
            #        bestEpoch = epoch
            #        save_model(model, name_formatter)
            #        bestModel = copy.deepcopy(model)
            #        
            #if epoch > (earlyStop +1):
            #    if rollingAverageNew > rollingAverageOld:
            #        stopped = True 
            #        break
            #    rollingAverageOld = rollingAverageNew 
            
        #return bestModel, trainingEpoch_loss, validationEpoch_loss, bestEpoch, dtype=float)

def save_model(model, target_path):
    # target_path = name_formatter.assign_name(folder, target_path)
    torch.save(copy.deepcopy(model.state_dict()), f"{target_path}/model.pth")
    #save_hyperparameters(model, target_path)

def save_hyperparameters(model, target_path):
    info_dict = {
                'ModelName:' : model.modelName, 
                'learning_rate' : config.learning_rate,
                'patience' : config.patience,
                'min_delta' : config.min_delta,
                'activation_function' : config.activation_function,
                'n_epochs' : config.n_epochs,
                'train_val_split' : config.train_val_split,
                'train_batch_size': config.train_batch_size,
                }
    with open(f"{target_path}/hyperparameters.json", "w") as f:
        json.dump(info_dict, f)


def load_model(model, target_path):
   # target_path = name_formatter.assign_name(folder, target_path)    
    # saveName = name_formatter.assign_name(target_path, model.modelName)
    model.load_state_dict(torch.load(f"{target_path}/model.pth"))
    

def save_losses(trainLoss, valLoss, bestEpoch, bestLosses, target_path):
    
    folder = f'{target_path}/losses'
    os.makedirs(f'{folder}', exist_ok=True)
    np.savetxt(f"{folder}/test.csv", valLoss, delimiter=",")
    np.savetxt(f"{folder}/train.csv", trainLoss, delimiter=",")
    np.savetxt(f"{folder}/best.csv", [bestEpoch,bestLosses[0],bestLosses[1]], delimiter=",")

def plot_losses(tagger, trainLoss, valLoss, bestEpoch, bestLosses, target_path):
    
    plt.figure()
    plt.plot(trainLoss, label='Training', c = 'orange')
    plt.plot(valLoss,label='Validation', c='blue')
    plt.axvline(bestEpoch, linestyle='--', color='tab:gray', label="Best epoch")
    plt.legend(loc = "best")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.title(f"{tagger}", fontsize=24)
    plt.savefig(f"{target_path}/Loss.pdf")
   
def plot_ROC(tagger, yPred, yTrue, target_path, yPredTrain = None, yTrueTrain = None):
    
    plt.figure()
    lw  = 2
    fpr_test, tpr_test,_ = roc_curve(yTrue, yPred)
    roc_auc_test = round(auc(fpr_test, tpr_test),5)
    plt.plot(fpr_test, tpr_test, color='darkblue',lw=lw, label=f'Test (area = {roc_auc_test})' )
    plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title(f"{tagger}", fontsize=24)
    if yPredTrain is not None:
        fpr, tpr,_ = roc_curve(yTrueTrain, yPredTrain)
        roc_auc = round(auc(fpr, tpr),5)
        plt.plot(fpr, tpr, color='darkorange',lw=lw, label=f'Train (area = {roc_auc})' )
        plt.legend(loc="lower right")
        plt.savefig(f"{target_path}/ROC_TRAIN_VAL.pdf")
    else:
        plt.savefig(f"{target_path}/ROC_TEST.pdf")

def logistic_regression(yPredTrain, yTrueTrain, target_path):
    
    clf = LogisticRegression().fit(yPredTrain, yTrueTrain.ravel())  
    pickle.dump(clf , open(f"{target_path}/LogReg.pck" , "wb"))
    return clf
'''
def plot_NNoutput_mistag (name, clf, yPredTest, yTrueTest, yPredTrain, yTrueTrain, target_path, nbins=100):
    
    plt.figure()
    LR_test = np.linspace(0, 1, 300)
    loss = expit(LR_test * clf.coef_ + clf.intercept_).ravel()
    plt.title("Logistic Regression")
    plt.grid()
    plt.plot(yPredTest[yTrueTest == 0][0:500], np.zeros(500) , "b.",alpha = 0.5, label = "Label = 0")
    plt.plot(yPredTest[yTrueTest == 1][0:500], np.ones(500) ,  "r.",alpha = 0.5, label = "Label = 1")
    plt.plot(LR_test, loss ,color = "k")
    plt.legend(loc = "best")
   # saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{target_path}/LogReg.pdf")
    
    prob_train_0_height , prob_train_0_bin_edges= np.histogram(yPredTrain[yTrueTrain == 0] , bins = nbins, density = True)
    prob_train_0_bin_edges = prob_train_0_bin_edges[:len(prob_train_0_bin_edges)-1]+ (prob_train_0_bin_edges[1]-prob_train_0_bin_edges[0])/2
    prob_train_1_height , prob_train_1_bin_edges= np.histogram(yPredTrain[yTrueTrain == 1] ,bins = nbins, density = True)
    prob_train_1_bin_edges = prob_train_1_bin_edges[:len(prob_train_1_bin_edges)-1]+ (prob_train_1_bin_edges[1]-prob_train_1_bin_edges[0])/2

    y_test_predict_LR = clf.predict_proba(yPredTest)[:,0]
    y_train_predict_LR = clf.predict_proba(yPredTrain)[:,0]

    prob_train_0_height_LR , prob_train_0_bin_edges_LR= np.histogram(y_train_predict_LR[yTrueTrain.ravel() == 0], bins = nbins, density = True)
    prob_train_0_bin_edges_LR = prob_train_0_bin_edges_LR[:len(prob_train_0_bin_edges_LR)-1]+ (prob_train_0_bin_edges_LR[1]-prob_train_0_bin_edges_LR[0])/2
    prob_train_1_height_LR , prob_train_1_bin_edges_LR= np.histogram(y_train_predict_LR[yTrueTrain.ravel() == 1], bins = nbins, density = True)
    prob_train_1_bin_edges_LR = prob_train_1_bin_edges_LR[:len(prob_train_1_bin_edges_LR)-1]+ (prob_train_1_bin_edges_LR[1]-prob_train_1_bin_edges_LR[0])/2
    
    fig, axs = plt.subplots(1,2, figsize = (10,5))
    axs[0].set_title("NN Output")
    axs[0].set_yscale("log")
    axs[0].hist(yPredTest[yTrueTest == 0],bins = nbins, density = True,histtype="stepfilled",color = "b", alpha = 0.5, label = "Test (Label = 0)")
    axs[0].hist(yPredTest[yTrueTest == 1],bins = nbins, density = True,histtype="stepfilled",color = "r", alpha = 0.5, label = "Test (Label = 1)")
    axs[0].plot(prob_train_0_bin_edges, prob_train_0_height, "b.", label = "Train (Label = 0)")
    axs[0].plot(prob_train_1_bin_edges, prob_train_1_height, "r.", label = "Train (Label = 1)")
    axs[0].grid()
    axs[0].set_xlabel(r"NN output")
    axs[0].set_ylabel("Normalized number of tracks")
    axs[0].legend(loc = "best")
    axs[1].set_title("LogReg Output")
    axs[1].set_yscale("log")
    axs[1].set_ylabel("Normalized number of tracks")
    axs[1].set_xlabel(r"Logistic(NN ouput)")
    axs[1].hist(y_test_predict_LR[yTrueTest.ravel() == 0],bins = nbins,density = True,histtype="stepfilled",color = "b", alpha = 0.5, label = "Test (Label = 0)")
    axs[1].hist(y_test_predict_LR[yTrueTest.ravel() == 1],bins = nbins,density = True,histtype="stepfilled",color = "r", alpha = 0.5, label = "Test (Label = 1)")
    axs[1].plot(prob_train_0_bin_edges_LR ,prob_train_0_height_LR, "b.", label = "Train (Label = 0)")
    axs[1].plot(prob_train_1_bin_edges_LR ,prob_train_1_height_LR, "r.", label = "Train (Label = 1)")
    axs[1].grid()
    axs[1].legend(loc = "best")

    #folder = 'plots'
    #saveName = name_formatter.assign_name(folder, name)
    plt.savefig(f"{target_path}/NNoutput_sigmoid.pdf")
    plt.close()
'''

def plot_mistag(tagger, yPred, yTrue, target_path, type,  clf = None, nbins=100):
    plt.figure()
    # plt.title("Mistag rate")
    plt.yscale("log")
    if clf:
        y_predict_LR = clf.predict_proba(yPred)[:,0]
        plt.hist(y_predict_LR[yTrue.ravel() == 0],bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = f"wrong tagging decision")
        plt.hist(y_predict_LR[yTrue.ravel() == 1],bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = f"correct tagging decision")
        plt.title(f'{tagger} mistag after Logistic Regression', fontsize=24)
    else:
        plt.hist(1-yPred[yTrue.ravel() == 0],bins = nbins, density = True, histtype="stepfilled", color = "b", alpha = 0.5, label = f"wrong tagging decision")
        plt.hist(1-yPred[yTrue.ravel() == 1],bins = nbins, density = True, histtype="stepfilled", color = "r", alpha = 0.5, label = f"correct tagging decision")
        plt.title(f"{tagger}", fontsize=24)

    plt.xlabel(r"NN output", fontsize=24)
    #plt.annotate(f'{len(yPred)} tracks', xy=(0, 1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    plt.grid()
    plt.ylabel("Normalized number of tracks", fontsize=24)
    plt.legend(loc = "best", title=f'{type}:{len(yPred)} total tracks')
    plt.savefig(f"{target_path}/mistag_{type}.pdf")
    

def plot_tagDec(tagger,df_TagParticles, target_path, nbins=100):
    # Get the particle with the lowest mistag for each event
    plt.figure()
    plt.yscale("log")
    plt.hist(df_TagParticles.loc[df_TagParticles.TagDec == -1].Eta ,bins = nbins , density = True , histtype = "stepfilled" ,range=(df_TagParticles.Eta.min(),0.5), color = "green" , alpha = 0.5, label = f"Tag. dec: b")
    plt.hist(df_TagParticles.loc[df_TagParticles.TagDec == 1].Eta ,bins = nbins , density = True , histtype = "stepfilled" ,range=(df_TagParticles.Eta.min(),0.5), color = "orange" ,  alpha = 0.5,label = f"Tag. dec: anti-b")
    plt.grid()
    plt.xlabel(r"$\eta$", fontsize=24)
    plt.ylabel("Normalized number of tracks", fontsize=24)
    #plt.annotate(f'{len(df_TagParticles)} tracks', xy=(0, 1), xycoords='axes fraction', fontsize=12, ha='left', va='top')
    plt.legend(loc = "best", title = f'{len(df_TagParticles)} total tracks')
    plt.title(f"{tagger}", fontsize=24)
    plt.savefig(f"{target_path}/mistag_VS_TagDec.pdf")
    plt.close()

def plot_tagDec_mistag(tagger, df_TagParticles, target_path, nbins=50):
    # Get the particle with the lowest mistag for each event
    plt.figure()
    plt.yscale("log")
    #plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == -1)&(df_TagParticles.label==0)].Eta ,bins = nbins , density = True , histtype = "step" ,range=(df_TagParticles.Eta.min(),0.5), edgecolor = "green" , linewidth=2, linestyle='--', label = f"Tag. dec: b, wrong")
    #plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == -1)&(df_TagParticles.label==1)].Eta ,bins = nbins , density = True , histtype = "step" ,range=(df_TagParticles.Eta.min(),0.5), edgecolor = "green" , linewidth=2, label = f"Tag. dec: b, correct")
    #plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == 1)&(df_TagParticles.label==0)].Eta ,bins = nbins , density = True , histtype = "step" ,range=(df_TagParticles.Eta.min(),0.5), edgecolor = "orange" , linewidth=2, linestyle='--', label = f"Tag. dec: anti-b, wrong")
    #plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == 1)&(df_TagParticles.label==1)].Eta ,bins = nbins , density = True , histtype = "step" ,range=(df_TagParticles.Eta.min(),0.5), edgecolor = "orange" , linewidth=2, label = f"Tag. dec: anti-b, correct")
    #plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == -1)&(df_TagParticles.label==0)].Eta ,bins = nbins , density = True , histtype = "stepfilled" ,range=(df_TagParticles.Eta.min(),0.5), color = "green" , alpha=0.5,  label = f"Tag. dec: b, wrong")
    plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == -1)&(df_TagParticles.label==0)].Eta ,bins = nbins , density = True , histtype = "step" ,range=(df_TagParticles.Eta.min(),0.5), edgecolor = "green" , linewidth=1, label = f"Tag. dec: b, wrong")
    
    plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == -1)&(df_TagParticles.label==1)].Eta ,bins = nbins , density = True , histtype = "stepfilled" ,range=(df_TagParticles.Eta.min(),0.5), color = "green" , alpha=0.5, label = f"Tag. dec: b, correct")
    #plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == 1)&(df_TagParticles.label==0)].Eta ,bins = nbins , density = True , histtype = "stepfilled" ,range=(df_TagParticles.Eta.min(),0.5), color = "orange" , alpha=0.5,  label = f"Tag. dec: anti-b, wrong")
    plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == 1)&(df_TagParticles.label==0)].Eta ,bins = nbins , density = True , histtype = "step" ,range=(df_TagParticles.Eta.min(),0.5), edgecolor = "orange" , linewidth=1, label = f"Tag. dec: anti-b, wrong")
    plt.hist(df_TagParticles.loc[(df_TagParticles.TagDec == 1)&(df_TagParticles.label==1)].Eta ,bins = nbins , density = True , histtype = "stepfilled" ,range=(df_TagParticles.Eta.min(),0.5), color = "orange" , alpha=0.5, label = f"Tag. dec: anti-b, correct")

    plt.grid()
    plt.xlabel(r"$\eta$",fontsize=24)
    plt.ylabel("Normalized number of tracks", fontsize=24)
    plt.legend(loc = "best", title = f'{len(df_TagParticles)} total tracks')
    plt.title(f"{tagger} mistag", fontsize=24)
    plt.savefig(f"{target_path}/TagDec_label.pdf")
    plt.close()


def calibration(tagger, df_tag, eventType, target_path, calibration_option='mistag'):

    #Calibration of the taggers and parameters saving
    import lhcb_ftcalib as ft

    taggers = ft.TaggerCollection()
    
    taggers.create_tagger(name = tagger, eta_data = df_tag.Eta.tolist(), dec_data = df_tag.TagDec.tolist(), B_ID = df_tag.B_TRUEID.tolist(),mode = 'Bu') # to be changed in mode = eventType[:2], B_ID = reconstructed ID when moving to data!
    
    if calibration_option=='logit':
        taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.logit))
    elif calibration_option=='mistag':
        taggers.set_calibration(ft.PolynomialCalibration(npar=2, link=ft.link.mistag)) 
    else:
        print('Not a valid calibration function')
    taggers.calibrate()

    # Plotting of calibration curves
    target_path = f'{target_path}/{calibration_option}'
    if os.path.isdir(f'{target_path}') == False:
        os.system(f"mkdir {target_path}")

    taggers.plot_calibration_curves(savepath = f'{target_path}', omega_range="minimal", nbins=10)
   
    info_dict = {"TaggingEfficiency" : taggers[tagger].stats.tagging_efficiency(calibrated = False),
    "TaggingPower" : taggers[tagger].stats.tagging_power(calibrated = False) ,
    "TaggingEfficiency_Cali" : taggers[tagger].stats.tagging_efficiency(calibrated = True), "TaggingPower_Cali" : taggers[tagger].stats.tagging_power(calibrated = True),
    "EffectiveMistag_Cali" : taggers[tagger].stats.effective_mistag(calibrated = True) , "EffectiveMistag" : taggers[tagger].stats.effective_mistag(calibrated = False) }
    with open(f"{target_path}/taggingInfo_{calibration_option}.json", "w") as f:
        json.dump(info_dict, f)
    print(f"Tagger parameters saved at {target_path}\n")
    print(f"Tagging information in a presentation-friendly format:\n")
    # Process the data
    processed_data = {key: propagate_and_round(value) for key, value in info_dict.items()}
    # Format the output
    formatted_data = {key: f"{values[0]} +- {values[1]}" for key, values in processed_data.items()}
    # Print the formatted data
    for key, value in formatted_data.items():
        print(f"{key}: {value}")

# Function to propagate and round the errors and values
def propagate_and_round(values):
    values = np.array(values) * 100  # Multiply all values by 100

    if len(values) > 2:  # For TaggingPower_Cali and EffectiveMistag_Cali
        combined_error = np.sqrt(np.sum(np.square(values[1:])))
        rounded_error = round(combined_error, -int(np.floor(np.log10(combined_error))))
        
        significant_digit = int(np.floor(np.log10(rounded_error)))
        rounded_value = round(values[0], -significant_digit)
        
        return [rounded_value, rounded_error]
    else:  # For other data
        max_error = max(values[1:])
        rounded_errors = [round(err, -int(np.floor(np.log10(max_error)))) for err in values[1:]]
        
        significant_digit = int(np.floor(np.log10(max_error)))
        rounded_value = round(values[0], -significant_digit)
        
        return [rounded_value] + rounded_errors

def print_taggingInfo(tag_file='taggingInfo.json'):
    # Read the data from the JSON file
    import json
    with open(tag_file, 'r') as file:
        info_dict = json.load(file)
    processed_data = {key: propagate_and_round(value) for key, value in info_dict.items()}
    # Format the output
    formatted_data = {key: f"{values[0]} +- {values[1]}" for key, values in processed_data.items()}
    # Print the formatted data
    for key, value in formatted_data.items():
        print(f"{key}: {value}")





