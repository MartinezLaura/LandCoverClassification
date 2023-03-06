__author__ = "Laura Martinez-Sanchez"
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Laura Martinez-Sanchez"
__email__ = "lmartisa@Gmail.com"



import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker


import pandas as pd
pd.options.mode.chained_assignment = None

from PIL import Image

import glob, os, sys

import plotly.graph_objects as go
import plotly.express as px
import plotly

import shutil
import json
import cv2
import pickle
import ast

import geopandas as gp

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split,  RandomizedSearchCV, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.multiclass import unique_labels


import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.data.transforms.presets.segmentation import test_transform
from gluoncv.utils.viz import get_color_pallete

def retrieve_ADE():
    """Retrieves a list of ADE20k classes.

    Returns:
    - list: A list of ADE20k classes.
    """

    ADECLASSES = ("wall", "building", "sky", "floor", "tree",
              "ceiling", "road", "bed", "window", "grass",
              "cabinet", "sidewalk",
              "person",
              "earth, ground", "door", "table", "mountain",
              "plant", "curtain",
              "chair", "car",
              "water", "painting","sofa", "shelf",
              "house", "sea", "mirror", "rug", "field", "armchair",
              "seat", "fence", "desk", "rock, stone", "wardrobe",
              "lamp", "bathtub", "railing", "cushion",
              "base", "box", "column", "sign",
              "dresser", "counter", "sand", "sink",
              "skyscraper", "fireplace", "refrigerator",
              "grandstand,", "path", "stairs", "runway",
              "case",
              "pool table", "pillow",
              "screen", "staircase", "river", "bridge",
              "bookcase", "blind", "coffee table",
              "toilet",
              "flower", "book", "hill", "bench", "countertop",
              "stove",
              "palm", "kitchen island",
              "computer"
              "electronic computer",
              "swivel chair", "boat", "bar", "arcade machine",
              "hovel",
              "bus"
              "passenger vehicle",
              "towel", "light", "truck", "tower",
              "chandelier", "awning",
              "streetlight", "kiosk",
              "tv"
              "goggle box",
              "airplane", "dirt track",
              "apparel",
              "pole", "soil",
              "bannister",
              "escalator",
              "ottoman",
              "bottle", "buffet",
              "poster",
              "stage", "van", "ship", "fountain",
              "conveyer belt",
              "canopy", "washer",
              "plaything", "swimming pool",
              "stool", "barrel", "basket", "waterfall",
              "shelter", "bag", "motorbike", "cradle",
              "oven", "ball", "food", "stair", "tank",
              "brand", "microwave",
              "flowerpot", "fauna",
              "cycle", "lake",
              "dishwasher",
              "screen",
              "blanket", "sculpture", "hood", "sconce", "vase",
              "traffic", "tray",
              "bin"
              "trash bin",
              "fan", "pier", "crt screen",
              "plate", "monitor", " notice board",
              "shower", "radiator", "glass", "clock", "flag")
    return ADECLASSES


def viz_segmentation(ds, predict, mask, img_path):
    """
    Takes the input image, the predicted mask, the ground truth mask, and the path to the image and visualizes 
    the segmentation. Returns the label, color, left, and height of the visualization.

    Args:
    - ds (str): Dataset name. Possible values are 'ade20k'.
    - predict (numpy.ndarray): Predicted segmentation mask.
    - mask (numpy.ndarray): Ground truth segmentation mask.
    - img_path (str): Path to the input image.

    Returns:
    - label (list): List of class labels.
    - col (list): List of RGB color values corresponding to each class label.
    - left (numpy.ndarray): Left coordinates of each bar in the visualization.
    - height (numpy.ndarray): Height of each bar in the visualization.
    """

    if ds == 'ade20k':
        ade20kclass = np.unique(predict)
        label = []
        [label.append(ADECLASSES[int(i)]) for i in ade20kclass]

        import gluoncv.utils.viz.segmentation as seg
        pallete = seg.adepallete

        left = np.array(range(len(label)))
        height = np.ones(len(label))

        col = []
        for i in ade20kclass:
            i = int(i)
            col.append([x / 255 for x in pallete[((i + 1) * 3):((i + 1) * 3 + 3)]])

    return label, col, left, height
    
def kfold(X, y, folds):
    """
    Splits the input data into train and test sets using stratified k-fold cross-validation.

    Args:
    - X (numpy.ndarray): Input features.
    - y (numpy.ndarray): Target variable.
    - folds (int): Number of folds for cross-validation.

    Returns:
    - kf (sklearn.model_selection.StratifiedKFold): Stratified k-fold cross-validation object.
    """
    kf = StratifiedKFold(n_splits=folds)
    kf.get_n_splits(X) 
    
    return kf
    
def split_dataset(df, field, name, path):
    """
    Splits the input dataset into train and test sets and saves the resulting arrays as a pickle file.

    Args:
    - df (pandas.DataFrame): Input dataset.
    - field (str): Target variable column name.
    - name (str): Name for the output pickle file.
    - path (str): path where to save the pickle with the datasets splits.
    
    Returns:
    - X_train (numpy.ndarray): Train set features.
    - X_test (numpy.ndarray): Test set features.
    """
    
    
    y = df[field].to_numpy()
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    X = df.id.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y)
    
    a = [X_train, X_test]
    
    with open('{}{}.pickle'.format(path, name), 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return X_train, X_test
    
def getlc(df, gt, field ='lc1'):
    """
    Check distibution of the data.
    Returns the input dataset with an additional column containing ground truth labels.

    Args:
    - df (pandas.DataFrame): Input dataset.
    - gt (pandas.DataFrame): Ground truth labels.
    - field (str): Ground truth label column name.

    Returns:
    - df (pandas.DataFrame): Input dataset with an additional column containing ground truth labels.
    """

    ids = gt['point_id'].tolist()
    boolean_series = df.id.isin(ids)
    df = df[boolean_series]
    df[field] = df.id.map(gt.set_index('point_id')[field])
    return df
    
def search_params(estimator, param_grid, search, X, y, cv):
    """
    This function performs hyperparameter tuning on a given estimator using either Grid Search or Random Search.

    Args:
    -----------
    - estimator(estimator object) This is the estimator object which should be a classifier or a regressor.
    - param_grid(: )dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
    - search(str):Type of search to perform. Should be either "grid" for Grid Search or "random" for Random Search.
    - X(array-like of shape(n_samples, n_features): Training input samples.
    - y(array-like of shape(n_samples,):  Target values (class labels in classification, real numbers in regression).
    - cv(int or cross-validation generator_: Determines the cross-validation splitting strategy, If int, the number of folds. Default is 5.

    Returns:
    --------
    - clf()estimator instance): The estimator instance fitted with the best hyperparameters.
    """
    
    try:
        if search == "grid":
            clf = GridSearchCV(
                estimator=estimator, 
                param_grid=param_grid, 
                scoring = {"Accuracy": make_scorer(accuracy_score)},
                n_jobs=28, 
                refit='Accuracy', 
                cv=cv, 
                verbose=1,
                return_train_score=True
            )
        elif search == "random":           
            clf = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=10,
                n_jobs=-1,
                cv=cv,
                verbose=0,
                random_state=1,
                return_train_score=True
            )
    except:
        print('Search argument has to be "grid" or "random"')
        sys.exit(0)
        
    # Fit the model
    clf.fit(X=X, y=y)
    
    return clf   
    
def perm_imp(X, y):
    """
    Compute the permutation feature importance for a given dataset using a random forest classifier.

    Args:
    -----------
    - X(pandas DataFrame): The input features for the model.
    - y(pandas Series): The target variable for the model.

    Returns:
    --------
    - tuple: A tuple of (result, lit), where `result` is the permutation feature importance result, 
            and `lit` is a list of feature names that have a non-zero importance.
    """
    
    # train
    rf = RandomForestClassifier()
    rf.fit(X, y)
    

    result = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=2)

    lit = pd.DataFrame(result.importances_mean, index =X.columns)
    lit = np.append(lit[lit[0]>0.0].index.values, 'id')
    # results.drop(results.columns.difference(lit), 1, inplace=True)
        
    return result, lit
    
def erase_empty_cols(results, THRESH = 0):
    """
    Removes columns from the input DataFrame that have all values equal to 0.
    
    Args:
    -----------
    - results(pandas.DataFrame): The input DataFrame to process.
    - THRESH(float, optional): A threshold value to replace all values below it with 0.

    Returns:
    --------
    - pandas.DataFrame: The processed DataFrame with empty columns removed.
    """

    results = results.loc[:, ~results.columns.str.contains('^Unnamed')]

    #see the main stast of the data
    df = results.copy()

    #drop the string columns
    df = df.drop(['lc'], axis=1)
    df = df.drop(['id'], axis=1)
    aux = df.describe().round()
    
    if THRESH!=0:
         df[df < THRESH] = 0

    #Drop all the cols that have min = 0 and max = 0 also in the results dataset
    droplist = [i for i in aux if aux[i].loc['max'] == 0 and aux[i].loc['min'] == 0]
    results.drop(droplist,axis=1,inplace=True)
    df.drop(droplist,axis=1,inplace=True)
    
    return results
    
def prepare_set(id_train, id_test, df, wet = False, check_fp = False):
    """
    Prepare the train and test sets for machine learning models.

    Args:
    -----------
    - id_train (list): A list of ids for the training set.
    - id_test (list): A list of ids for the test set.
    - df (pandas.DataFrame): The data frame containing the features and labels.
    - wet (bool, optional): Indicates whether the data includes WET objects. Defaults to False.
    - check_fp (bool, optional): Indicates whether to check for false positive objects in the test set. Defaults to False.

    Returns:
    --------
    - tuple: A tuple containing the training and test sets and their corresponding labels.
    """

    if wet:
        boolean_series = df.id.isin(id_train)
        X_train = df[boolean_series]
        X_train = X_train[X_train.lc != 'H']

        boolean_series = df.id.isin(id_test)
        X_test = df[boolean_series]
        X_test = X_test[X_test.lc != 'H']
        
    else:
        boolean_series = df.id.isin(id_train)
        X_train = df[boolean_series]
       
        boolean_series = df.id.isin(id_test)
        X_test = df[boolean_series]
        
    if check_fp:
        y_test = X_test['lc'].to_numpy()
        test = X_test['id'].reset_index(drop=True)
        X_test.drop(['lc', 'id'], axis=1, inplace = True)
        y_train = X_train['lc'].to_numpy()
        train = X_train['id'].reset_index(drop=True)
        X_train.drop(['lc', 'id'], axis=1, inplace = True)
        return X_train, X_test, y_train, y_test, test, train
    
    y_test = X_test['lc'].to_numpy()
    X_test.drop(['lc', 'id'], axis=1, inplace = True)
    y_train = X_train['lc'].to_numpy()
    X_train.drop(['lc', 'id'], axis=1, inplace = True)
    
    return X_train, X_test, y_train, y_test     
    
    
def plot_relation_feat_LC(results):
    """
    Plot the distribution of each object type feature for each Land Cover class in a bar chart.
    Also, generate a Sankey diagram to visualize the relationship between objects and Land Cover classes.

    Args:
    --------    
    - results (pd.DataFrame): a pandas DataFrame containing the data to be plotted.

    Returns:
    --------
    None
    """

    plt.rcParams["figure.figsize"] = (15,20)

    df = results.groupby(field).sum()
    df = df.drop(['id'], axis=1)
    df = df.loc[:, (df != 0).any(axis=0)].reset_index()
    df["lc"].replace({"A":"Artificial Land", "B":"Cropland", "C":"Woodland", "D":"Shrubland",
                        "E":"Grassland", "F":"Bare soil", "G":"Water areas", "H":"Wetland"}, inplace=True)

    df = df.melt(id_vars=[field], var_name="ObjType", value_name="Count")
    all_nodes = df.ObjType.values.tolist() + df[field].values.tolist()
    source_indices = [all_nodes.index(obj) for obj in df.ObjType]
    target_indices = [all_nodes.index(clas) for clas in df[field]]
    
    sns.barplot(x="lc", y="Count", hue="ObjType", data=df)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig('{}/Distrib-class-featrues.jpg'.format(path_plots))
    plt.show()
    

    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 20,
          thickness = 20,
          line = dict(color = "black", width = 1.0),
          label =  all_nodes,
        ),

        link = dict(
          source =  source_indices,
          target =  target_indices,
          value =  df.Count,
    ))])

#     fig.update_layout(title_text="Relation objects in the image and LC class",
#                       font_size=18, width=1000,height=2500)
    fig.show()

    plotly.offline.plot(fig, filename='{}sankey-class-featrues.html'.format(path_plots))    
    
    
    
def other_class(vals, col, path ):
    """
    Takes in an array of values `vals` and a corresponding array of colors `col`, 
    and returns a modified `col` array where values that contribute to less than
    5% of the total sum are grouped together into a single "Other" category.
    
    Args:
    --------    
    - vals (array-like): An array of numerical values.
    - col (array-like): An array of colors corresponding to the values in `vals`.
    
    Returns:
    --------    
    - tuple: A tuple containing two arrays: the modified `col` array and an array of the non-grouped values from `vals`.
    """
    coll = [col[i] for i in np.argsort(vals)]

    sort_val = []
    other = 0
    for i in np.sort(vals):
        if (other+i) < (np.sum(vals)*0.05):
            other = other + i
        else:
            sort_val.append(i)
    
    if other != 0:
        to_erase = list(range(len(vals)-len(sort_val)))
        sort_val = np.delete(np.sort(vals), to_erase, axis=0)
        sort_val = np.append(sort_val, other)
        coll = coll[len(vals)-len(sort_val)+1:]
        coll.append((0,0,0))# append at the begining

    return coll, sort_val    
    
def plot_falsep(path_images, f, proba, test, pred):
    
     """
    Plots the segmentation of an input image using a pre-trained model, and extracts features and areas from the predicted
    segmentation to generate a boxplot.

    Args:
    -----------
    - path_images(str): Path to the input image file
    - test(str): Ground truth label for the input image
    - pred(str): Predicted label for the input image
    - proba(numpy.ndarray): Array containing probability data for different classes

    Returns:
    --------
    None
    """

    plt.rcParams["figure.figsize"] = (10,7)
    ctx = mx.cpu(0)
    ade = retrieve_ADE()
    img = image.imread(path_images)

    modelname = os.path.basename(f).split('-')[1].split('.')[0]
    temp = test_transform(img, ctx)

    model = gluoncv.model_zoo.get_model(modelname, pretrained=True)
    output = model.predict(temp)
    predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

    df = pd.DataFrame(columns=['Area','Feature'])
    df['Area'] = df['Area'].astype("float")
    df['Feature'] = df['Feature'].astype("string")

    mak = get_color_pallete(predict, 'ade20k')
    #plotiing!
    labs, col, left, height = viz_segmentation('ade20k', predict, mak, path_images)
    (lab, vals) = np.unique(mak, return_counts=True)
    
    for cla in np.unique(predict):
        lab = ade[int(cla)+1]
        mask = predict.astype(np.uint8)
        mask[mask != cla] = 0
        mask[mask == cla] = 1
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for idx in range(len(contours)):
            area = cv2.contourArea(contours[idx], oriented = False) 
            df = df.append({'Area': area, 'Feature':lab}, ignore_index=True)
            
    
    '''plo'''        
    print('Start plot!')
    plt.rc('xtick', labelsize=8) 
    plt.rc('ytick', labelsize=8) 

    aux = mpimg.imread(path_images)
    gs = gridspec.GridSpec(8,8)
    plt.title('The truth values is: {} but has been classified as: {}'.format(test, pred), fontsize=16)
    ax1 = plt.subplot(gs[0:5, 0:2])
    ax1.imshow(aux)
    # ax1.imshow(mak, alpha=0.5)
    ax1.axis('off')


    ax2 = plt.subplot(gs[0:5, 3:7])
    ax2.set_xlabel("X Label",fontsize=8)
    ax2.set_ylabel("Y Label",fontsize=8)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2 = sns.boxplot(x="Area", y="Feature", data=df, orient = 'h', palette="Set3").set(xscale="log")


    ax3 = plt.subplot(gs[6:7, :]) 
    category_names = ["Artificial Land", "Cropland", "Woodland", "Shrubland",
                           "Grassland", "Bare soil", "Water areas"]
    data = proba
    data_cum = data.cumsum(axis=0)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[0]))

    ax3.invert_yaxis()
    ax3.xaxis.set_visible(False)
    ax3.set_xlim(0, np.sum(data))
    

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[i]
        starts = data_cum[i] - widths
        ax3.barh('Probability', widths, left=starts, height=0.01,
                    label=colname, color=color)
        xcenters = starts + widths / 2
        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax3.text(xcenters, 0, str(widths), ha='center', va='center',
                        color=text_color, fontsize=8)

    ax3.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                  loc='lower left', fontsize=8)

    plt.savefig("../Plots/nomask/plot-{}{}-{}".format(test, pred, os.path.splitext(path_images)[0].split('/')[-1]))
        
def func(pct, allvals):
    """
    This function calculates the percentage contribution of 'pct' in the total sum of 'allvals' and returns
    a formatted string with the percentage rounded off to one decimal place.
    
    Args:
    --------       
    - pct (numerical): A numerical value representing the portion of the total sum of 'allvals' for which we want to calculate the percentage.
    - allvals (array-like): A list or array of numerical values representing the total sum from which we want to calculate the percentage.

    Returns:
    -------- 
    - string: A formatted string with the percentage contribution of 'pct' in the total sum of 'allvals' rounded off to one decimal place.

    """
    absolute = (pct*100)/np.sum(allvals)
    
    return "{:} %\n".format(round(pct, 1))

    
def pie_class_plot(path_plots, results, path_images):
        """
    Plots the mask and image for each class and type of model to see differences,
    and saves the resulting pie charts in the specified directory.

    Args:
    - path_plots (str): Directory to save the resulting pie charts.
    - results (pandas.DataFrame): DataFrame with results of the model predictions.
    - path_images (str): Directory with the images used for the predictions.
    
    Returns:
    --------
    None
    """
    #Plot the mask and image for each class and type of model to see differences
    plt.rcParams["figure.figsize"] = (10,5)
    ctx = mx.cpu(0)

    df = results.groupby(['lc'])['id'].apply(np.random.choice).reset_index()
    print(df['id'])
    modelname = os.path.basename(final_path_results).split('-')[1].split('.')[0]
    print(modelname)

    for i in df['id']:
        img = image.imread('{}{}P.jpg'.format(path_images, int(i)))

        img = test_transform(img, ctx)

        model = gluoncv.model_zoo.get_model(modelname, pretrained=True)
        output = model.predict(img)
        predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()
        mask = get_color_pallete(predict, 'ade20k')

        label, col, left, height = viz_segmentation('ade20k', predict, mask, '{}{}P.jpg'.format(path_images, int(i)))
        print('{}{}P.jpg'.format(path_images, int(i)))
        (lab, vals) = np.unique(mask, return_counts=True)

        im = mpimg.imread('{}{}P.jpg'.format(path_images, int(i)))

        gs = gridspec.GridSpec(1, 3, width_ratios=[3,2,0.40]) 
        ax0 = plt.subplot(gs[0, 0])
        ax1 = plt.subplot(gs[0, 1])
        ax2 = plt.subplot(gs[0, 2])

        ax0.imshow(im)
        ax0.imshow(mask, alpha=0.5)
        ax0.axis('off')

        #handel othersclass of the pie chart
        coll, sort_val = utils.other_class(vals, col)

        wedges, texts, autotexts = ax1.pie(sort_val, autopct=lambda pct: func(pct, sort_val),
                                          colors = coll, textprops=dict(color="w"))

        ax2.barh(left, height, tick_label=label, align="center", color=col)
        ax2.tick_params(axis='y', labelsize=8, direction='inout',labelleft = False, labelright = True)
        ax2.get_xaxis().set_visible(False)

        plt.savefig('{}Pie-{}.jpg'.format(path_plots, int(i)))
        plt.show()

def create_dataframe(cols, images_list):
    """
    Creates a new Pandas DataFrame with columns as specified in cols argument and the number
    of rows as the length of the images_list argument. Adds two additional columns, "id" and "lc",
    initialized with zeros.

    Args:
    - cols(list): A list of strings representing the names of columns to be created.
    - images_list(list): A list of image file names.

    Returns:
    - d(DataFrame): A new Pandas DataFrame with columns as specified in cols argument
    and the number of rows as the length of the images_list argument.
    """
    d = pd.DataFrame(0, index=np.arange(len(images_list)), columns=cols)
    d["id"] = 0
    d['lc'] = 0
    
    return d    
    
    
    
    
    
    
    

