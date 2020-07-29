def train_models(
    estimator, params, X, y,
    k_folds=5, scoring='neg_root_mean_squared_error', fast_train=False,
    verbose_level=0
):

    '''
    This function takes a given estimator with lists of values of hyperparameters and
    performs a gridsearch with cross validation. This function may also be cancelled
    mid operation with a KeyboardInterupt (kernal interupt). When this happens, all
    estimators with their respective hyperparameters that were not trained will be discarded.

    Inputs:
        estimator: The estimator wanting to be tuned.
        params: The dictionary of hyperparmeters to be parsed into the estimator.
                e.g.: {
                        'n_estimators': range(35, 56),
                        'max_depth': range(10, 21),
                        'min_samples_split': range(10, 21),
                        'random_state': [0]
                    }
        X: The array training input samples.
        y: The array target values.
        k_folds: The number of folds to be performed in cross validation.
        scoring: The scorer to be used in both training and scoring of estimators.
                 To read more, see here:
                    https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
        fast_train: True will train models using 80% of the input data and validate with the remaining 20%.
                    False will use cross validation, this will be slower as the model will be trained and
                    compared more than one time. RMSE will be used as the accuracy metric.
        verbose_level: The level of verbosity to be outputed in cross validation.
                       0 = Output nothing
                       1 = Output some level of detail
                       2+ = Output all details

    Returns:
        models: A pandas.DataFrame of all hyperparameters and the corresponding score obtained.
    '''

    # IMPORTS
    from pandas import DataFrame
    import itertools
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error


    # list(itertools.product(*params.values()))
    # This gets all combinations of all paramaters and makes a tuple of each
    models = DataFrame(
        data=list(itertools.product(*params.values())),
        columns=list(params.keys())
    )

    scores = []

    if fast_train:
        # If not using cross validation, use train/test split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=0)

    for model_num, hyper_params in enumerate(models.to_dict(orient='records')):
        # models.to_dict(orient='records') converts all rows into a dictionary
        # of arguments to parse into an estimator
        try:
            if fast_train:
                model = estimator(**hyper_params)
                model.fit(X=X_train, y=y_train)
                predictions = model.predict(X_valid)

                if scoring == 'neg_mean_absolute_error':
                    model_score = mean_absolute_error(predictions, y_valid)
                elif scoring == 'neg_root_mean_squared_error':
                    model_score = mean_squared_error(predictions, y_valid, squared=False)
                else:
                    # If parsing in custom function
                    model_score = scoring(predictions, y_valid)

            else:
                # Train model with cross validation
                model_score = -cross_val_score(
                    estimator=estimator(**hyper_params),
                    X=X, y=y,
                    scoring=scoring, cv=k_folds,
                    verbose=verbose_level, error_score='raise'
                ).mean()

            scores.append(model_score)
            printProgressBar(model_num + 1, len(models), prefix='Progress:', suffix='Complete', length=50)

        except KeyboardInterrupt:
            # This is here so that if this operation is canceled, all the
            # data will not be lost.
            printProgressBar(model_num + 1, len(models), prefix='Progress:', suffix='Model Stopped!', length=50)
            break

    idx_not_calced = (len(scores) - 1)
    models = models.drop(list(range(len(scores), len(models))))
    models['Score'] = scores

    return models

def train_unsupervised_models(
    estimator, params, data, scoring,
):

    '''
    This function takes a given estimator with lists of values of hyperparameters and
    performs a gridsearch. This function may also be cancelled
    mid operation with a KeyboardInterupt (kernal interupt). When this happens, all
    estimators with their respective hyperparameters that were not trained will be discarded.

    Inputs:
        estimator: The estimator wanting to be tuned.
        params: The dictionary of hyperparmeters to be parsed into the estimator.
                e.g.: {
                        'n_estimators': range(35, 56),
                        'max_depth': range(10, 21),
                        'min_samples_split': range(10, 21),
                        'random_state': [0]
                    }
        data: Dataframe to perform unsupervised learning on.
        scoring: The scorer to be used in both training and scoring of estimators.
                 To read more, see here:
                    https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values

    Returns:
        models: A pandas.DataFrame of all hyperparameters and the corresponding score obtained.
    '''

    # IMPORTS
    from pandas import DataFrame
    import itertools
    from sklearn.model_selection import train_test_split, cross_val_score


    # list(itertools.product(*params.values()))
    # This gets all combinations of all paramaters and makes a tuple of each
    models = DataFrame(
        data=list(itertools.product(*params.values())),
        columns=list(params.keys())
    )

    scores = []


    for model_num, hyper_params in enumerate(models.to_dict(orient='records')):
        # models.to_dict(orient='records') converts all rows into a dictionary
        # of arguments to parse into an estimator
        try:
            model = estimator(**hyper_params)
            model.fit(data)

            if scoring == 'kmeans':
                model_score = model.inertia_
            scores.append(model_score)
            printProgressBar(model_num + 1, len(models), prefix='Progress:', suffix='Complete', length=50)

        except KeyboardInterrupt:
            # This is here so that if this operation is canceled, all the
            # data will not be lost.
            printProgressBar(model_num + 1, len(models), prefix='Progress:', suffix='Model Stopped!', length=50)
            break

    idx_not_calced = (len(scores) - 1)
    models = models.drop(list(range(len(scores), len(models))))
    models['Score'] = scores

    return models


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Code taken from here:
        https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/34325723#34325723
    Credit To:
        Greenstick

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def check_coerce_problems(series, dtype, print_values=True):
    '''
    Checks all values to see if they can be coerced into a dtype. Values that cannot are flagged and can be printed.
    '''
    
    bad_values = []
    
    for value in series:
        if dtype == 'numeric':
            try:
                _ = int(value)
            except ValueError:
                bad_values.append(value)
                
                if print_values:
                    print(f'Error coercing: {value}')
        else:
            raise NotImplementedError
            
    return bad_values


def ridgeline(data, overlap=0, fill=True, labels=None, n_points=150, bw_method=0.5):
    """
    https://glowingpython.blogspot.com/2020/03/ridgeline-plots-in-pure-matplotlib.html
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    from scipy.stats.kde import gaussian_kde
    from scipy.stats import norm
    import numpy as np
    import matplotlib.pyplot as plt

    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0, 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    curves = []
    ys = []
    for i, d in enumerate(data):
        pdf = gaussian_kde(d, bw_method)
        y = i*(1.0-overlap)
        ys.append(y)
        curve = pdf(xx)
        if fill:
            plt.fill_between(xx, np.ones(n_points)*y, 
                             curve+y, zorder=len(data)-i+1, color=fill)
        plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
    if labels:
        plt.yticks(ys, labels)