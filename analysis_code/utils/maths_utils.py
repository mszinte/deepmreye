def weighted_regression(x_reg,y_reg,weight_reg):
    """
    Function to compute regression parameter weighted by a matrix (e.g. r2 value).

    Parameters
    ----------
    x_reg : array (1D)
        x values to regress
    y_reg : array
        y values to regress
    weight_reg : array (1D) 
        weight values (0 to 1) for weighted regression

    Returns
    -------
    coef_reg : array
        regression coefficient
    intercept_reg : str
        regression intercept
    """

    from sklearn import linear_model
    import numpy as np
    
    regr = linear_model.LinearRegression()
    
    x_reg = np.array(x_reg)
    y_reg = np.array(y_reg)
    weight_reg = np.array(weight_reg)
    
    def m(x, w):
        return np.sum(x * w) / np.sum(w)

    def cov(x, y, w):
        # see https://www2.microstrategy.com/producthelp/archive/10.8/FunctionsRef/Content/FuncRef/WeightedCov__weighted_covariance_.htm
        return np.sum(w * (x - m(x, w)) * (y - m(y, w))) / np.sum(w)

    def weighted_corr(x, y, w):
        # see https://www2.microstrategy.com/producthelp/10.4/FunctionsRef/Content/FuncRef/WeightedCorr__weighted_correlation_.htm
        return cov(x, y, w) / np.sqrt(cov(x, x, w) * cov(y, y, w))

    
    x_reg_nan = x_reg[(~np.isnan(x_reg) & ~np.isnan(y_reg))]
    y_reg_nan = y_reg[(~np.isnan(x_reg) & ~np.isnan(y_reg))]
    weight_reg_nan = weight_reg[~np.isnan(weight_reg)]

    regr.fit(x_reg_nan.reshape(-1, 1), y_reg_nan.reshape(-1, 1),weight_reg_nan)
    coef_reg, intercept_reg = regr.coef_, regr.intercept_

    return coef_reg, intercept_reg

def r2_score_surf(bold_signal, model_prediction):
    """
    Compute r2 between bold signal and model. The gestion of nan values 
    is down with created a non nan mask on the model prediction 

    Parameters
    ----------
    bold_signal: bold signal in 2-dimensional np.array (time, vertex)
    model_prediction: model prediction in 2-dimensional np.array (time, vertex)
    
    Returns
    -------
    r2_scores: the R2 score for each vertex
    """
    import numpy as np
    from sklearn.metrics import r2_score
    
    # Check for NaN values in bold_signal
    nan_mask = np.isnan(model_prediction).any(axis=0)
    valid_vertices = ~nan_mask
    
    # Set R2 scores for vertices with NaN values to NaN
    r2_scores = np.full_like(nan_mask, np.nan, dtype=float)
    
    # Compute R2 scores for vertices without NaN values
    r2_scores[valid_vertices] = r2_score(bold_signal[:, valid_vertices], model_prediction[:, valid_vertices], multioutput='raw_values')
    
    return r2_scores