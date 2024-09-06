import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def onehot_encoder(X, va_name):
    X_oh = pd.get_dummies(X[[va_name]], drop_first = False, dummy_na = False)
    rest_names = [name for name in X.columns.tolist() if name != va_name]
    Xt = pd.concat([X[rest_names], X_oh], axis=1)
    return Xt

def ordinal_encoder(X, va_name, cat_name_list):
    va_str = X.loc[:, va_name].astype(object).values[:, np.newaxis]
    va_num = OrdinalEncoder(categories=[cat_name_list], handle_unknown = 'use_encoded_value',  unknown_value = np.nan).fit_transform(va_str)
    Xt = X.drop(va_name, axis=1)
    Xt.loc[:, va_name] = va_num
    Xt = Xt.loc[:, X.columns.tolist()]
    return Xt

def ENCODER(X):
    X = ordinal_encoder(X, 'T category', ['T1','T2','T3'])
    X = ordinal_encoder(X, 'N category', ['N0','N1'])
    X = ordinal_encoder(X, 'M category', ['M0','M1'])
    X = ordinal_encoder(X, 'AJCC stage', ['I','II','III','IV'])
    X = ordinal_encoder(X, 'Extent', ['CTO','DM/PE'])
    X = ordinal_encoder(X, 'Grade', ['G1','>G1'])
    X = ordinal_encoder(X, 'Hysterectomy', ['No','Yes'])
    X = ordinal_encoder(X, 'Chemotherapy', ['No/Unknown','Yes'])
    
    X = onehot_encoder(X, 'Surgery')
    X = onehot_encoder(X, 'Radiotherapy')
    return X

def preprocessor_test(X_test, encoder, imputer_, scaler_):
    X_test_encode = encoder(X_test)

    X_test_impute = imputer_.transform(X_test_encode)
    X_test_impute = pd.DataFrame(X_test_impute, columns = X_test_encode.columns)

    X_test_scale = scaler_.transform(X_test_impute)
    X_test_scale = pd.DataFrame(X_test_scale, columns = X_test_impute.columns)
    
    return X_test_scale