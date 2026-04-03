import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def remove_ip_fields(df):
    return df.drop(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'], axis=1)

def cap_numerical_data(df, numerical_cols, lower_bound=0, upper_bound=100000000):
    df_capped = df.copy()
    df_capped[numerical_cols] = df_capped[numerical_cols].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def min_max_log_norm(train_df, test_df, unlabeled_df, numerical_cols):
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    unlabeled_scaled = unlabeled_df.copy()
    
    train_scaled[numerical_cols] = np.log1p(train_scaled[numerical_cols])
    test_scaled[numerical_cols] = np.log1p(test_scaled[numerical_cols])
    unlabeled_scaled[numerical_cols] = np.log1p(unlabeled_scaled[numerical_cols])
    
    scaler = MinMaxScaler()
    
    scaler.fit(train_scaled[numerical_cols])
    
    train_scaled[numerical_cols] = scaler.transform(train_scaled[numerical_cols])
    test_scaled[numerical_cols] = scaler.transform(test_scaled[numerical_cols])
    unlabeled_scaled[numerical_cols] = scaler.transform(unlabeled_scaled[numerical_cols])
    
    return train_scaled, test_scaled, unlabeled_scaled, scaler


def keep_top_categorical_level(train_df, test_df, unlabeled_df, cat_cols, max_levels=32):
    train_cat = train_df.copy()
    test_cat = test_df.copy()
    unlabeled_cat = unlabeled_df.copy()
    
    top_categories_dict = {}
    
    for col in cat_cols:
        top_cats = train_cat[col].value_counts().nlargest(max_levels - 1).index.tolist()
        top_categories_dict[col] = top_cats        
        mask_func = lambda x: x if x in top_cats else 'Other'
        train_cat[col] = train_cat[col].apply(mask_func)
        test_cat[col] = test_cat[col].apply(mask_func)
        unlabeled_cat[col] = unlabeled_cat[col].apply(mask_func)
        
    return train_cat, test_cat, unlabeled_cat, top_categories_dict


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_encode_categorical(train_df, test_df, unlabeled_df, cat_cols):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[cat_cols])
    
    new_col_names = encoder.get_feature_names_out(cat_cols)
    
    def apply_encoding(df):
        encoded_data = encoder.transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded_data, columns=new_col_names, index=df.index)
        df_final = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)
        return df_final

    train_encoded = apply_encoding(train_df)
    test_encoded = apply_encoding(test_df)
    unlabeled_encoded = apply_encoding(unlabeled_df)
    
    return train_encoded, test_encoded, unlabeled_encoded, encoder