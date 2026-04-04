import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def remove_ip_fields(df):
    return df.drop(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'], axis=1)

def cap_numerical_data(df, numerical_cols, lower_bound=0, upper_bound=100000000):
    df_capped = df
    df_capped[numerical_cols] = df_capped[numerical_cols].clip(lower=lower_bound, upper=upper_bound)
    return df_capped

def min_max_log_norm(train_df, test_df, numerical_cols):
    train_scaled = train_df
    test_scaled = test_df
    
    # Trasformazione logaritmica (log1p gestisce bene gli zeri)
    train_scaled[numerical_cols] = np.log1p(train_scaled[numerical_cols])
    test_scaled[numerical_cols] = np.log1p(test_scaled[numerical_cols])
    
    scaler = MinMaxScaler()
    
    scaler.fit(train_scaled[numerical_cols])
    
    train_scaled[numerical_cols] = scaler.transform(train_scaled[numerical_cols])
    test_scaled[numerical_cols] = scaler.transform(test_scaled[numerical_cols])
    
    return train_scaled, test_scaled, scaler


def keep_top_categorical_level(train_df, test_df, cat_cols, max_levels=32):
    top_categories_dict = {}
    
    for col in cat_cols:
        # Assicuriamoci che siano stringhe
        train_df[col] = train_df[col].astype(str) 
        test_df[col] = test_df[col].astype(str)
        
        # Trova le categorie top
        top_cats = train_df[col].value_counts().nlargest(max_levels - 1).index.tolist()
        top_categories_dict[col] = top_cats        
        
        # Sostituzione vettorizzata (NO apply)
        train_df[col] = np.where(train_df[col].isin(top_cats), train_df[col], 'Other')
        test_df[col] = np.where(test_df[col].isin(top_cats), test_df[col], 'Other')
        
    return train_df, test_df, top_categories_dict

def one_encode_categorical(train_df, test_df, cat_cols):
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
    
    return train_encoded, test_encoded, encoder


from sklearn.preprocessing import OrdinalEncoder
import numpy as np

def ordinal_encode_categorical(train_df, test_df, cat_cols):
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    train_encoded = train_df
    test_encoded = test_df
    
    train_encoded[cat_cols] = encoder.fit_transform(train_encoded[cat_cols])
    test_encoded[cat_cols] = encoder.transform(test_encoded[cat_cols])
    
    train_encoded[cat_cols] = train_encoded[cat_cols].astype(np.int32)
    test_encoded[cat_cols] = test_encoded[cat_cols].astype(np.int32)
    
    return train_encoded, test_encoded, encoder