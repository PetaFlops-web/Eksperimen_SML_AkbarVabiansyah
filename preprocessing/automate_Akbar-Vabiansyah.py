import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing():
    
    raw_data_path = 'dataset_raw.csv'
    output_folder = os.path.join('preprocessing/dataset_preprocessing')
    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        return

    df_clean = df.drop(columns=['ID', 'Gender'], errors='ignore')
    
    if 'Discount_offered' in df_clean.columns:
        q_limit = df_clean['Discount_offered'].quantile(0.95)
        df_clean['Discount_offered'] = df_clean['Discount_offered'].clip(upper=q_limit)
    
    importance_mapping = {'low': 1, 'medium': 2, 'high': 3}
    if 'Product_importance' in df_clean.columns:
        df_clean['Product_importance'] = df_clean['Product_importance'].str.lower().map(importance_mapping)
    
    df_encoded = pd.get_dummies(df_clean, columns=['Mode_of_Shipment', 'Warehouse_block'], drop_first=False)
    
    bool_cols = df_encoded.select_dtypes(include=['bool']).columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)
    
    target_col = 'Reached.on.Time_Y.N'
    if target_col not in df_encoded.columns:
        print(f"Error: Kolom target '{target_col}' tidak ada di data. Cek dataset mentahmu.")
        return

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    cols_to_scale = ['Customer_care_calls', 'Cost_of_the_Product', 
                     'Prior_purchases', 'Discount_offered', 'Weight_in_gms', 
                     'Customer_rating', 'Product_importance']
    cols_to_scale = [c for c in cols_to_scale if c in X_train.columns]
    
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_path = os.path.join(output_folder, 'train_clean.csv')
    test_path = os.path.join(output_folder, 'test_clean.csv')
    
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    
if __name__ == "__main__":
    run_preprocessing()