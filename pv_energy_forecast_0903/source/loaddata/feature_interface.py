import pandas as pd
import numpy

def change_feature_name(df, mapping_info, target, site_id):
     
    col_list = list(mapping_info.columns)
    target_col_list = [temp for temp in col_list if target in temp]
    
    mapping_info = mapping_info[target_col_list]
    
    old_col = mapping_info.loc[site_id , :].to_dict()
    new_col = mapping_info.loc['common' , :].to_dict()
        
    for temp_list in target_col_list:
        df.rename(columns = {old_col[temp_list] : new_col[temp_list]}, inplace = True)
    
    return df