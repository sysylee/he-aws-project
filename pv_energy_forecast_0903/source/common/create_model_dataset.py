import numpy as np



def sliding_window(input_df, input_win_len):
    
    num_feature = len(input_df.columns)
    
    for idx in range(int(len(input_df) / input_win_len)):
        temp_array = np.array(input_df.iloc[idx * input_win_len: (idx + 1) * input_win_len, :]).reshape(1,input_win_len, num_feature)
        if idx == 0:
            result = temp_array
        else:
            result = np.concatenate((result, temp_array), axis = 0)
    
    return result



def split_data_set(input_df, val_start, ts_start, input_window):
    
    input_df.reset_index(drop = True, inplace = True)     
    
    train_df = input_df[input_df['base_time'] < ts_start] 
    test_df = input_df[input_df['base_time'] >= ts_start]    

    val_df = input_df[input_df['base_time']>= val_start]
    train_df = train_df[train_df['base_time'] < val_start]

    train_df.drop(columns=['base_time'], axis=1, inplace=True)
    val_df.drop(columns=['base_time'], axis=1, inplace=True)
    test_df.drop(columns=['base_time'], axis=1, inplace=True)
    

    train_df = sliding_window(train_df, input_window)
    val_df = sliding_window(val_df, input_window)
    test_df = sliding_window(test_df, input_window)
    
    return train_df, val_df, test_df



