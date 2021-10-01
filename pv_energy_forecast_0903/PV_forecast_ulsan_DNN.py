import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

from source.loaddata.datetime_utils import date_time_split, join_date_time, to_datetime, delay_time_hour, date_offset_hour,datetime_delay, create_full_time_df
from source.loaddata.feature_interface import change_feature_name
from source.AI_models.create_common_model import create_dnn_model
from source.common.create_model_dataset import split_data_set

# 데이터 로드 및 api 연결과 관련 클래스
class PVforecast:
    def __init__(self):
        self.data_path = ''
        self.energy = ''
        self.api = None
        self.weather_feature = ''

# 지역명이 들어간 변수명 다 수정 (unsla_fcst -> weather_fcst)
class PVforecast_ulsan_DNN(PVforecast):
    
    # init : 데이터 불러오기, 모델 생성 시 필요한 변수 선언, Feature 이름 mapping 
    def __init__(self):
        super(PVforecast_ulsan_DNN, self).__init__()
        self.data_path = './data/'
        self.site_id  = 'ulsan'
        self.model = 'DNN'
        self.model_path = './model/' + self.site_id + '/' +self.model +'/'
        
        # 분석에 필요한 데이터 불러오기
        self.energy = pd.read_csv(self.data_path + 'energy.csv')
        self.weather_fcst = pd.read_csv(self.data_path + 'ulsan_fcst_data.csv')
        self.model_info = pd.read_csv(self.data_path + 'model_info.csv')
        self.feature_mapping = pd.read_csv(self.data_path + 'feature_mapping.csv', index_col = 'index') # feature_mapping 불러올때 index 지정해야함        
        
        #  데이터 별 추후 분석에 사용할 feature 선언
        self.pv_feature = ['target_value']
        self.weather_feature = ['Temperature', 'Humidity', 'WindSpeed', 'WindDirection', 'Cloud']
        
        # 최종 featture 선언 (모델 생성 시 입력 feature / label column)
        self.input_feature = self.weather_feature
        self.output_feature = ['target_value']
        
        ####--------------------------------------#####        
        # model 생성시 필요한 변수 선언 (csv 파일 읽어와서 각 변수 선언하기, model_param은 모델 생성 시 입력변수로 들어감)
        self.model_info = self.model_info[(self.model_info['site_id'] == self.site_id) & (self.model_info['model'] == self.model)]
        self.model_info.reset_index(drop = True, inplace = True)
        self.model_info = self.model_info.to_dict()
        
        for key, value in self.model_info.items():
            temp_value = self.model_info[key][0]
            if type(temp_value) == str and '{' in temp_value: # dict 형태면 str -> dict 형태로 변환
                setattr(self,'%s'%key, literal_eval(temp_value)) 
            else:
                setattr(self,'%s'%key, self.model_info[key][0]) 
        
        # feature 이름 변경하기 (csv 파일 읽어와서 site_id 행을 common 행으로 column 이름 변경하기)
        self.energy = change_feature_name(self.energy, self.feature_mapping, 'pv', self.site_id)
        self.weather_fcst = change_feature_name(self.weather_fcst, self.feature_mapping, 'wea', self.site_id)
        
        print(self.energy.head())
        
        ####--------------------------------------#####

    # 학습 모델 생성을 위해 데이터 전처리 작업
    def preprocess(self): # 하기 코드 부터는 feature mapping된 이름으로 사용 (time -> issue_time 등)
    
        # 데이터 1 : 태양광 발전량 데이터 (target_value 사용)
        self.energy['issue_time'] = self.energy['issue_time'].apply(lambda x : datetime_delay(x, 1)) 
        self.energy['base_time'] = to_datetime(self.energy['issue_time'])
        self.energy[['target_value']] = self.energy[['target_value']].astype(float)
        self.energy = self.energy[['base_time', 'target_value']]
        
        # 하루 뒤 발전량 예측이 목적이기 때문에 time_shift 필요함  
        self.energy['target_value'] = self.energy['target_value'].shift(-24)
        self.energy['target_value'].reset_index(drop = True, inplace = True)
        # target_value의 경우 base_time 기준 예측 대상인 하루 뒤 발전량임 (shift 필요함)
        
        # 데이터 2: 동네예보 데이터 (input data로 사용)
        self.weather_fcst['issue_time'] = to_datetime(self.weather_fcst['issue_time'])
        self.weather_fcst['Predict time'] = self.weather_fcst['issue_time'] + self.weather_fcst['forecast'].map(date_offset_hour)
        
        self.weather_fcst['Humidity'] = self.weather_fcst['Humidity'] / 100
        self.weather_fcst['WindDirection'] = self.weather_fcst['WindDirection'] / 360
        
        self.weather_fcst_sort = self.weather_fcst.sort_values(['Predict time', 'forecast'], ascending=[True, False])
    
        
        if self.target_hour == 10: # 10시 예측 시 8시에 제공하는 기상데이터 사용
            self.fcst_hour = 8
            self.target_hour = [16, 19, 22, 25, 28, 31, 34, 37] # 해당 시간을 더해야 다음날 기상 데이터 생성됨
        else: # 17시 예측 시 14시에 제공하는 기상데이터 사용
            self.fcst_hour = 14
            self.target_hour = [10, 13, 16, 19, 22 , 25, 28, 31]
        
        # 추후 하기내용 확인 필요 (기상청에서 1시간 단위로 데이터 줌)
        
        # 10시 예측의 경우 8시에 제공한 기상데이터만 불러오기 (17시 예측의 경우 14시 제공한 기상데이터 불러오기)
        self.weather_fcst_remove = self.weather_fcst_sort[self.weather_fcst_sort['issue_time'].dt.hour == self.fcst_hour].reset_index(drop=True)        
        
        # 다음날 기상 데이터만 확보하기 위해 target_hour에 해당되는 기상데이터만 filter
        self.weather_fcst_remove = self.weather_fcst_remove[self.weather_fcst_remove['forecast'].isin(self.target_hour)].reset_index(drop=True)
        
        # 기상데이터의 경우 3시간 간격이기 때문에 1시간 간격 interpolation 하기 위헤 1시간 기준의 데이터 프레임 틀 만들기
        self.start_date = self.weather_fcst_remove.loc[0 , 'Predict time']
        self.end_date = self.weather_fcst_remove.loc[len(self.weather_fcst_remove)-1 , 'Predict time']
        
        self.wea_full_time_df = create_full_time_df(self.start_date, self.end_date, '1hour', 'Predict time') # 시작일 ~ 종료일 까지 1시간 간격의 dataframe 만들기 (column 명 : Predict time)
        
        self.weather_df = pd.merge(self.wea_full_time_df, self.weather_fcst_remove, on='Predict time', how = 'outer') # 기상예보 data랑 1시간 간격 dataframe join 하기
        self.weather_df['base_time'] = self.weather_df['Predict time'] - pd.DateOffset(days=1)
        
        
        self.weather_df = self.weather_df[['base_time'] + self.weather_feature]
        self.weather_df = self.weather_df.interpolate()
     
        # 분석 데이터 합치기 (날씨 데이터 + 발전량)
        self.base_df = pd.merge(self.energy, self.weather_df, on = 'base_time', how = 'outer')
        
        # null 존재 행 제거하기
        self.base_df.dropna(axis=0, inplace = True)
        self.base_df.reset_index(drop = True, inplace = True)
        
        self.input_df = self.base_df[['base_time'] + self.input_feature]
        self.input_df['hour'] = self.input_df['base_time'].astype(str).str.split(' ').str[1].astype(str).str.split(':').str[0].astype(int)
        self.output_df = self.base_df[['base_time'] + self.output_feature]
        
        # 날짜 기준으로 train / validation / test 데이터 구분하기 (base_time column 필요함)         
        self.train_df, self.val_df, self.test_df = split_data_set(self.input_df, self.val_start, self.ts_start, self.input_win_len)
        self.train_label, self.val_label, self.test_label = split_data_set(self.output_df, self.val_start, self.ts_start , self.input_win_len)

  
    def train_model(self):        
        
        model = create_dnn_model(self.model_param, self.input_win_len, self.output_win_len, self.train_df.shape[2])
        model.fit(self.train_df, self.train_label, epochs = self.epochs, batch_size = self.batch_size)

        model.save(self.model_path  + 'save_model')
    
        # 예측값 
        self.prediction = model.predict(self.test_df)
        self.prediction = np.abs(self.prediction)
        self.prediction = np.round(self.prediction, 0)
        self.predict_y = np.ravel(self.prediction)
        
        # test data 실제값
        self.test_y = np.ravel(self.test_label)

        over_idx = np.where(self.test_y > 500 * 0.1)
        accu = np.abs(self.predict_y[over_idx] - self.test_y[over_idx]) * 100 / 500
        accu = accu.sum(axis=0) / len(accu)
        print(accu)
                
        plt.figure()
        plt.plot(self.test_y)
        plt.plot(self.predict_y)
        
        
      
        

