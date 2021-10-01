import os
import logging
import PV_forecast_ulsan_DNN 



if __name__ == "__main__":
    #--------------site1 _algorithm1--------------
    PV_ulsan = PV_forecast_ulsan_DNN.PVforecast_ulsan_DNN()
    PV_ulsan.preprocess()
    PV_ulsan.train_model()
    
