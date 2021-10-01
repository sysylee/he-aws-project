import pandas as pd

# 날짜 시간 항목을 날짜, 시간으로 구분함
def date_time_split(datetime):
    date, time = datetime.split(' ')
    return date, time

# 시간 항목에서, 해당 값만큼 딜레이 함. 예: 2:30:00 --> 1:30:00
def delay_time_hour(time, delay):
    hour, minutes, seconds = time.split(':')
    hour = str(int(hour)-delay)
    time = ':'.join([hour, minutes, seconds])
    return time

# 날짜, 시간 데이터를 다시 합침
def join_date_time(date, time):
    datetime = ' '.join([date, time])
    return datetime

# pandas datetime으로 변환
def to_datetime(datetime):
    datetime = pd.to_datetime(datetime)
    return datetime

def date_offset_hour(datetime):
    return pd.DateOffset(hours=datetime)

def rolling_mean(energy, day):
    window = day * 24
    roll_mean = energy.rolling(window=window).mean()
    return roll_mean


def datetime_delay(datetime, delay):
    date, time = date_time_split(datetime)
    time = delay_time_hour(time, delay)
    datetime = join_date_time(date, time)
    return datetime

def create_full_time_df(start_date, end_date, resolution, col_name): 
    if resolution =='15min':
        freq = resolution
    elif resolution == '5min':
         freq = resolution     
    elif resolution == '1hour':
        freq = '1H'    
    else:
        freq = '1H' # default를 1시간으로 설정
    
    full_time_list = pd.date_range(start = start_date, end = end_date, freq = freq)
    result =  pd.DataFrame(full_time_list, columns = [col_name])

    return result