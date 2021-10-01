# Day-Ahead PV Estimation
## Project Name: AWS기반 신재생 발전량 예측 및 설비 이상탐지 기술 개발
## PowerICT Lab., Hyundai-Electric
## August. 24. 2021, Version 0.0.1
### 데이터 폴더 구성
+ data
	- dangin_fcst_data: 당진 사이트 동네예보
	- dangjin_obs_data: 당진 발전소 주위 예보
	- energy: 당진, 울산 사이트 발전량
	- site_info: 당진, 울산 사이트 태양광 설치 정보
	- ulsan_fcst_data: 울산 사이트 동네예보
	- ulsan_obs_data: 울산 발전소 주위 예보
+ source
	+ common
		- __init__.py
		- dilated_regression_cnn.py
	+ loaddata
		- __init__.py
		- datetime_utils.py
	+ preprocess
		- __init__.py
		- 
	+ AI_models
		- __init__.py
		- 
	+ postprocess
		- __init__.py
		- 
	+ visualization
		- __init__.py
		- 
	- __init__.py
	- __version__.py
+ Readme.md
+ main.py
+ PV_forecast_ulsan.py
+ requirements.txt


