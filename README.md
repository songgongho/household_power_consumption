# household_power_consumption

전력 수요 예측 모델 개발


다양한 딥러닝 모델을 활용한 전력 수요 예측 모델 구현
주어진 텍스트파일(가정용 전기 사용량 시계열 데이터)를 활용하여 전력 수요 예측 모델 개발
   * 데이터셋 (Individual Household Electric Power Consumption)
   다운로드 링크 : https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
Global_active_power 컬럼을 활용하고, Training 80%, Test 20%으로 구성 
   (데이터셋을 읽어오고, 활용하는 방안은 read_data.py 참고)
총 4년 간의 데이터 중 Training 및 Test로 활용할 시간 구간은 자체 선정 가능
수업에서 배운 예측 모델 간 성능을 정량적으로 비교하고 그 이유를 분석
<img width="1692" height="492" alt="image" src="https://github.com/user-attachments/assets/e55095e8-8b3c-4b67-a94d-fa15f95339ad" />
