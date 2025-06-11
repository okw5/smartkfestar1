import joblib
import pandas as pd
import numpy as np
import re
import sys
import json

import traceback # 상세한 오류 로깅을 위해 추가

# Python 3.7+ 에서 stdout/stderr 인코딩을 UTF-8로 설정 시도
# (깨진 한글 로그 출력을 방지하기 위함)
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr.encoding != 'utf-8':
        sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    # reconfigure가 없는 구형 Python 버전일 경우 무시
    pass
except Exception:
    # 다른 예외 발생 시 무시 (예: IOBase에는 reconfigure 없음)
    pass

# --- 파일 경로 설정 ---
# 서버 구동 시 model 폴더 하단에 이 파일과 모델/전처리기가 있다고 가정합니다.
# 실제 경로에 맞게 수정해주세요.
MODEL_PATH = 'best_model.joblib'
PREPROCESSOR_PATH = 'preprocessor.joblib'
PCA_TRANSFORMER_PATH = 'pca_transformer.joblib'
FREQ_MAPS_PATH = 'address_freq_maps.joblib'
# 원본 엑셀 파일은 '서울과거리' 및 장소 관련 피처 조회를 위해 필요합니다.
EXCEL_DATA_PATH = '축제_데이터셋_업로드용.xlsx'

# --- 전역 변수로 모델 및 전처리기, 엑셀 데이터 로드 ---
# 스크립트 로드 시점에 미리 로드하여 예측 요청마다 다시 로드하지 않도록 합니다.
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    pca_transformer = joblib.load(PCA_TRANSFORMER_PATH)
    address_freq_maps = joblib.load(FREQ_MAPS_PATH)
    # 원본 엑셀 데이터 로드 (조회용)
    excel_df = pd.read_excel(EXCEL_DATA_PATH)

    # 예산 텍스트 처리 → 숫자형 변환 (노트북 전처리 재현)
    excel_df['예산\n(백만)'] = excel_df['예산\n(백만)'].replace({'미확정': 0, '무응답': 0}).astype(float)

    # 거리 숫자 추출 (노트북 전처리 재현)
    # NaN 값을 먼저 처리하거나, regex 추출 후 float 변환 시 오류 방지
    excel_df['서울과거리'] = excel_df['서울과거리'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

    # 장소 관련 열 결측치 처리 (노트북 전처리 재현)
    PLACE_COLS = ['공원/유원지', '관광농원/허브마을', '일반관광지', '동식물원', '먹거리/패션거리',
                  '산림욕장/휴향림/수목원', '폭포/계곡/호수/저수지', '해수욕장']
    excel_df[PLACE_COLS] = excel_df[PLACE_COLS].fillna(0)

    print("모델, 전처리기, PCA 변환기, 주소 빈도 맵, 엑셀 데이터를 성공적으로 불러왔습니다.", file=sys.stderr)
except FileNotFoundError as e:
    print(f"오류: 필요한 파일 중 일부를 찾을 수 없습니다. 경로를 확인하세요: {e}", file=sys.stderr)
    sys.exit(1) # 스크립트 종료
except Exception as e:
    print(f"파일 로드 중 오류 발생: {e}", file=sys.stderr)
    sys.exit(1) # 스크립트 종료

# --- Define feature_order (Crucial for preprocessor) ---
# This list MUST match the feature names and order expected by your
# preprocessor and model, as determined during training in your notebook.
# Based on your notebook: numeric_cols + categorical_cols
PREPROCESSOR_FEATURE_ORDER = [
    '예산\n(백만)', '서울과거리', '1인당예산', '장소_PCA', '주소점수', # Numeric features from notebook
    '축제유형_대분류', '계절' # Categorical features from notebook
]

# --- 노트북의 전처리 함수 재현 ---

def clean_type(x):
    """
    축제유형 문자열을 정제하여 대분류로 매핑합니다. (노트북의 clean_type 함수와 동일 로직)
    """
    if not isinstance(x, str):
        return '기타'

    s = x.lower()
    s = s.replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s)
    s = s.strip()

    if '문화' in s or '예술' in s: return '문화예술'
    elif '전통' in s or '역사' in s: return '전통역사'
    elif '생태' in s or '자연' in s or '환경' in s: return '생태자연'
    elif '특산' in s or '먹거리' in s or '농산' in s or '수산' in s: return '특산물'
    elif '주민화합' in s or '시민화합' in s or '화합' in s: return '주민화합'
    elif '관광' in s: return '관광'
    elif '기타' in s: return '기타'
    elif '체험' in s: return '체험행사'
    else: return '기타'

def get_additional_features(si, gungu, dong, excel_data):
    """
    주어진 지역(시, 군구, 법정동)의 가장 최근 '시작년' 데이터를 기준으로
    8개의 장소 관련 피처와 '서울과거리'를 '축제_데이터셋_업로드용.xlsx'에서 조회합니다.
    반환: (8개 장소 피처 리스트, 서울과거리 값) 또는 데이터 없는 경우 ([0.0]*8, 0.0)
    """
    location_filter = (excel_data['시'] == si) & \
                      (excel_data['군구'] == gungu) & \
                      (excel_data['법정동'] == dong)
    relevant_data = excel_data[location_filter]

    features_to_fetch = PLACE_COLS + ['서울과거리']
    num_place_features = len(PLACE_COLS)

    if relevant_data.empty:
        # print(f"경고: {si} {gungu} {dong}에 대한 과거 데이터를 찾을 수 없습니다. 추가 피처는 0으로 설정됩니다.", file=sys.stderr)
        return [0.0] * num_place_features , 0.0
    else:
        # '시작년' 기준으로 내림차순 정렬하여 가장 최근 데이터 선택
        most_recent_data = relevant_data.sort_values(by='시작년', ascending=False).iloc[0]

        place_feature_values = []
        seoul_distance_value = 0.0

        # 장소 피처 추출
        for feature_name in PLACE_COLS:
            # 결측치는 로드 시점에 0으로 채워졌다고 가정
            place_feature_values.append(most_recent_data.get(feature_name, 0.0))

        # 서울과거리 피처 추출
        seoul_distance_feature_name = '서울과거리'
        # 결측치는 로드 시점에 float(NaN)으로 변환되었을 수 있으므로 확인
        seoul_distance_value = most_recent_data.get(seoul_distance_feature_name, np.nan)
        # 노트북에서는 SimpleImputer(strategy='mean')를 사용하므로,
        # 여기서 0으로 대체하는 대신 NaN을 유지하여 preprocessor가 처리하도록 합니다.
        # (excel_df 로드 시점에 이미 float으로 변환되어 NaN일 수 있음)

        return place_feature_values, seoul_distance_value

# --- 예측 함수 ---

def predict_festival_outcome(si, gungu, dong, festival_date, festival_type, budget):
    """
    사용자 입력과 추가 피처를 결합하여 전처리 후 모델 예측을 수행합니다.

    Args:
        si (str): 광역자치단체 (예: "서울특별시")
        gungu (str): 기초자치단체 시/군/구 (예: "종로구")
        dong (str): 읍/면/동 (예: "세종로")
        festival_date (str): 축제 시작일 (YYYY-MM-DD 형식의 문자열)
        festival_type (str): 축제 종류 (8개 범주 중 하나 또는 유사 문자열)
        budget (float): 예산 (백만원 단위)

    Returns:
        float: 예측 방문객 수 (원래 스케일)
    """

    try:
        # 날짜 파싱 및 피처 생성
        start_date = pd.to_datetime(festival_date)
        start_year = start_date.year
        start_month = start_date.month

        # '계절' 계산
        season_map = {
            12: '겨울', 1: '겨울', 2: '겨울',
            3: '봄', 4: '봄', 5: '봄',
            6: '여름', 7: '여름', 8: '여름',
            9: '가을', 10: '가을', 11: '가을'
        }
        calculated_season = season_map.get(start_month, '기타')

        # '축제유형_대분류' 계산
        calculated_festival_type_category = clean_type(festival_type)

        # '예산\n(백만)' 변환 (원 단위) - 노트북과 동일하게 처리
        budget_won = float(budget) * 1_000_000

        # '서울과거리' 및 8개 장소 피처 조회
        raw_place_features, seoul_distance = get_additional_features(si, gungu, dong, excel_df)

        # '장소_PCA' 계산
        # pca_transformer는 (n_samples, n_features) 형태의 입력을 기대
        # raw_place_features는 1D list이므로 2D numpy array로 변환
        if raw_place_features is not None and len(raw_place_features) == pca_transformer.n_features_in_:
            place_features_df = pd.DataFrame([raw_place_features], columns=PLACE_COLS) # 또는 pca_transformer.feature_names_in_
            place_pca_value = pca_transformer.transform(place_features_df)[0, 0]
        else:
            # PCA 입력 피처 개수가 맞지 않거나 raw_place_features가 None일 경우
            # print(f"경고: PCA 변환을 위한 장소 피처 개수가 맞지 않거나 데이터가 없습니다. 장소_PCA는 0으로 설정됩니다.", file=sys.stderr)
            place_pca_value = 0.0

        # '주소점수' 계산
        si_freq = address_freq_maps['si'].get(si, 0)
        gungu_freq = address_freq_maps['gungu'].get(gungu, 0)
        dong_freq = address_freq_maps['dong'].get(dong, 0)
        calculated_address_score = si_freq + gungu_freq + dong_freq

        # '1인당예산' 처리 (주의: 데이터 누수 가능성으로 인해 0으로 설정)
        # 이 피처는 학습 시 타겟 변수('방문객합계')를 사용하여 계산되었습니다.
        # 예측 시점에는 '방문객합계'를 알 수 없으므로, 정확한 계산이 불가능합니다.
        # 노트북에서는 (df1['예산\n(백만)'] / (df1['방문객합계'] + 1)) 로 계산.
        # 예측 시점에는 방문객합계를 모르므로, 이 피처는 0 또는 평균값 등으로 대체하거나,
        # 모델 학습 시 이 피처를 제외하는 것이 더 바람직할 수 있습니다. 여기서는 0으로 설정합니다.
        calculated_per_capita_budget = 0.0

        # 입력 데이터를 DataFrame으로 구성
        input_data_dict = {
            '예산\n(백만)': [budget_won],
            '서울과거리': [seoul_distance],
            '1인당예산': [calculated_per_capita_budget], # 0.0으로 설정
            '장소_PCA': [place_pca_value],
            '주소점수': [calculated_address_score],
            '축제유형_대분류': [calculated_festival_type_category],
            '계절': [calculated_season]
        }

        # Create DataFrame using all available keys from input_data_dict first
        temp_df = pd.DataFrame(input_data_dict)

        # Ensure all columns expected by the preprocessor are present, in the correct order.
        # Add missing columns with np.nan if they are not in input_data_dict,
        # assuming the preprocessor's imputer will handle them.
        for col in PREPROCESSOR_FEATURE_ORDER:
            if col not in temp_df.columns:
                temp_df[col] = np.nan # Or a more appropriate default/imputation strategy
        
        input_df = temp_df[PREPROCESSOR_FEATURE_ORDER] # Select columns in the correct order

        # 데이터 전처리
        # preprocessor는 학습 데이터로 fit된 상태여야 합니다.
        preprocessed_data = preprocessor.transform(input_df)

        # 예측 수행
        # 모델이 로그 변환된 타겟(y = df1['방문객합계_log'])으로 학습되었으므로,
        # 예측값도 로그 스케일입니다. np.expm1()을 사용하여 원래 스케일로 되돌립니다.
        prediction_log = model.predict(preprocessed_data)
        prediction_original_scale = np.expm1(prediction_log[0]) # np.expm1 사용

        # 예측 결과가 음수일 경우 0으로 처리 (방문객 수는 음수가 될 수 없음)
        prediction_original_scale = max(0, prediction_original_scale)

        return prediction_original_scale

    except Exception as e:
        print(f"예측 함수 실행 중 오류 발생: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # 오류 발생 시 적절한 기본값 또는 오류 코드를 반환하도록 처리
        return -1 # 예: 오류를 나타내는 값 반환

# --- 스크립트 실행 부분 (서버에서 호출될 때 사용될 수 있는 형태) ---
# 이 부분은 스크립트를 직접 실행하여 테스트하거나,
# 서버에서 subprocess 등으로 호출하여 표준 입/출력을 통해 통신할 때 사용됩니다.
if __name__ == "__main__":
    # 서버로부터 JSON 형태의 입력 데이터를 표준 입력으로 받는다고 가정
    
    # UTF-8로 표준 입력 읽기
    try:
        input_bytes = sys.stdin.buffer.read()
        input_json_str = input_bytes.decode('utf-8')
        if not input_json_str.strip():
            print("오류: stdin으로부터 빈 입력을 받았습니다.", file=sys.stderr)
            sys.exit(1)
        print(f"Python: Received raw stdin ({len(input_bytes)} bytes): '{input_json_str}'", file=sys.stderr)
    except Exception as e:
        print(f"오류: stdin 읽기 또는 UTF-8 디코딩 중 오류 발생: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        input_data = json.loads(input_json_str)
        print(f"Python: Parsed JSON: {input_data}", file=sys.stderr)

        # 입력 데이터 매핑
        si_param = input_data.get("광역자치단체")
        gungu_param = input_data.get("기초자치단체 시/군/구")
        dong_param = input_data.get("읍/면/동")
        festival_date_param = input_data.get("축제 시작일") # Expected as "YYYY-MM-DD" string from TS server
        festival_type_param = input_data.get("축제 종류")
        budget_param = input_data.get("예산") # Expected as float (millions) from TS server

        # 디버깅을 위해 각 파라미터 값 로깅
        print(f"Python internal params: si_param = {repr(si_param)}, gungu_param = {repr(gungu_param)}, dong_param = {repr(dong_param)}, "
              f"festival_date_param = {repr(festival_date_param)}, festival_type_param = {repr(festival_type_param)}, budget_param = {repr(budget_param)}", file=sys.stderr)
        
        # 필수 입력값 확인
        # budget_param은 0일 수도 있으므로, None인지 여부만 확인
        # 다른 파라미터들은 문자열이므로, 비어있지 않은지 확인 (None 또는 빈 문자열이면 False로 평가됨)
        params_to_check = [
            si_param,             # True if non-empty string
            gungu_param,          # True if non-empty string
            dong_param,           # True if non-empty string
            festival_date_param,  # True if non-empty string
            festival_type_param,  # True if non-empty string
            budget_param is not None # True if budget_param is not None (0 is acceptable)
        ]
        
        all_params_valid = all(params_to_check)
        print(f"Python: Params for all() check: {[repr(p) for p in params_to_check[:-1]] + [params_to_check[-1]]}", file=sys.stderr)
        print(f"Python: Result of all() check (all_params_valid): {all_params_valid}", file=sys.stderr)

        if not all_params_valid:
             missing_details = []
             if not si_param: missing_details.append("광역자치단체")
             if not gungu_param: missing_details.append("기초자치단체 시/군/구")
             # ... (다른 필드들도 유사하게 추가 가능)
             if budget_param is None: missing_details.append("예산")
             print(f"오류: 필수 입력 데이터가 누락되었습니다. (Python __main__ check). 누락 의심 항목: {', '.join(missing_details) if missing_details else '확인 필요'}", file=sys.stderr)
             sys.exit(1)

        print("Python: Passed input validation in __main__. Calling predict_festival_outcome...", file=sys.stderr)
        # 예측 수행
        predicted_visitors = predict_festival_outcome(
            si_param, gungu_param, dong_param, festival_date_param, festival_type_param, budget_param
        )

        # 예측 결과를 JSON 형태로 표준 출력
        # 오류 값 (-1)인 경우 별도 처리 가능
        output_data = {"predicted_visitors": round(predicted_visitors)} # 소수점 반올림하여 정수로
        print(json.dumps(output_data))

    except json.JSONDecodeError:
        print("오류: 유효하지 않은 JSON 입력입니다.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"스크립트 실행 중 예상치 못한 오류 발생: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)