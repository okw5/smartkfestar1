from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import re
import sys
import json
import traceback
import os

# --- 파일 경로 설정 ---
MODEL_DIR = 'model'
STATIC_DIR = 'static' # static 폴더 경로 추가
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.joblib')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')
PCA_TRANSFORMER_PATH = os.path.join(MODEL_DIR, 'pca_transformer.joblib')
FREQ_MAPS_PATH = os.path.join(MODEL_DIR, 'address_freq_maps.joblib')
EXCEL_DATA_PATH = os.path.join(MODEL_DIR, '축제_데이터셋_업로드용.xlsx')

# --- 전역 변수 로드 (기존과 동일) ---
try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    pca_transformer = joblib.load(PCA_TRANSFORMER_PATH)
    address_freq_maps = joblib.load(FREQ_MAPS_PATH)
    excel_df = pd.read_excel(EXCEL_DATA_PATH)
    excel_df['예산\n(백만)'] = excel_df['예산\n(백만)'].replace({'미확정': 0, '무응답': 0}).astype(float)
    excel_df['서울과거리'] = excel_df['서울과거리'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
    PLACE_COLS = ['공원/유원지', '관광농원/허브마을', '일반관광지', '동식물원', '먹거리/패션거리',
                  '산림욕장/휴향림/수목원', '폭포/계곡/호수/저수지', '해수욕장']
    excel_df[PLACE_COLS] = excel_df[PLACE_COLS].fillna(0)
    print("모든 모델과 데이터를 성공적으로 불러왔습니다.", file=sys.stderr)
except Exception as e:
    print(f"파일 로드 중 치명적 오류 발생: {e}", file=sys.stderr)
    sys.exit(1)

PREPROCESSOR_FEATURE_ORDER = ['예산\n(백만)', '서울과거리', '1인당예산', '장소_PCA', '주소점수', '축제유형_대분류', '계절']

# --- 전처리 및 예측 함수 (기존과 동일) ---
def clean_type(x):
    if not isinstance(x, str): return '기타'
    s = x.lower().replace('\n', ' ').strip()
    s = re.sub(r'\s+', ' ', s)
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
    location_filter = (excel_data['시'] == si) & (excel_data['군구'] == gungu) & (excel_data['법정동'] == dong)
    relevant_data = excel_data[location_filter]
    if relevant_data.empty: return [0.0] * len(PLACE_COLS), 0.0
    most_recent_data = relevant_data.sort_values(by='시작년', ascending=False).iloc[0]
    place_feature_values = [most_recent_data.get(col, 0.0) for col in PLACE_COLS]
    seoul_distance_value = most_recent_data.get('서울과거리', np.nan)
    return place_feature_values, seoul_distance_value

def predict_festival_outcome(si, gungu, dong, festival_date, festival_type, budget):
    try:
        start_month = pd.to_datetime(festival_date).month
        season_map = {12: '겨울', 1: '겨울', 2: '겨울', 3: '봄', 4: '봄', 5: '봄', 6: '여름', 7: '여름', 8: '여름', 9: '가을', 10: '가을', 11: '가을'}
        calculated_season = season_map.get(start_month, '기타')
        calculated_festival_type_category = clean_type(festival_type)
        budget_won = float(budget) * 1_000_000
        raw_place_features, seoul_distance = get_additional_features(si, gungu, dong, excel_df)
        place_pca_value = pca_transformer.transform(pd.DataFrame([raw_place_features], columns=PLACE_COLS))[0, 0] if raw_place_features and len(raw_place_features) == pca_transformer.n_features_in_ else 0.0
        calculated_address_score = address_freq_maps['si'].get(si, 0) + address_freq_maps['gungu'].get(gungu, 0) + address_freq_maps['dong'].get(dong, 0)
        input_data_dict = {
            '예산\n(백만)': [budget_won], '서울과거리': [seoul_distance], '1인당예산': [0.0], '장소_PCA': [place_pca_value],
            '주소점수': [calculated_address_score], '축제유형_대분류': [calculated_festival_type_category], '계절': [calculated_season]
        }
        input_df = pd.DataFrame(input_data_dict, columns=PREPROCESSOR_FEATURE_ORDER)
        preprocessed_data = preprocessor.transform(input_df)
        prediction_log = model.predict(preprocessed_data)
        return max(0, np.expm1(prediction_log[0]))
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return -1

# [수정] Flask 앱 초기화 시 static 폴더 경로 지정
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='/static')

# --- 웹 페이지 라우트 ---
@app.route('/', methods=['GET'])
def index():
    # [수정] 초기 렌더링 시에도 original_input을 전달하여 템플릿 오류 방지
    return render_template('index.html', original_input={})

@app.route('/predict', methods=['POST'])
def predict_web():
    form_data = request.form
    try:
        si_param = form_data['광역자치단체']
        gungu_param = form_data['기초자치단체 시/군/구']
        # '읍/면/동'은 이제 form에서 올바르게 전달됨
        dong_param = form_data['읍/면/동']
        festival_date_param = form_data['축제 시작일']
        festival_type_param = form_data['축제 종류']
        budget_param_str = form_data['예산']
        
        if not all([si_param, gungu_param, dong_param, festival_date_param, festival_type_param, budget_param_str.strip()]):
            return render_template('index.html', error="모든 필드를 채워주세요.", original_input=form_data)
        
        budget_param = float(budget_param_str)

        predicted_visitors = predict_festival_outcome(
            si_param, gungu_param, dong_param, festival_date_param, festival_type_param, budget_param
        )

        if predicted_visitors == -1:
            return render_template('index.html', error="예측 중 내부 오류가 발생했습니다. 서버 로그를 확인하세요.", original_input=form_data)

        result_message = f"예상 방문객 수: {round(predicted_visitors):,}명"
        return render_template('index.html', prediction_result=result_message, original_input=form_data)

    except ValueError:
        return render_template('index.html', error="예산은 유효한 숫자여야 합니다.", original_input=form_data)
    except KeyError as e:
        return render_template('index.html', error=f"필수 입력 필드({e})가 누락되었습니다.", original_input=form_data)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return render_template('index.html', error=f"예측 처리 중 오류 발생: {e}", original_input=form_data)


# (API 엔드포인트는 변경 없음)

if __name__ == '__main__':
    # 디버그 모드는 개발 중에만 사용하세요.
    app.run(host='0.0.0.0', port=5000, debug=True)