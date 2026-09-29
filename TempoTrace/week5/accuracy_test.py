# accuracy_test.py
# ------------------------------------------------------------
# 목적: 원곡 음원과 내 연주 음원을 비교해서 박자 정확도(%)를 계산한다.
#       (3주차 계획서 "정확도 계산 설계"의 4단계를 그대로 구현)
#
# 1단계: 원곡과 내 연주 각각에서 onset(소리 시작 지점) 시각 리스트를 뽑음
# 2단계: 원곡의 각 onset마다, 내 연주에서 가장 가까운 시각을 찾아 시간차 계산
# 3단계: 시간차가 허용 오차(TOLERANCE) 이내면 "맞음"으로 판정
# 4단계: 정확도(%) = 맞은 개수 / 원곡 onset 총개수 × 100
# ------------------------------------------------------------

import librosa
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import os   # 스크립트 파일의 위치를 기준으로 경로를 잡기 위해 사용

matplotlib.rcParams['font.family'] = 'Malgun Gothic'   # 한글 깨짐 방지 (윈도우 기준)
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# 설정값 — 본인 파일명/기준에 맞게 수정하는 부분
# ------------------------------------------------------------

# __file__ : 지금 실행 중인 이 스크립트 파일 자체의 경로를 담고 있는 파이썬 기본 변수
#            (예: "week4/accuracy_test.py" 처럼 상대적으로 나올 수도 있음)
#
# os.path.abspath(__file__) : 위 경로를 "어디서 실행하든 항상 똑같은" 완전한 절대경로로 바꿔줌
#            (예: "C:\SW_Project\TempoTrace\week4\accuracy_test.py")
#
# os.path.dirname(...) : 그 경로에서 파일 이름만 떼어내고 "폴더 경로"만 남김
#            (예: "C:\SW_Project\TempoTrace\week4")
#
# 즉 BASE_DIR에는 "이 스크립트가 실제로 들어있는 폴더 경로"가 항상 정확히 담기게 됨.
# 터미널을 어느 폴더에서 열어서 실행했든, VS Code 디버거로 실행했든 상관없이
# 항상 같은 결과가 나오기 때문에 "파일을 못 찾는" 문제를 근본적으로 막을 수 있음
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# os.path.join(폴더, 파일명) : 폴더 경로와 파일명을 운영체제에 맞게 이어붙여주는 함수
#            (윈도우는 \, 맥/리눅스는 / 를 쓰는데 이 차이를 신경 안 써도 되게 해줌)
#
# 결과적으로 ORIGINAL_FILE에는 예를 들어
# "C:\SW_Project\TempoTrace\week4\original.wav" 처럼 완전한 경로가 담김
ORIGINAL_FILE = os.path.join(BASE_DIR, "original.wav")   # 원곡 음원 파일
MY_TAKE_FILE = os.path.join(BASE_DIR, "my_take.wav")     # 내가 연주하고 녹음한 파일

TOLERANCE = 0.15                 # 허용 오차 (초 단위). 0.15 = 150ms 이내면 "맞음"으로 인정


def get_onset_times(file_path):
    """
    오디오 파일 경로를 받아서 onset(소리 시작 지점) 시각 리스트를 반환하는 함수.
    이 함수를 원곡, 내 연주 두 번 호출해서 각각의 onset 리스트를 얻는다.
    """
    # librosa.load : 오디오 파일을 읽어서 숫자 배열로 바꿔주는 함수
    #   y  : 소리의 파형 데이터 (시간에 따른 진폭 값들의 배열)
    #   sr : 샘플레이트(sample rate) — 1초에 몇 개의 숫자로 소리를 표현했는지 (보통 22050 또는 44100)
    #   sr=None : 원본 파일의 샘플레이트를 그대로 사용하겠다는 뜻 (임의로 변환하지 않음)
    y, sr = librosa.load(file_path, sr=None)

    # librosa.onset.onset_detect : y(파형)에서 "소리가 갑자기 커지는 지점"들을 찾아주는 핵심 함수
    #   결과는 "몇 번째 프레임에서 발생했는지"를 나타내는 정수 배열로 나옴 (아직 초 단위 아님)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

    # librosa.frames_to_time : 위에서 얻은 "프레임 번호"를 실제 "몇 초 지점인지"로 변환
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return y, sr, onset_times


def compare_onsets(original_times, user_times, tolerance):
    """
    원곡 onset 리스트와 내 연주 onset 리스트를 비교하는 함수.
    원곡의 각 음마다 내 연주에서 가장 가까운 음을 찾아 매칭한다.

    비유하자면: 원곡에 찍힌 각 타이밍(정답)을 하나씩 짚어가면서
    "내가 그 타이밍 근처에서 실제로 소리를 냈는지"를 확인하는 채점 방식이다.

    반환값:
      matches   : [(원곡시각, 내연주시각, 시간차), ...] — 허용 오차 안에 든 것들
      missed    : [원곡시각, ...] — 원곡엔 있는데 내 연주에서 못 찾은 것 (놓친 음)
      extra     : [내연주시각, ...] — 내 연주에는 있는데 원곡엔 없는 것 (엉뚱하게 친 음)
    """
    matches = []
    missed = []
    # used_user_indices : 이미 어떤 원곡 음과 짝지어진 "내 연주" onset의 번호를 기록해두는 집합(set)
    #   왜 필요하냐면, 내가 어떤 타이밍에 한 번만 쳤는데 그게 원곡의 여러 음과
    #   동시에 "맞음" 처리되어 중복으로 점수를 받는 것을 막기 위함
    used_user_indices = set()

    # 원곡의 onset을 하나씩 순서대로 확인 (이게 "정답지"를 한 문제씩 채점하는 것과 같음)
    for ot in original_times:
        if len(user_times) == 0:
            # 내가 아예 아무것도 안 쳤다면 원곡의 모든 음이 "놓친 음"이 됨
            missed.append(ot)
            continue

        # np.abs(user_times - ot) : 내 연주의 모든 onset 시각과 지금 원곡 음(ot) 사이의
        #   시간차(절댓값)를 한 번에 계산 (numpy 배열 연산이라 반복문 없이 빠르게 처리됨)
        diffs = np.abs(user_times - ot)

        # np.argmin(diffs) : 그 시간차들 중에서 "가장 작은 값의 위치(인덱스)"를 찾음
        #   → 즉, 내 연주 중에서 지금 이 원곡 음과 시간상 가장 가까운 소리를 찾는 것
        best_idx = np.argmin(diffs)
        best_diff = diffs[best_idx]

        # 조건 1) best_diff <= tolerance : 가장 가까운 소리라도 허용 오차(예: 150ms)보다 크면 탈락
        # 조건 2) best_idx not in used_user_indices : 이미 다른 원곡 음에 매칭된 소리는 재사용 불가
        # 두 조건을 모두 만족해야만 "맞음(matches)"으로 인정
        if best_diff <= tolerance and best_idx not in used_user_indices:
            matches.append((float(ot), float(user_times[best_idx]), float(best_diff)))
            used_user_indices.add(best_idx)   # 이 내 연주 onset은 이제 "사용됨" 표시
        else:
            missed.append(float(ot))         # 허용 오차를 넘으면 "놓친 음"으로 처리

    # 내 연주 onset 전체 중에서, 위 반복문에서 한 번도 매칭에 쓰이지 않은 것들을 뽑아냄
    # → 이건 "원곡에는 없는 타이밍인데 내가 소리를 낸 것" = 엉뚱하게 친 음(extra)
    extra = [float(t) for i, t in enumerate(user_times) if i not in used_user_indices]

    return matches, missed, extra


# ------------------------------------------------------------
# 실행부
# ------------------------------------------------------------

print("원곡 분석 중...")
y_orig, sr_orig, original_times = get_onset_times(ORIGINAL_FILE)
print(f"  원곡에서 감지된 onset: {len(original_times)}개")

print("내 연주 분석 중...")
y_mine, sr_mine, user_times = get_onset_times(MY_TAKE_FILE)
print(f"  내 연주에서 감지된 onset: {len(user_times)}개")

matches, missed, extra = compare_onsets(original_times, user_times, TOLERANCE)

accuracy = len(matches) / len(original_times) * 100 if len(original_times) > 0 else 0
avg_error_ms = np.mean([m[2] for m in matches]) * 1000 if matches else 0

# ------------------------------------------------------------
# 결과 출력
# ------------------------------------------------------------
print("\n" + "=" * 50)
print(f"박자 정확도: {accuracy:.1f}%  ({len(matches)} / {len(original_times)} 맞음)")
print(f"평균 타이밍 오차: {avg_error_ms:.1f}ms")
print(f"놓친 음(missed): {len(missed)}개")
print(f"엉뚱하게 친 음(extra): {len(extra)}개")
print("=" * 50)

if missed:
    print("\n놓친 지점 (초):")
    for t in missed:
        print(f"  - {int(t//60)}분 {t%60:.2f}초")

if extra:
    print("\n엉뚱하게 친 지점 (초):")
    for t in extra:
        print(f"  - {int(t//60)}분 {t%60:.2f}초")

# ------------------------------------------------------------
# 그래프로 비교 시각화 (원곡 위: 파란 점선 / 내 연주 아래: 빨간 실선)
# ------------------------------------------------------------
# 6. 웹뷰어에서 읽을 수 있도록 결과를 정리 (그래프를 띄우기 전에 먼저 준비)
# ------------------------------------------------------------
def downsample(y, target_points=2000):
    """
    파형 데이터(y)는 보통 수십만~수백만 개의 숫자로 이루어져 있어서
    그대로 웹페이지에 넘기면 파일이 너무 커지고 브라우저도 느려짐.
    그래서 일정 간격으로 건너뛰며 점을 뽑아(다운샘플링) 개수를 줄여줌.
    (그래프 모양은 거의 그대로 유지되면서 데이터 양만 확 줄어듦)
    """
    step = max(1, len(y) // target_points)   # 몇 개마다 하나씩 뽑을지 계산 (예: 500개마다 1개)
    return y[::step].tolist()                # step 간격으로 건너뛰며 뽑고, 파이썬 리스트로 변환
                                              # (numpy 배열은 JSON으로 바로 저장이 안 되기 때문에 tolist() 필요)

# 지금까지 계산한 모든 결과를 하나의 딕셔너리(사전)로 모음
# 이 딕셔너리가 그대로 JSON 형태로 저장되어, 웹페이지(자바스크립트)에서 읽게 됨
result_data = {
    "accuracy": round(accuracy, 1),                      # 정확도(%), 소수점 첫째 자리까지
    "avg_error_ms": round(avg_error_ms, 1),               # 평균 타이밍 오차(밀리초)
    "tolerance_ms": round(TOLERANCE * 1000, 0),           # 이번 계산에 쓰인 허용 오차 값(ms) — 웹페이지 슬라이더 초기값으로 사용
    "original_duration": float(len(y_orig) / sr_orig),     # 원곡 길이(초) = 전체 샘플 수 ÷ 샘플레이트
    "original_onsets": [float(t) for t in original_times], # 원곡 onset 시각들 (그래프의 파란 선 위치)
    "original_waveform": downsample(y_orig),               # 원곡 파형 (다운샘플링된 것)
    "user_duration": float(len(y_mine) / sr_mine),         # 내 연주 길이(초)
    "user_onsets": [float(t) for t in user_times],         # 내 연주 onset 원본 전체 리스트 (허용 오차 재계산용)
    "matched": [{"original": m[0], "user": m[1], "error": m[2]} for m in matches],  # 맞은 음들의 상세 정보
    "missed": missed,                                      # 놓친 음 시각 리스트
    "extra": extra,                                        # 엉뚱하게 친 음 시각 리스트
    "user_waveform": downsample(y_mine),                   # 내 연주 파형 (다운샘플링된 것)
}

with open(os.path.join(BASE_DIR, "accuracy_result.json"), "w", encoding="utf-8") as f:
    json.dump(result_data, f, ensure_ascii=False)   # ensure_ascii=False : 한글이 깨지지 않고 그대로 저장되게 함

print("\naccuracy_result.json 파일이 저장되었습니다.")

# ------------------------------------------------------------
# 7. 결과를 브라우저로 자동으로 띄우기
#
#    기존 방식의 문제: JSON 파일을 따로 만들고, 사람이 직접 브라우저에서
#    accuracy_viewer.html을 열고, "파일 선택" 버튼을 눌러 JSON을 골라야 했음
#    → 손이 많이 감
#
#    개선한 방식: accuracy_viewer.html의 내용을 파이썬이 통째로 읽어온 뒤,
#    그 안에 분석 결과(JSON)를 "이미 채워진 상태"로 직접 심어서
#    accuracy_report.html 이라는 새 파일을 만들고, 파이썬이 그 파일을
#    바로 브라우저로 열어줌 → 사람이 할 일이 하나도 없어짐
# ------------------------------------------------------------
import webbrowser   # 기본 웹 브라우저로 파일을 자동으로 열어주는 파이썬 표준 라이브러리 (별도 설치 필요 없음)

VIEWER_TEMPLATE = os.path.join(BASE_DIR, "accuracy_viewer.html")  # 뼈대가 되는 뷰어 HTML (기존 파일, 안 바뀜)
REPORT_FILE = os.path.join(BASE_DIR, "accuracy_report.html")      # 결과가 심어진 새 HTML 파일 (매번 새로 생성됨)

# accuracy_viewer.html이 같은 폴더에 있는지 먼저 확인 (없으면 아래 과정을 건너뜀)
if os.path.exists(VIEWER_TEMPLATE):

    # 1) accuracy_viewer.html 파일의 내용을 통째로 문자열(html)로 읽어옴
    with open(VIEWER_TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()

    # 2) 분석 결과(result_data, 파이썬 딕셔너리)를 자바스크립트가 읽을 수 있는
    #    문자열 형태로 변환. json.dumps()는 파이썬 데이터를 JSON 형식의
    #    "텍스트"로 바꿔주는 함수 (json.dump는 파일에 저장, dumps는 문자열로 반환 — s 차이에 주의)
    #
    #    변환 결과 예시: <script>const EMBEDDED_DATA = {"accuracy": 87.5, ...};</script>
    #    이 한 줄을 HTML 안에 넣으면, 웹페이지가 열리자마자 자바스크립트에
    #    EMBEDDED_DATA라는 이름의 변수로 분석 결과가 이미 들어있게 됨
    embedded_script = f"<script>const EMBEDDED_DATA = {json.dumps(result_data, ensure_ascii=False)};</script>"

    # 3) 문자열 치환(replace)으로 HTML의 </head> 태그 바로 앞에 위 스크립트를 끼워 넣음
    #    (파이썬의 문자열 replace는 "이 텍스트를 찾아서 저 텍스트로 바꿔라"는 뜻)
    html = html.replace("</head>", embedded_script + "\n</head>")

    # 4) accuracy_viewer.html은 원래 사람이 "파일 선택"을 눌러야 renderResult() 함수가
    #    실행되도록 만들어져 있음. 여기서는 그 과정을 건너뛰고, 페이지가 열리자마자
    #    (DOMContentLoaded = "웹페이지 로딩이 끝났다"는 신호) 자동으로 renderResult()를
    #    호출하도록 자바스크립트 코드를 추가로 심음
    auto_render_script = """
<script>
  window.addEventListener('DOMContentLoaded', function () {
    if (typeof EMBEDDED_DATA !== 'undefined') {
      document.getElementById('toleranceSlider').value = EMBEDDED_DATA.tolerance_ms || 150;  // 슬라이더 시작 위치도 맞춰줌
      renderResult(EMBEDDED_DATA);          // 파일 업로드 없이 바로 결과를 그림
      document.querySelector('.upload-box').style.display = 'none';  // 이제 필요 없는 업로드 박스는 화면에서 숨김
    }
  });
</script>
"""
    html = html.replace("</body>", auto_render_script + "\n</body>")

    # 5) 완성된 HTML 내용을 accuracy_report.html이라는 새 파일로 저장
    #    (원본 accuracy_viewer.html은 건드리지 않고 그대로 둠 — 나중에 다른 JSON을
    #     수동으로 열어보고 싶을 때 계속 재사용 가능하도록)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"결과 리포트가 생성되었습니다: {REPORT_FILE}")

    # 6) webbrowser.open()은 "이 파일을 컴퓨터의 기본 웹 브라우저로 열어라"는 명령
    #    file:// 는 인터넷 주소가 아니라 "내 컴퓨터 안의 파일"이라는 뜻의 접두사
    webbrowser.open("file://" + REPORT_FILE)
    print("브라우저에서 결과 화면이 자동으로 열립니다.")
else:
    print(f"\n주의: {VIEWER_TEMPLATE} 파일이 같은 폴더에 없어서 자동 리포트 생성은 건너뜁니다.")
    print("accuracy_viewer.html을 이 스크립트와 같은 폴더에 넣어주세요.")

# ------------------------------------------------------------
# 8. 그래프를 창으로도 띄워서 확인 (원하면 창을 닫아도 위 브라우저 결과는 그대로 남아있음)
# ------------------------------------------------------------
plt.figure(figsize=(14, 6))

plt.subplot(2, 1, 1)
librosa.display.waveshow(y_orig, sr=sr_orig, alpha=0.6, color='steelblue')
plt.vlines(original_times, -1, 1, color='blue', linestyle='--', label='원곡 onset')
plt.title("원곡")
plt.legend()

plt.subplot(2, 1, 2)
librosa.display.waveshow(y_mine, sr=sr_mine, alpha=0.6, color='indianred')
plt.vlines([m[1] for m in matches], -1, 1, color='green', linestyle='--', label='맞은 음')
plt.vlines(extra, -1, 1, color='red', linestyle='--', label='엉뚱하게 친 음')
plt.title(f"내 연주 — 정확도 {accuracy:.1f}%")
plt.legend()

plt.tight_layout()
plt.subplots_adjust(hspace=0.5)   # 위/아래 그래프 사이 간격을 넓혀서 제목과 x축 라벨이 겹치지 않게 함
plt.savefig(os.path.join(BASE_DIR, "accuracy_result.png"))
plt.show()
