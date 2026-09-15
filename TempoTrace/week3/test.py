# onset_test.py
# ------------------------------------------------------------
# 목적: librosa의 onset detection이 실제로 잘 동작하는지 확인하는 테스트 스크립트
# 사용법: 같은 폴더에 샘플 음원 파일(wav, mp3 등)을 넣고 아래 FILE_PATH만 바꿔서 실행
# 결과: 1) 콘솔 출력  2) 그래프 창(파이썬 내에서 바로 확인)  3) onset_result.json (웹뷰어용 데이터)
# ------------------------------------------------------------

import librosa                      # 오디오 분석 라이브러리 (onset detection 핵심 기능 제공)
import librosa.display              # 파형/스펙트로그램 그림 그릴 때 필요
import numpy as np                  # 숫자 배열 계산용
import matplotlib.pyplot as plt     # 그래프 그리기용
import matplotlib
import json                         # 분석 결과를 JSON 파일로 저장하기 위해 사용

# 그래프에 한글이 깨지지 않도록 폰트 설정 (윈도우 기준 '맑은 고딕' 사용)
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 1. 분석할 음원 파일 경로 (본인 샘플 음원으로 수정)
FILE_PATH = "sample.wav"

# 2. 음원 파일 불러오기
#    y: 오디오 파형 데이터 (숫자 배열)
#    sr: 샘플레이트 (초당 샘플 개수, 보통 22050 또는 44100)
y, sr = librosa.load(FILE_PATH, sr=None)

print(f"파일 로드 완료 — 길이: {len(y)/sr:.2f}초, 샘플레이트: {sr}Hz")

# 3. onset detection 실행
#    소리가 갑자기 커지는(새로 시작되는) 지점들을 프레임 단위로 찾아냄
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

# 4. 프레임 번호를 실제 "몇 초 지점인지"로 변환
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

print(f"\n감지된 onset(소리 시작 지점) 개수: {len(onset_times)}개")
print("각 지점 (초 단위):")
for i, t in enumerate(onset_times):
    minutes = int(t // 60)
    seconds = t % 60
    print(f"  {i+1}번째: {minutes}분 {seconds:.2f}초")

# 5. 결과를 그래프 창으로 바로 띄워서 확인
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr, alpha=0.6)          # 원본 파형 그리기
plt.vlines(onset_times, ymin=-1, ymax=1, color='r',    # onset 지점에 빨간 세로선 표시
           linestyle='--', label='감지된 onset')
plt.xlabel("시간 (초)")
plt.ylabel("진폭")
plt.title("Onset Detection 결과")
plt.legend()
plt.tight_layout()
plt.savefig("onset_result.png")  # 발표자료 캡처용 이미지 파일로도 저장됨
plt.show()

print("\n그래프 창이 떴습니다. onset_result.png 파일로도 저장되었습니다.")

# ------------------------------------------------------------
# 6. 웹페이지(onset_viewer.html)에서 읽을 수 있도록 결과를 JSON 파일로 저장
# ------------------------------------------------------------

# 파형 전체를 그대로 저장하면 용량이 너무 커지므로, 화면에 그릴 정도로만 점 개수를 줄임 (다운샘플링)
TARGET_POINTS = 2000                          # 웹페이지에 그릴 파형 점 개수
step = max(1, len(y) // TARGET_POINTS)        # 몇 개 샘플마다 하나씩 뽑을지 계산
waveform_downsampled = y[::step].tolist()     # step 간격으로 샘플을 뽑아 리스트로 변환

result_data = {
    "duration": float(len(y) / sr),           # 전체 음원 길이 (초)
    "onset_times": [float(t) for t in onset_times],  # 감지된 onset 시각들 (초 단위 리스트)
    "waveform": waveform_downsampled          # 웹페이지에 그릴 다운샘플링된 파형 데이터
}

with open("onset_result.json", "w", encoding="utf-8") as f:
    json.dump(result_data, f, ensure_ascii=False)  # 한글 깨짐 방지 옵션 포함

print("분석 결과가 onset_result.json 파일로 저장되었습니다.")
print("이 파일을 onset_viewer.html의 '파일 선택'에서 불러오면 웹페이지에서 결과를 확인할 수 있습니다.")
