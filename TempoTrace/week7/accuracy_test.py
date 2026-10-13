# accuracy_test.py  —  TempoTrace (7주차 버전)
# ------------------------------------------------------------
# 목적: 원곡과 내 연주를 비교해서 박자 정확도(%)를 계산하고,
#       (1) 곡의 어느 시간대에서 틀렸는지
#       (2) 연습을 반복할수록 정확도가 어떻게 변하는지
#       까지 한 화면(웹 리포트)으로 보여준다.
#
# 사용법
#   1) 그냥 실행 (VS Code의 ▶ 버튼 포함)
#        → original.wav 와 my_take.wav 를 비교
#   2) python accuracy_test.py 연주1.wav 연주2.wav 연주3.wav
#        → 여러 번의 연습(take)을 순서대로 분석하고, 모두 연습 기록에 남김
#   3) python accuracy_test.py 연주.wav --original=곡.wav
#        → 원곡 파일을 직접 지정
#   옵션
#        --no-history     이번 결과를 연습 기록(practice_history.json)에 남기지 않음
#        --reset-history  기존 연습 기록을 모두 지우고 새로 시작
#
# 3주차 계획서의 4단계(onset 추출 → 가장 가까운 음 매칭 → 허용 오차 판정 → 정확도 계산)는
# 그대로이고, 이번 버전에서 "시간대별 집계"와 "연습 기록 누적"이 추가되었다.
# ------------------------------------------------------------

import sys
import os
import json
import math
import datetime
import webbrowser
import pathlib

import numpy as np
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt

# 그래프 한글 깨짐 방지: 윈도우(맑은 고딕) → 맥(AppleGothic) → 리눅스(나눔고딕) 순으로 찾고,
# 아무것도 없으면 기본 폰트(DejaVu Sans)로 대체한다. (경고 없이 어느 환경에서나 실행되도록)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# 설정값 — 필요하면 수정하는 부분
# ------------------------------------------------------------

# 이 스크립트가 들어있는 폴더 (실행 위치와 상관없이 항상 같은 폴더에서 파일을 찾기 위함)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_ORIGINAL = "original.wav"      # 원곡 음원 파일 (인자를 안 주면 이 파일 사용)
DEFAULT_TAKES = ["my_take.wav"]        # 내 연주 파일 (인자를 안 주면 이 파일 사용)

TOLERANCE = 0.15       # 허용 오차(초). 0.15 = 150ms 이내면 "맞음"으로 인정

SEGMENT_SECONDS = 2.0  # 시간대별 분석에서 곡을 나누는 기본 구간 길이(초)
MAX_SEGMENTS = 15      # 곡이 길어도 구간이 이 개수를 넘지 않도록 구간 길이를 자동으로 늘림
MAX_HISTORY_TOTAL = 100  # 연습 기록 파일에 보관할 최대 개수 (넘으면 오래된 것부터 삭제)
MAX_HISTORY_SHOWN = 30   # 웹 리포트에 보여줄 최근 연습 기록 개수

HISTORY_FILE = os.path.join(BASE_DIR, "practice_history.json")     # 연습 기록 저장 파일
RESULT_JSON = os.path.join(BASE_DIR, "accuracy_result.json")       # 최신 분석 결과 저장 파일
VIEWER_TEMPLATE = os.path.join(BASE_DIR, "accuracy_viewer.html")   # 웹 리포트의 뼈대 HTML
REPORT_FILE = os.path.join(BASE_DIR, "accuracy_report.html")       # 결과가 심어진 웹 리포트
RESULT_PNG = os.path.join(BASE_DIR, "accuracy_result.png")         # 그래프 이미지


# ------------------------------------------------------------
# 1. 명령줄 인자 처리
# ------------------------------------------------------------
def parse_args(argv):
    """
    터미널에서 입력한 인자를 해석한다.
      - '--'로 시작하는 것: 옵션
      - 나머지: 내 연주(take) 파일 이름들
    """
    flags = [a for a in argv if a.startswith("--")]
    files = [a for a in argv if not a.startswith("--")]

    options = {"no_history": False, "reset_history": False, "original": DEFAULT_ORIGINAL}
    for f in flags:
        if f == "--no-history":
            options["no_history"] = True
        elif f == "--reset-history":
            options["reset_history"] = True
        elif f.startswith("--original="):
            options["original"] = f.split("=", 1)[1]
        else:
            print(f"알 수 없는 옵션은 무시합니다: {f}")

    takes = files if files else list(DEFAULT_TAKES)
    return options, takes


def resolve_path(name):
    """
    파일 이름을 실제 경로로 바꿔준다.
    스크립트와 같은 폴더에 있으면 그 파일을 쓰고, 아니면 입력한 경로 그대로 사용한다.
    """
    if os.path.isabs(name):
        return name
    candidate = os.path.join(BASE_DIR, name)
    return candidate if os.path.exists(candidate) else os.path.abspath(name)


def require_file(path):
    """파일이 없으면 이유를 알려주고 프로그램을 종료한다."""
    if not os.path.exists(path):
        print(f"\n[오류] 파일을 찾을 수 없습니다: {path}")
        print("파일 이름이 맞는지, 스크립트와 같은 폴더에 있는지 확인해주세요.")
        sys.exit(1)


# ------------------------------------------------------------
# 2. onset 추출 / 비교 (3~4주차에서 만든 핵심 로직 — 변경 없음)
# ------------------------------------------------------------
def get_onset_times(file_path):
    """
    오디오 파일 경로를 받아서 onset(소리 시작 지점) 시각 리스트를 반환하는 함수.
    이 함수를 원곡, 내 연주 각각에 대해 호출해서 onset 리스트를 얻는다.
    """
    # librosa.load : 오디오 파일을 읽어서 숫자 배열로 바꿔주는 함수
    #   y  : 소리의 파형 데이터 (시간에 따른 진폭 값들의 배열)
    #   sr : 샘플레이트 — 1초에 몇 개의 숫자로 소리를 표현했는지 (보통 22050 또는 44100)
    #   sr=None : 원본 파일의 샘플레이트를 그대로 사용
    y, sr = librosa.load(file_path, sr=None)

    # librosa.onset.onset_detect : "소리가 갑자기 커지는 지점"을 프레임 번호로 찾아줌
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

    # librosa.frames_to_time : 프레임 번호를 실제 "몇 초 지점인지"로 변환
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    return y, sr, onset_times


def compare_onsets(original_times, user_times, tolerance):
    """
    원곡 onset 리스트와 내 연주 onset 리스트를 비교하는 함수.
    원곡의 각 음마다 내 연주에서 가장 가까운 음을 찾아 매칭한다.

    비유하자면: 원곡에 찍힌 각 타이밍(정답)을 하나씩 짚어가면서
    "내가 그 타이밍 근처에서 실제로 소리를 냈는지"를 확인하는 채점 방식이다.

    반환값:
      matches : [(원곡시각, 내연주시각, 시간차), ...] — 허용 오차 안에 든 것들
      missed  : [원곡시각, ...] — 원곡엔 있는데 내 연주에서 못 찾은 것 (놓친 음)
      extra   : [내연주시각, ...] — 내 연주에는 있는데 원곡엔 없는 것 (엉뚱하게 친 음)
    """
    matches = []
    missed = []
    # 이미 어떤 원곡 음과 짝지어진 "내 연주" onset의 번호를 기록하는 집합.
    # 한 번 친 소리가 원곡의 여러 음과 동시에 "맞음" 처리되는 것을 막기 위함.
    used_user_indices = set()

    for ot in original_times:
        if len(user_times) == 0:
            missed.append(float(ot))     # 아무것도 안 쳤다면 원곡의 모든 음이 "놓친 음"
            continue

        # 내 연주의 모든 onset과 이 원곡 음(ot) 사이의 시간차(절댓값)를 한 번에 계산
        diffs = np.abs(user_times - ot)

        # 그중 가장 작은 값의 위치 = 시간상 가장 가까운 내 연주 소리
        best_idx = np.argmin(diffs)
        best_diff = diffs[best_idx]

        # 조건 1) 가장 가까운 소리라도 허용 오차보다 멀면 탈락
        # 조건 2) 이미 다른 원곡 음과 매칭된 소리는 재사용 불가
        if best_diff <= tolerance and best_idx not in used_user_indices:
            matches.append((float(ot), float(user_times[best_idx]), float(best_diff)))
            used_user_indices.add(best_idx)
        else:
            missed.append(float(ot))

    # 한 번도 매칭에 쓰이지 않은 내 연주 소리 = 원곡에 없는 타이밍에 낸 소리
    extra = [float(t) for i, t in enumerate(user_times) if i not in used_user_indices]

    return matches, missed, extra


# ------------------------------------------------------------
# 3. [신규] 시간대별 집계
# ------------------------------------------------------------
def choose_segment_length(duration):
    """
    곡을 몇 초 단위로 나눌지 정한다.
    기본은 SEGMENT_SECONDS(2초)이고, 곡이 길어서 구간이 MAX_SEGMENTS(15)개를 넘으면
    구간 길이를 2배, 3배... 로 늘려서 그래프가 너무 빽빽해지지 않게 한다.
    (예: 12초 곡 → 2초 × 6구간 / 3분 곡 → 12초 × 15구간)
    """
    n = max(1, math.ceil(round(duration / SEGMENT_SECONDS, 3)))
    factor = math.ceil(n / MAX_SEGMENTS)
    return SEGMENT_SECONDS * factor


def segment_count(duration, seg_len):
    """
    구간 개수를 계산한다.
    곡 끝에 아주 짧은 조각(구간 길이의 25% 미만)이 남으면 따로 세지 않고
    마지막 구간에 합쳐서, 의미 없는 얇은 구간이 생기지 않게 한다.
    (웹페이지의 자바스크립트도 똑같은 규칙으로 계산한다)
    """
    n = max(1, math.ceil(round(duration / seg_len, 3)))
    if n > 1 and (duration - (n - 1) * seg_len) < 0.25 * seg_len:
        n -= 1
    return n


def compute_segments(original_times, missed, extra, duration, seg_len):
    """
    곡을 seg_len초 단위 구간으로 나누고, 구간마다
      total  : 그 구간에 있는 원곡 음 개수
      missed : 그중 놓친 음 개수 (원곡 시각 기준)
      extra  : 그 구간에서 엉뚱하게 친 음 개수 (내 연주 시각 기준)
    를 센다. "어느 시간대에서 얼마나 틀렸는가"를 보여주기 위한 데이터.
    """
    n = segment_count(duration, seg_len)
    segments = [
        {"start": i * seg_len, "end": min((i + 1) * seg_len, duration),
         "total": 0, "missed": 0, "extra": 0}
        for i in range(n)
    ]

    def index_of(t):
        # 시각 t가 몇 번째 구간에 속하는지 (범위를 벗어나면 처음/끝 구간으로 맞춤)
        return min(n - 1, max(0, math.floor(t / seg_len)))

    for t in original_times:
        segments[index_of(float(t))]["total"] += 1
    for t in missed:
        segments[index_of(float(t))]["missed"] += 1
    for t in extra:
        segments[index_of(float(t))]["extra"] += 1
    return segments


def format_time(t):
    """초 단위 시각을 '0분 4.53초' 형태의 문자열로 바꾼다."""
    return f"{int(t // 60)}분 {t % 60:.2f}초"


# ------------------------------------------------------------
# 4. 연주 하나를 분석
# ------------------------------------------------------------
def analyze_take(take_path, original_times, original_duration, seg_len):
    """내 연주 파일 하나를 분석해서 결과를 딕셔너리로 돌려준다."""
    y, sr, user_times = get_onset_times(take_path)
    matches, missed, extra = compare_onsets(original_times, user_times, TOLERANCE)

    accuracy = len(matches) / len(original_times) * 100 if len(original_times) > 0 else 0
    avg_error_ms = float(np.mean([m[2] for m in matches])) * 1000 if matches else 0.0
    segments = compute_segments(original_times, missed, extra, original_duration, seg_len)

    return {
        "path": take_path, "name": os.path.basename(take_path),
        "y": y, "sr": sr, "user_times": user_times,
        "matches": matches, "missed": missed, "extra": extra,
        "accuracy": accuracy, "avg_error_ms": avg_error_ms, "segments": segments,
    }


def print_summary(res, original_count, seg_len):
    """분석 결과를 터미널에 보기 좋게 출력한다."""
    print(f"  내 연주에서 감지된 onset: {len(res['user_times'])}개")
    print("\n" + "=" * 50)
    print(f"[{res['name']}]")
    print(f"박자 정확도: {res['accuracy']:.1f}%  ({len(res['matches'])} / {original_count} 맞음)")
    print(f"평균 타이밍 오차: {res['avg_error_ms']:.1f}ms")
    print(f"놓친 음(missed): {len(res['missed'])}개")
    print(f"엉뚱하게 친 음(extra): {len(res['extra'])}개")
    print("=" * 50)

    if res["missed"]:
        print("\n놓친 지점:")
        for t in res["missed"]:
            print(f"  - {format_time(t)}")
    if res["extra"]:
        print("\n엉뚱하게 친 지점:")
        for t in res["extra"]:
            print(f"  - {format_time(t)}")

    # 오류가 있는 구간만 골라서 출력
    bad = [s for s in res["segments"] if s["missed"] + s["extra"] > 0]
    print(f"\n시간대별 오류 ({seg_len:g}초 단위 구간):")
    if not bad:
        print("  모든 구간에서 오류가 없습니다.")
    for s in bad:
        print(f"  - {s['start']:g}~{s['end']:g}초  놓침 {s['missed']} · 엉뚱 {s['extra']}")


# ------------------------------------------------------------
# 5. [신규] 연습 기록 (practice_history.json)
# ------------------------------------------------------------
def load_history():
    """저장된 연습 기록을 읽어온다. 파일이 없거나 깨져 있으면 빈 목록으로 시작한다."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        print("주의: 연습 기록 파일을 읽지 못해 새로 시작합니다.")
        return []


def save_history(history):
    """연습 기록을 파일로 저장한다. (너무 많으면 오래된 것부터 버린다)"""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history[-MAX_HISTORY_TOTAL:], f, ensure_ascii=False, indent=1)


def make_history_entry(res, original_name, seg_len):
    """이번 연습 결과를 기록 한 줄(딕셔너리)로 만든다. 화면에 그리는 데 필요한 요약만 담는다."""
    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "original": original_name,
        "take": res["name"],
        "accuracy": round(res["accuracy"], 1),
        "avg_error_ms": round(res["avg_error_ms"], 1),
        "tolerance_ms": round(TOLERANCE * 1000),
        "missed": len(res["missed"]),
        "extra": len(res["extra"]),
        "segment_seconds": seg_len,
        "segment_missed": [s["missed"] for s in res["segments"]],
        "segment_extra": [s["extra"] for s in res["segments"]],
    }


# ------------------------------------------------------------
# 6. 웹 리포트 만들기
# ------------------------------------------------------------
def downsample(y, target_points=2000):
    """
    파형 데이터는 수십만 개의 숫자라서 그대로 웹페이지에 넘기면 너무 크다.
    일정 간격으로 건너뛰며 점을 뽑아(다운샘플링) 개수를 줄인다. (그래프 모양은 거의 그대로)
    """
    step = max(1, len(y) // target_points)
    return y[::step].tolist()    # numpy 배열 → 파이썬 리스트 (JSON으로 저장하려면 필요)


def build_result_data(res, y_orig, sr_orig, original_times, original_name, seg_len, history_shown):
    """웹 리포트가 사용할 모든 데이터를 하나의 딕셔너리로 모은다."""
    return {
        "accuracy": round(res["accuracy"], 1),
        "avg_error_ms": round(res["avg_error_ms"], 1),
        "tolerance_ms": round(TOLERANCE * 1000),
        "original_name": original_name,
        "take_name": res["name"],
        "original_duration": float(len(y_orig) / sr_orig),
        "original_onsets": [float(t) for t in original_times],
        "original_waveform": downsample(y_orig),
        "user_duration": float(len(res["y"]) / res["sr"]),
        "user_onsets": [float(t) for t in res["user_times"]],
        "user_waveform": downsample(res["y"]),
        "matched": [{"original": m[0], "user": m[1], "error": m[2]} for m in res["matches"]],
        "missed": res["missed"],
        "extra": res["extra"],
        "segment_seconds": seg_len,
        "segments": res["segments"],
        "history": history_shown,
    }


def write_report(result_data):
    """
    분석 결과를 accuracy_viewer.html 안에 "이미 채워진 상태"로 심어서
    accuracy_report.html을 만들고, 기본 브라우저로 자동으로 연다.
    (뷰어 HTML은 EMBEDDED_DATA가 있으면 파일 선택 없이 바로 결과를 그린다)
    """
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False)
    print("\naccuracy_result.json 파일이 저장되었습니다.")

    if not os.path.exists(VIEWER_TEMPLATE):
        print(f"\n주의: {VIEWER_TEMPLATE} 파일이 같은 폴더에 없어서 웹 리포트 생성은 건너뜁니다.")
        print("accuracy_viewer.html을 이 스크립트와 같은 폴더에 넣어주세요.")
        return

    with open(VIEWER_TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()

    # json.dumps : 파이썬 데이터를 JSON "문자열"로 바꿈.
    # '</' 는 HTML 안에서 태그 닫힘으로 오해될 수 있어서 '<\/' 로 바꿔 안전하게 심는다.
    json_text = json.dumps(result_data, ensure_ascii=False).replace("</", "<\\/")
    embedded_script = f"<script>const EMBEDDED_DATA = {json_text};</script>"
    html = html.replace("</head>", embedded_script + "\n</head>", 1)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"결과 리포트가 생성되었습니다: {REPORT_FILE}")

    # as_uri() : 윈도우 경로를 브라우저가 이해하는 file:///C:/... 형태로 안전하게 변환
    webbrowser.open(pathlib.Path(REPORT_FILE).as_uri())
    print("브라우저에서 결과 화면이 자동으로 열립니다.")


def show_plot(res, y_orig, sr_orig, original_times):
    """원곡과 내 연주를 위아래로 비교하는 그래프를 저장하고 창으로 띄운다."""
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y_orig, sr=sr_orig, alpha=0.6, color='steelblue')
    plt.vlines(original_times, -1, 1, color='blue', linestyle='--', label='원곡 onset')
    plt.title("원곡")
    plt.legend()

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(res["y"], sr=res["sr"], alpha=0.6, color='indianred')
    plt.vlines([m[1] for m in res["matches"]], -1, 1, color='green', linestyle='--', label='맞은 음')
    plt.vlines(res["extra"], -1, 1, color='red', linestyle='--', label='엉뚱하게 친 음')
    plt.title(f"내 연주 — 정확도 {res['accuracy']:.1f}%")
    plt.legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)   # 위/아래 그래프 제목·라벨이 겹치지 않게 간격 확보
    plt.savefig(RESULT_PNG)
    plt.show()


# ------------------------------------------------------------
# 7. 실행부
# ------------------------------------------------------------
def main():
    options, take_names = parse_args(sys.argv[1:])

    original_path = resolve_path(options["original"])
    take_paths = [resolve_path(n) for n in take_names]
    require_file(original_path)
    for p in take_paths:
        require_file(p)

    if options["reset_history"] and os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        print("연습 기록을 초기화했습니다.")

    # --- 원곡은 한 번만 분석 ---
    original_name = os.path.basename(original_path)
    print(f"원곡 분석 중... ({original_name})")
    y_orig, sr_orig, original_times = get_onset_times(original_path)
    original_duration = float(len(y_orig) / sr_orig)
    seg_len = choose_segment_length(original_duration)
    print(f"  원곡에서 감지된 onset: {len(original_times)}개")

    # --- 연주(take)를 하나씩 분석하고 기록에 남김 ---
    history = load_history()
    last = None
    for i, take_path in enumerate(take_paths, start=1):
        print(f"\n내 연주 분석 중... ({i}/{len(take_paths)}) {os.path.basename(take_path)}")
        res = analyze_take(take_path, original_times, original_duration, seg_len)
        print_summary(res, len(original_times), seg_len)
        if not options["no_history"]:
            history.append(make_history_entry(res, original_name, seg_len))
        last = res

    if not options["no_history"]:
        save_history(history)

    # --- 이 곡에 대한 연습 기록만 골라서 웹 리포트에 전달 ---
    same_song = [h for h in history if h.get("original") == original_name]
    history_shown = same_song[-MAX_HISTORY_SHOWN:]
    if history_shown:
        trend = " → ".join(f"{h['accuracy']:.1f}%" for h in history_shown[-6:])
        print(f"\n연습 기록 {len(same_song)}회 (최근 정확도 추이): {trend}")

    # 마지막으로 분석한 연주의 결과를 웹 리포트로 보여줌
    result_data = build_result_data(last, y_orig, sr_orig, original_times,
                                    original_name, seg_len, history_shown)
    write_report(result_data)

    # 그래프 창 (닫아도 브라우저 리포트는 그대로 남아있음)
    show_plot(last, y_orig, sr_orig, original_times)


if __name__ == "__main__":
    main()
