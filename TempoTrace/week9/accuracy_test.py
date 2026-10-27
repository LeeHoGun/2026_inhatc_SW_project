# accuracy_test.py  —  TempoTrace (9주차 버전)
# ------------------------------------------------------------
# 목적: 원곡과 내 연주를 비교해서 박자 정확도(%)를 계산하고,
#       어느 시간대에서 틀렸는지, 연습할수록 어떻게 변하는지를 웹 리포트로 보여준다.
#
# 7주차 피드백을 반영한 변경점
#   1) 기준값(허용 오차)을 바꾸면 그 값이 "기본값"으로 저장되고, 연습 기록(결과 추이)도 새 기준으로 다시 계산된다.
#        - 기록마다 원본 onset 시각을 같이 저장하기 때문에, 기준을 바꿔도 과거 회차를 전부 다시 계산할 수 있다.
#        - 기본값은 tempotrace_settings.json 에 저장된다. (리포트 페이지에서 바꾸면 자동 저장)
#   2) 연습 기록(practice_history.json)은 개수 제한 없이 계속 누적된다.
#        - 한 줄씩 덧붙이는 방식이라, 기록이 수만 개로 늘어도 한 번 실행하는 시간이 거의 늘지 않는다.
#   3) 기준값을 리포트 페이지 안에서 바로 바꾸면(슬라이더/숫자 입력) 모든 결과가 즉시 다시 그려진다.
#
# 사용법
#   python accuracy_test.py                          original.wav 와 my_take.wav 를 비교 (VS Code ▶ 버튼도 동일)
#   python accuracy_test.py 연주1.wav 연주2.wav      여러 번의 연습(take)을 순서대로 분석하고 기록에 누적
#   python accuracy_test.py 연주.wav --original=곡.wav   원곡 파일을 직접 지정
#
#   옵션
#     --tolerance=200   기본 기준값(ms)을 200으로 바꾸고 저장한 뒤 실행
#     --no-history      이번 결과를 연습 기록에 남기지 않음
#     --reset-history   기존 연습 기록을 모두 지우고 새로 시작
#     --no-serve        리포트 서버를 켜지 않고, 파일(accuracy_report.html)만 만들어서 엽니다
#     --no-browser      브라우저를 자동으로 열지 않음
#     --plot            matplotlib 비교 그래프 창도 함께 띄움 (accuracy_result.png 저장)
#     --batch           서버·브라우저·그래프 없이 분석과 기록만 하고 끝냄 (반복 실행 검증용)
#     --quiet           터미널 출력을 최소화
# ------------------------------------------------------------

import sys
import os
import json
import math
import time
import datetime
import webbrowser
import pathlib
import threading
import http.server
from collections import deque

import numpy as np
import librosa

# ------------------------------------------------------------
# 설정값
# ------------------------------------------------------------

# 이 스크립트가 들어있는 폴더 (실행 위치와 상관없이 항상 같은 폴더에서 음원·뷰어 파일을 찾기 위함)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 기록·설정·결과 파일을 저장하는 폴더. 보통은 스크립트 폴더와 같고,
# 환경변수 TEMPOTRACE_DATA_DIR 로 바꿀 수 있다. (반복 실행 검증이 진짜 연습 기록을 건드리지 않게 하려는 용도)
DATA_DIR = os.environ.get("TEMPOTRACE_DATA_DIR") or BASE_DIR

DEFAULT_ORIGINAL = "original.wav"       # 원곡 음원 파일 (인자를 안 주면 이 파일 사용)
DEFAULT_TAKES = ["my_take.wav"]         # 내 연주 파일 (인자를 안 주면 이 파일 사용)

DEFAULT_TOLERANCE_MS = 150              # 설정 파일이 없을 때 쓰는 처음 기준값 (150ms)
MIN_TOLERANCE_MS = 10                   # 기준값으로 허용하는 범위
MAX_TOLERANCE_MS = 1000

SEGMENT_SECONDS = 2.0                   # 시간대별 분석에서 곡을 나누는 기본 구간 길이(초)
MAX_SEGMENTS = 15                       # 곡이 길어도 구간이 이 개수를 넘지 않도록 구간 길이를 자동으로 늘림
MAX_HISTORY_SHOWN = 30                  # 웹 리포트에 보여줄 최근 연습 기록 개수 (파일에는 전부 저장됨)

PREFERRED_PORT = 8765                   # 리포트 서버가 먼저 시도하는 포트 (사용 중이면 빈 포트를 자동으로 찾음)

HISTORY_FILE = os.path.join(DATA_DIR, "practice_history.json")      # 연습 기록 (계속 누적)
SETTINGS_FILE = os.path.join(DATA_DIR, "tempotrace_settings.json")  # 기본 기준값 저장 파일
RESULT_JSON = os.path.join(DATA_DIR, "accuracy_result.json")        # 최신 분석 결과
REPORT_FILE = os.path.join(DATA_DIR, "accuracy_report.html")        # 결과가 심어진 웹 리포트
RESULT_PNG = os.path.join(DATA_DIR, "accuracy_result.png")          # 비교 그래프 이미지 (--plot)
VIEWER_TEMPLATE = os.path.join(BASE_DIR, "accuracy_viewer.html")    # 웹 리포트의 뼈대 HTML

_quiet = False


def say(*args):
    """--quiet 가 아닐 때만 터미널에 출력한다."""
    if not _quiet:
        print(*args)


# ------------------------------------------------------------
# 1. 기본 기준값 설정 (tempotrace_settings.json)
# ------------------------------------------------------------
_settings_lock = threading.Lock()   # 서버 스레드와 메인 스레드가 동시에 저장해도 꼬이지 않게 함


def clamp_tolerance(ms):
    """기준값을 허용 범위(10~1000ms) 안의 정수로 맞춘다."""
    return int(max(MIN_TOLERANCE_MS, min(MAX_TOLERANCE_MS, round(float(ms)))))


def load_settings():
    """
    저장된 기본 기준값을 읽는다. 파일이 없거나 깨져 있으면 처음 기준값(150ms)을 쓴다.
    version : 기준값이 바뀔 때마다 1씩 늘어나는 번호.
              웹 리포트가 "내가 저장해 둔 값이 최신 설정과 같은 세대인지" 확인할 때 쓴다.
    """
    settings = {"tolerance_ms": DEFAULT_TOLERANCE_MS, "version": 0}
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            if "tolerance_ms" in raw:
                settings["tolerance_ms"] = clamp_tolerance(raw["tolerance_ms"])
            if isinstance(raw.get("version"), int):
                settings["version"] = raw["version"]
    except (OSError, ValueError, TypeError):
        pass
    return settings


def save_settings(tolerance_ms):
    """
    기본 기준값을 저장한다. 값이 실제로 바뀐 경우에만 version 을 올린다.
    임시 파일에 먼저 쓰고 한 번에 교체(os.replace)해서, 저장 도중 꺼져도 파일이 깨지지 않는다.
    """
    with _settings_lock:
        current = load_settings()
        new_value = clamp_tolerance(tolerance_ms)
        if new_value == current["tolerance_ms"] and os.path.exists(SETTINGS_FILE):
            return current
        new = {"tolerance_ms": new_value, "version": current["version"] + 1}
        tmp = SETTINGS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(new, f, ensure_ascii=False, indent=1)
        os.replace(tmp, SETTINGS_FILE)
        return new


# ------------------------------------------------------------
# 2. 명령줄 인자 처리
# ------------------------------------------------------------
def parse_args(argv):
    """
    터미널에서 입력한 인자를 해석한다.
      - '--'로 시작하는 것: 옵션
      - 나머지: 내 연주(take) 파일 이름들
    """
    opts = {
        "no_history": False, "reset_history": False, "original": DEFAULT_ORIGINAL,
        "tolerance": None, "plot": False, "serve": True, "browser": True, "quiet": False,
    }
    files = []
    for a in argv:
        if not a.startswith("--"):
            files.append(a)
        elif a == "--no-history":
            opts["no_history"] = True
        elif a == "--reset-history":
            opts["reset_history"] = True
        elif a == "--plot":
            opts["plot"] = True
        elif a == "--no-serve":
            opts["serve"] = False
        elif a == "--no-browser":
            opts["browser"] = False
        elif a == "--batch":
            opts["serve"] = False
            opts["browser"] = False
        elif a == "--quiet":
            opts["quiet"] = True
        elif a.startswith("--original="):
            opts["original"] = a.split("=", 1)[1]
        elif a.startswith("--tolerance="):
            try:
                raw_value = float(a.split("=", 1)[1])
                opts["tolerance"] = clamp_tolerance(raw_value)
                if opts["tolerance"] != round(raw_value):
                    print(f"기준값은 {MIN_TOLERANCE_MS}~{MAX_TOLERANCE_MS}ms 범위만 쓸 수 있어 {opts['tolerance']}ms 로 조정합니다.")
            except ValueError:
                print(f"기준값 형식이 올바르지 않아 무시합니다: {a}  (예: --tolerance=200)")
        else:
            print(f"알 수 없는 옵션은 무시합니다: {a}")
    return opts, (files if files else list(DEFAULT_TAKES))


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
# 3. onset 추출 / 비교 (핵심 로직)
# ------------------------------------------------------------
def get_onset_times(file_path):
    """
    오디오 파일 경로를 받아서 onset(소리 시작 지점) 시각 리스트를 반환하는 함수.
    원곡, 내 연주 각각에 대해 호출해서 onset 리스트를 얻는다.
    """
    # librosa.load : 오디오 파일을 숫자 배열(y)과 샘플레이트(sr)로 읽어온다. sr=None 이면 원본 그대로.
    y, sr = librosa.load(file_path, sr=None)

    # librosa.onset.onset_detect : "소리가 갑자기 커지는 지점"을 프레임 번호로 찾아준다.
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr)

    # librosa.frames_to_time : 프레임 번호를 "몇 초 지점인지"로 변환한다.
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    # 시각을 1ms(소수 셋째 자리) 단위로 반올림한다.
    # 기록 파일에 저장된 값과 계산에 쓰인 값을 똑같이 맞추고, 파일 크기도 줄이기 위함.
    # (기준값이 최소 10ms 이므로 1ms 단위 반올림은 결과에 영향을 주지 않는다)
    onset_times = np.round(np.asarray(onset_times, dtype=float), 3)

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
# 4. 시간대별 집계
# ------------------------------------------------------------
def choose_segment_length(duration):
    """
    곡을 몇 초 단위로 나눌지 정한다.
    기본은 SEGMENT_SECONDS(2초)이고, 구간이 MAX_SEGMENTS(15)개를 넘으면 구간 길이를 2배, 3배... 로 늘린다.
    (예: 12초 곡 → 2초 × 6구간 / 3분 곡 → 12초 × 15구간)
    """
    n = max(1, math.ceil(round(duration / SEGMENT_SECONDS, 3)))
    factor = math.ceil(n / MAX_SEGMENTS)
    return SEGMENT_SECONDS * factor


def segment_count(duration, seg_len):
    """구간 개수. 곡 끝에 구간 길이의 25% 미만 조각이 남으면 마지막 구간에 합친다. (웹페이지도 같은 규칙)"""
    n = max(1, math.ceil(round(duration / seg_len, 3)))
    if n > 1 and (duration - (n - 1) * seg_len) < 0.25 * seg_len:
        n -= 1
    return n


def compute_segments(original_times, missed, extra, duration, seg_len):
    """곡을 seg_len초 구간으로 나누고, 구간마다 원곡 음 개수 / 놓친 음 / 엉뚱하게 친 음을 센다."""
    n = segment_count(duration, seg_len)
    segments = [
        {"start": i * seg_len, "end": min((i + 1) * seg_len, duration),
         "total": 0, "missed": 0, "extra": 0}
        for i in range(n)
    ]

    def index_of(t):
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
# 5. 연습 기록 저장소 (practice_history.json)
#
#   파일 모양 (항상 올바른 JSON 배열이면서, 기록 한 건이 정확히 한 줄):
#       [
#       {"id":1, ...},
#       {"id":2, ...}
#       ]
#
#   새 기록을 추가할 때는 파일 전체를 읽고 다시 쓰지 않고, 맨 끝의 "]" 앞에 한 줄만 덧붙인다.
#   → 기록이 1만 개든 10만 개든 "추가" 자체는 항상 같은 시간이 걸리고,
#     덧붙이는 도중 문제가 생겨도 이전 기록은 건드리지 않는다.
# ------------------------------------------------------------
class HistoryFormatError(Exception):
    """기록 파일의 끝부분이 예상한 모양이 아닐 때 발생."""


def _entry_bytes(entry):
    """기록 한 건을 한 줄짜리 JSON(바이트)으로 만든다."""
    return json.dumps(entry, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _write_history_lines(lines):
    """줄 목록으로 기록 파일을 통째로 새로 쓴다. (임시 파일에 쓴 뒤 교체해서 안전하게)"""
    tmp = HISTORY_FILE + ".tmp"
    with open(tmp, "wb") as f:
        if lines:
            f.write(b"[\n" + b",\n".join(lines) + b"\n]\n")
        else:
            f.write(b"[\n]\n")
    os.replace(tmp, HISTORY_FILE)


def repair_history():
    """
    기록 파일이 중간에 끊기거나 일부 줄이 깨진 경우, 읽을 수 있는 줄만 모아서 다시 만든다.
    반환값: (살린 개수, 버린 개수)
    """
    good, bad = [], 0
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as f:
            for raw in f:
                line = raw.strip()
                if line.endswith(b","):
                    line = line[:-1]
                if not line.startswith(b"{"):
                    continue
                try:
                    json.loads(line)
                    good.append(line)
                except ValueError:
                    bad += 1
    _write_history_lines(good)
    return len(good), bad


def prepare_history():
    """
    매 실행 처음에 기록 파일 상태를 점검한다.
      - 비어 있으면 삭제
      - 7주차 이전 형식(여러 줄짜리 기록 / 원본 onset 없음)이면 이름을 바꿔 보관하고 새로 시작
        (원본 onset이 없어서 기준값을 바꿔도 다시 계산할 수 없기 때문)
      - 끝이 잘려 있으면 복구
    """
    if not os.path.exists(HISTORY_FILE):
        return
    if os.path.getsize(HISTORY_FILE) == 0:
        os.remove(HISTORY_FILE)
        return

    with open(HISTORY_FILE, "rb") as f:
        head = [f.readline().strip() for _ in range(4)]
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 64))
        tail = f.read().rstrip()

    if any(line == b"{" for line in head):          # 예전 형식: 기록이 여러 줄에 걸쳐 있음
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(DATA_DIR, f"practice_history.old_{stamp}.json")
        os.replace(HISTORY_FILE, backup)
        say(f"이전 형식의 연습 기록을 보관 파일로 옮기고 새로 시작합니다: {os.path.basename(backup)}")
        return

    if not tail.endswith(b"]"):                      # 쓰다가 끊긴 파일
        kept, dropped = repair_history()
        say(f"연습 기록 파일을 복구했습니다. (살린 기록 {kept}개, 버린 줄 {dropped}개)")


def append_history_entry(entry):
    """
    기록 한 건을 파일 맨 끝에 덧붙인다. (파일 전체를 다시 쓰지 않음)
    파일 끝의 "]" 바로 앞에서 잘라낸 뒤, ",\n{새 기록}\n]\n" 를 이어 쓴다.
    """
    data = _entry_bytes(entry)

    if not os.path.exists(HISTORY_FILE) or os.path.getsize(HISTORY_FILE) == 0:
        _write_history_lines([data])
        return

    try:
        with open(HISTORY_FILE, "r+b") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            back = min(size, 256)                    # 파일 끝 256바이트만 읽어서 모양을 확인
            f.seek(size - back)
            tail = f.read(back)

            stripped = tail.rstrip()                 # 끝의 공백·줄바꿈 제거
            if not stripped.endswith(b"]"):
                raise HistoryFormatError("끝 대괄호가 없습니다")
            body = stripped[:-1].rstrip()            # "]" 앞부분
            if not body:
                raise HistoryFormatError("배열이 시작되지 않았습니다")

            cut = size - back + len(body)            # 마지막 기록 바로 뒤의 위치
            last_char = body[-1:]
            if last_char == b"}":                    # 기록이 이미 있는 경우
                f.seek(cut)
                f.truncate()
                f.write(b",\n" + data + b"\n]\n")
            elif last_char == b"[":                  # 비어 있는 배열인 경우
                f.seek(cut)
                f.truncate()
                f.write(b"\n" + data + b"\n]\n")
            else:
                raise HistoryFormatError("예상하지 못한 모양입니다")
    except HistoryFormatError:
        kept, dropped = repair_history()
        say(f"연습 기록 파일을 복구했습니다. (살린 기록 {kept}개, 버린 줄 {dropped}개)")
        append_history_entry(entry)


def scan_history(original_name, k):
    """
    기록 파일을 앞에서부터 한 번 훑으면서 아래를 구한다. (메모리에는 최근 k건만 올려둠)
      total      : 전체 기록 개수 (모든 곡)
      song_total : 이 곡(original_name)의 기록 개수
      last_id    : 가장 마지막 기록의 id (새 기록의 번호를 이어 붙이기 위함)
      recent     : 이 곡의 최근 k건 (웹 리포트에 보여줄 것)
    """
    out = {"total": 0, "song_total": 0, "last_id": 0, "recent": [], "bad_lines": 0}
    if not os.path.exists(HISTORY_FILE):
        return out

    # 기록 한 줄 안에서 이 곡의 기록인지 빠르게 판별하기 위한 문자열 ("original":"곡이름")
    needle = b'"original":' + json.dumps(original_name, ensure_ascii=False).encode("utf-8")
    recent = deque(maxlen=k)
    last_line = None

    with open(HISTORY_FILE, "rb") as f:
        for raw in f:                                # 파일을 한 줄씩 읽음 (한 줄 = 기록 한 건)
            line = raw.strip()
            if line.endswith(b","):
                line = line[:-1]
            if not line.startswith(b"{"):            # "[", "]", 빈 줄
                continue
            if not line.endswith(b"}"):              # 쓰다가 끊긴 줄
                out["bad_lines"] += 1
                continue
            out["total"] += 1
            last_line = line
            if needle in line:
                out["song_total"] += 1
                recent.append(line)

    for line in recent:
        try:
            out["recent"].append(json.loads(line))
        except ValueError:
            out["bad_lines"] += 1
    if last_line is not None:
        try:
            out["last_id"] = int(json.loads(last_line).get("id", 0))
        except (ValueError, TypeError):
            out["last_id"] = out["total"]
    return out


def make_history_entry(entry_id, res, original_name, original_times, original_duration, seg_len, tolerance_ms):
    """
    이번 연습 결과를 기록 한 건(딕셔너리)으로 만든다.
    onset 시각 원본을 함께 저장하기 때문에, 나중에 기준값을 바꿔도 이 회차를 다시 채점할 수 있다.
    """
    return {
        "id": entry_id,                                  # 1부터 이어지는 고유 번호 (누락·중복 검사에 사용)
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original": original_name,
        "take": res["name"],
        "tolerance_ms": tolerance_ms,                    # 이 기록을 남길 당시의 기준값
        "accuracy": round(res["accuracy"], 1),           # 당시 기준으로 계산한 요약값
        "avg_error_ms": round(res["avg_error_ms"], 1),
        "missed": len(res["missed"]),
        "extra": len(res["extra"]),
        "original_duration": original_duration,
        "segment_seconds": seg_len,
        "original_onsets": [float(t) for t in original_times],
        "user_onsets": [float(t) for t in res["user_times"]],
    }


# ------------------------------------------------------------
# 6. 연주 하나를 분석
# ------------------------------------------------------------
def analyze_take(take_path, original_times, original_duration, seg_len, tolerance):
    """내 연주 파일 하나를 분석해서 결과를 딕셔너리로 돌려준다."""
    y, sr, user_times = get_onset_times(take_path)
    matches, missed, extra = compare_onsets(original_times, user_times, tolerance)

    accuracy = len(matches) / len(original_times) * 100 if len(original_times) > 0 else 0
    avg_error_ms = float(np.mean([m[2] for m in matches])) * 1000 if matches else 0.0
    segments = compute_segments(original_times, missed, extra, original_duration, seg_len)

    return {
        "path": take_path, "name": os.path.basename(take_path),
        "y": y, "sr": sr, "user_times": user_times,
        "matches": matches, "missed": missed, "extra": extra,
        "accuracy": accuracy, "avg_error_ms": avg_error_ms, "segments": segments,
    }


def print_summary(res, original_count, seg_len, tolerance_ms):
    """분석 결과를 터미널에 보기 좋게 출력한다."""
    say(f"  내 연주에서 감지된 onset: {len(res['user_times'])}개")
    say("\n" + "=" * 50)
    say(f"[{res['name']}]  (기준 {tolerance_ms}ms)")
    say(f"박자 정확도: {res['accuracy']:.1f}%  ({len(res['matches'])} / {original_count} 맞음)")
    say(f"평균 타이밍 오차: {res['avg_error_ms']:.1f}ms")
    say(f"놓친 음(missed): {len(res['missed'])}개")
    say(f"엉뚱하게 친 음(extra): {len(res['extra'])}개")
    say("=" * 50)

    if res["missed"]:
        say("\n놓친 지점:")
        for t in res["missed"]:
            say(f"  - {format_time(t)}")
    if res["extra"]:
        say("\n엉뚱하게 친 지점:")
        for t in res["extra"]:
            say(f"  - {format_time(t)}")

    bad = [s for s in res["segments"] if s["missed"] + s["extra"] > 0]
    say(f"\n시간대별 오류 ({seg_len:g}초 단위 구간):")
    if not bad:
        say("  모든 구간에서 오류가 없습니다.")
    for s in bad:
        say(f"  - {s['start']:g}~{s['end']:g}초  놓침 {s['missed']} · 엉뚱 {s['extra']}")


# ------------------------------------------------------------
# 7. 웹 리포트 만들기
# ------------------------------------------------------------
def downsample(y, target_points=2000):
    """파형 데이터를 일정 간격으로 건너뛰며 뽑아(다운샘플링) 웹페이지로 넘길 크기를 줄인다."""
    step = max(1, len(y) // target_points)
    return y[::step].tolist()


def build_result_data(res, y_orig, sr_orig, original_times, original_name, seg_len,
                      tolerance_ms, settings_version, scan):
    """웹 리포트가 사용할 모든 데이터를 하나의 딕셔너리로 모은다."""
    return {
        "accuracy": round(res["accuracy"], 1),
        "avg_error_ms": round(res["avg_error_ms"], 1),
        "tolerance_ms": tolerance_ms,                    # 이 리포트가 열릴 때의 기본 기준값
        "settings_version": settings_version,            # 기본값 설정의 세대 번호
        "original_name": original_name,
        "take_name": res["name"],
        "original_duration": round(float(len(y_orig) / sr_orig), 3),
        "original_onsets": [float(t) for t in original_times],
        "original_waveform": downsample(y_orig),
        "user_duration": round(float(len(res["y"]) / res["sr"]), 3),
        "user_onsets": [float(t) for t in res["user_times"]],
        "user_waveform": downsample(res["y"]),
        "matched": [{"original": m[0], "user": m[1], "error": m[2]} for m in res["matches"]],
        "missed": res["missed"],
        "extra": res["extra"],
        "segment_seconds": seg_len,
        "segments": res["segments"],
        "history": scan["recent"],                       # 이 곡의 최근 기록 (원본 onset 포함)
        "history_song_total": scan["song_total"],         # 이 곡의 전체 기록 개수
        "history_total": scan["total"],                   # 파일에 누적된 전체 기록 개수
    }


def write_report(result_data):
    """
    분석 결과를 accuracy_viewer.html 안에 "이미 채워진 상태"로 심어서 accuracy_report.html을 만든다.
    (뷰어 HTML은 EMBEDDED_DATA 가 있으면 파일 선택 없이 바로 결과를 그린다)
    """
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False)

    if not os.path.exists(VIEWER_TEMPLATE):
        say(f"\n주의: {VIEWER_TEMPLATE} 파일이 없어서 웹 리포트 생성은 건너뜁니다.")
        say("accuracy_viewer.html을 이 스크립트와 같은 폴더에 넣어주세요.")
        return False

    with open(VIEWER_TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()

    # '</' 는 HTML 안에서 태그 닫힘으로 오해될 수 있어서 '<\/' 로 바꿔 안전하게 심는다.
    json_text = json.dumps(result_data, ensure_ascii=False).replace("</", "<\\/")
    html = html.replace("</head>", f"<script>const EMBEDDED_DATA = {json_text};</script>\n</head>", 1)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(html)
    say(f"\n결과 리포트가 생성되었습니다: {REPORT_FILE}")
    return True


# ------------------------------------------------------------
# 8. 리포트 서버 (내 컴퓨터 안에서만 열리는 작은 웹 서버)
#
#   리포트 페이지에서 기준값을 바꾸면, 그 값을 이 서버가 받아서 tempotrace_settings.json 에 저장한다.
#   (웹페이지는 혼자서 컴퓨터의 파일을 바꿀 수 없기 때문에 서버가 대신 저장해 주는 역할)
#   127.0.0.1(내 컴퓨터)에서만 접속할 수 있고, 외부 사이트에서 보낸 요청은 거부한다.
# ------------------------------------------------------------
class ReportHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):      # 접속 기록을 터미널에 출력하지 않음
        pass

    def _reply(self, code, payload, content_type="application/json; charset=utf-8"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self):
        # Host 헤더가 내 컴퓨터 주소가 아니면 거부 (다른 사이트가 우리 서버를 가리키게 하는 공격 방지)
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self._host_ok():
            return self._reply(403, {"error": "허용되지 않은 주소입니다"})
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html", "/accuracy_report.html"):
            try:
                with open(REPORT_FILE, "rb") as f:
                    return self._reply(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                return self._reply(404, {"error": "리포트 파일이 없습니다"})
        if path == "/api/settings":
            return self._reply(200, load_settings())
        return self._reply(404, {"error": "찾을 수 없습니다"})

    def do_POST(self):
        if not self._host_ok():
            return self._reply(403, {"error": "허용되지 않은 주소입니다"})
        if self.path.split("?", 1)[0] != "/api/settings":
            return self._reply(404, {"error": "찾을 수 없습니다"})

        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            return self._reply(403, {"error": "다른 사이트의 요청은 받지 않습니다"})
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._reply(415, {"error": "JSON 형식으로 보내야 합니다"})

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._reply(400, {"error": "요청 길이가 올바르지 않습니다"})
        if length <= 0 or length > 1024:
            return self._reply(400, {"error": "요청 크기가 올바르지 않습니다"})

        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))["tolerance_ms"]
        except (ValueError, KeyError, TypeError):
            return self._reply(400, {"error": "tolerance_ms 값이 필요합니다"})
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not (MIN_TOLERANCE_MS <= value <= MAX_TOLERANCE_MS):
            return self._reply(400, {"error": f"기준값은 {MIN_TOLERANCE_MS}~{MAX_TOLERANCE_MS}ms 사이여야 합니다"})

        return self._reply(200, save_settings(value))


def start_server():
    """리포트 서버를 시작하고 (서버, 포트)를 돌려준다. 기본 포트가 사용 중이면 빈 포트를 자동으로 찾는다."""
    for port in (PREFERRED_PORT, 0):
        try:
            server = http.server.ThreadingHTTPServer(("127.0.0.1", port), ReportHandler)
            break
        except OSError:
            continue
    else:
        return None, None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


# ------------------------------------------------------------
# 9. 그래프 창 (--plot 옵션일 때만)
# ------------------------------------------------------------
def show_plot(res, y_orig, sr_orig, original_times):
    """원곡과 내 연주를 위아래로 비교하는 그래프를 저장하고 창으로 띄운다."""
    import matplotlib
    import matplotlib.pyplot as plt
    import librosa.display

    # 그래프 한글 깨짐 방지: 윈도우(맑은 고딕) → 맥(AppleGothic) → 리눅스(나눔고딕) 순서로 찾음
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['font.sans-serif'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

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
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(RESULT_PNG)
    plt.show()


# ------------------------------------------------------------
# 10. 실행부
# ------------------------------------------------------------
def main(argv=None):
    """
    프로그램의 전체 흐름. argv 를 직접 넘기면 다른 파이썬 코드(예: stress_test.py)에서도 호출할 수 있다.
    """
    global _quiet
    opts, take_names = parse_args(sys.argv[1:] if argv is None else argv)
    _quiet = opts["quiet"]

    # --- 기본 기준값 결정: --tolerance 를 주면 그 값을 저장하고, 아니면 저장된 기본값을 사용 ---
    if opts["tolerance"] is not None:
        settings = save_settings(opts["tolerance"])
        say(f"기본 기준값을 {settings['tolerance_ms']}ms 로 저장했습니다.")
    else:
        settings = load_settings()
    tolerance_ms = settings["tolerance_ms"]
    tolerance = tolerance_ms / 1000

    original_path = resolve_path(opts["original"])
    take_paths = [resolve_path(n) for n in take_names]
    require_file(original_path)
    for p in take_paths:
        require_file(p)

    os.makedirs(DATA_DIR, exist_ok=True)
    if opts["reset_history"] and os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        say("연습 기록을 초기화했습니다.")
    prepare_history()

    # --- 원곡은 한 번만 분석 ---
    original_name = os.path.basename(original_path)
    say(f"원곡 분석 중... ({original_name})")
    y_orig, sr_orig, original_times = get_onset_times(original_path)
    original_duration = round(float(len(y_orig) / sr_orig), 3)
    seg_len = choose_segment_length(original_duration)
    say(f"  원곡에서 감지된 onset: {len(original_times)}개")
    say(f"  기준값(허용 오차): {tolerance_ms}ms")

    # --- 기록 파일은 이번 실행에서 딱 한 번만 훑는다 (번호 이어붙이기 + 개수 집계 + 최근 기록 읽기) ---
    scan = scan_history(original_name, MAX_HISTORY_SHOWN)
    next_id = scan["last_id"] + 1

    # --- 연주(take)를 하나씩 분석하고, 기록 파일 맨 끝에 덧붙임 ---
    last = None
    for i, take_path in enumerate(take_paths, start=1):
        say(f"\n내 연주 분석 중... ({i}/{len(take_paths)}) {os.path.basename(take_path)}")
        res = analyze_take(take_path, original_times, original_duration, seg_len, tolerance)
        print_summary(res, len(original_times), seg_len, tolerance_ms)
        if not opts["no_history"]:
            entry = make_history_entry(
                next_id, res, original_name, original_times, original_duration, seg_len, tolerance_ms)
            append_history_entry(entry)
            next_id += 1
            # 이미 읽어 둔 결과에 방금 추가한 기록을 반영한다. (파일을 다시 훑지 않아도 됨)
            scan["total"] += 1
            scan["song_total"] += 1
            scan["last_id"] = entry["id"]
            scan["recent"] = (scan["recent"] + [entry])[-MAX_HISTORY_SHOWN:]
        last = res

    # --- 이 곡의 최근 기록을 터미널에 요약하고, 웹 리포트에 전달 ---
    if scan["song_total"]:
        trend = " → ".join(f"{h['accuracy']:.1f}%" for h in scan["recent"][-6:])
        say(f"\n연습 기록: 이 곡 {scan['song_total']}회 / 전체 {scan['total']}회 누적 (최근 정확도 추이): {trend}")
    if scan["bad_lines"]:
        say(f"주의: 읽을 수 없는 기록 {scan['bad_lines']}줄이 있었습니다.")

    result_data = build_result_data(last, y_orig, sr_orig, original_times, original_name,
                                    seg_len, tolerance_ms, settings["version"], scan)
    made = write_report(result_data)

    # --- 리포트 열기: 서버 모드(기본) 또는 파일 모드 ---
    server = None
    if made and opts["serve"]:
        server, port = start_server()
        if server is None:
            say("리포트 서버를 시작하지 못해 파일 모드로 엽니다.")
    if server is not None:
        url = f"http://127.0.0.1:{port}/"
        say(f"리포트 서버 실행 중: {url}")
        if opts["browser"]:
            webbrowser.open(url)
            say("브라우저에서 결과 화면이 자동으로 열립니다.")
    elif made and opts["browser"]:
        webbrowser.open(pathlib.Path(REPORT_FILE).as_uri())   # file:///C:/... 형태로 안전하게 변환
        say("브라우저에서 결과 화면이 자동으로 열립니다.")

    if opts["plot"]:
        show_plot(last, y_orig, sr_orig, original_times)

    # --- 서버 모드: 사용자가 Ctrl+C 를 누를 때까지 유지 ---
    if server is not None:
        say("\n페이지에서 기준값을 바꾸면 자동으로 기본값으로 저장됩니다.")
        say("종료하려면 이 창에서 Ctrl+C 를 누르세요.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.shutdown()
            say("\n서버를 종료했습니다.")

    return {"history_total": scan["total"], "song_total": scan["song_total"],
            "tolerance_ms": tolerance_ms, "report": REPORT_FILE if made else None}


if __name__ == "__main__":
    main()
