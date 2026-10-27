# stress_test.py  —  TempoTrace 반복 실행 검증
# ------------------------------------------------------------
# 목적: accuracy_test.py 를 아주 많이(기본 10,000번) 실행해도
#       practice_history.json 에 기록이 빠짐없이 누적되는지, 실행이 점점 느려지지 않는지 확인한다.
#
# 사용법
#   python stress_test.py                10,000번 실행 (오디오 분석 결과는 재사용해서 빠르게)
#   python stress_test.py 500            500번만 실행
#   python stress_test.py 10000 --full   매번 오디오를 새로 분석 (실제 실행과 같지만 오래 걸림)
#   python stress_test.py 10000 --keep   검증이 끝난 뒤에도 검증용 폴더(stress_data)를 남김
#
# 검증은 stress_data 라는 별도 폴더에서만 이루어지므로, 진짜 연습 기록(practice_history.json)은 건드리지 않는다.
# ------------------------------------------------------------

import os
import sys
import io
import json
import time
import shutil
import contextlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRESS_DIR = os.path.join(BASE_DIR, "stress_data")

# accuracy_test 를 불러오기 "전에" 기록 폴더를 검증용 폴더로 바꿔야 한다. (불러오는 순간 경로가 정해지기 때문)
os.environ["TEMPOTRACE_DATA_DIR"] = STRESS_DIR
sys.path.insert(0, BASE_DIR)
import accuracy_test as at          # noqa: E402


def parse():
    runs, full, keep = 10000, False, False
    for a in sys.argv[1:]:
        if a == "--full":
            full = True
        elif a == "--keep":
            keep = True
        elif a.isdigit():
            runs = int(a)
        else:
            print(f"알 수 없는 인자는 무시합니다: {a}")
    return runs, full, keep


def mb(n):
    return f"{n / 1024 / 1024:.2f}MB"


def main():
    runs, full, keep = parse()
    print("=" * 62)
    print(f" TempoTrace 반복 실행 검증  —  {runs:,}회")
    print("=" * 62)

    # 검증용으로 번갈아 쓸 연주 파일 (스크립트 폴더에 있는 것만)
    candidates = ["my_take.wav", "my_take_2.wav", "my_take_3.wav", "my_take_4.wav"]
    takes = [t for t in candidates if os.path.exists(os.path.join(BASE_DIR, t))]
    if not takes or not os.path.exists(os.path.join(BASE_DIR, at.DEFAULT_ORIGINAL)):
        print("original.wav 와 my_take.wav 가 이 폴더에 있어야 합니다.")
        sys.exit(1)

    shutil.rmtree(STRESS_DIR, ignore_errors=True)        # 항상 빈 상태에서 시작
    os.makedirs(STRESS_DIR)

    if not full:
        # 같은 음원을 매번 다시 분석하면 시간이 오래 걸리므로, 분석 결과(onset)는 한 번만 구해서 재사용한다.
        # 기록 저장·비교·집계·리포트 생성 등 나머지는 실행마다 전부 그대로 수행한다.
        original_fn = at.get_onset_times
        cache = {}

        def cached(path):
            if path not in cache:
                cache[path] = original_fn(path)
            return cache[path]
        at.get_onset_times = cached

    print(f" 방식: {'매번 오디오 분석(--full)' if full else '오디오 분석 결과 재사용(기본)'} | 연주 파일 {len(takes)}개를 번갈아 사용")
    print(f" 검증 폴더: {STRESS_DIR}\n")

    step = max(1, runs // 10)
    done = 0
    block = []
    block_means = []
    t_start = time.perf_counter()
    try:
        for i in range(1, runs + 1):
            take = takes[(i - 1) % len(takes)]
            s = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):     # 실행 중 출력은 숨김
                at.main(["--batch", "--quiet", take])
            block.append(time.perf_counter() - s)
            done = i
            if i % step == 0 or i == runs:
                mean = sum(block) / len(block)
                block_means.append(mean)
                size = os.path.getsize(at.HISTORY_FILE)
                print(f"  {i:>7,}회 완료 | 최근 {len(block):,}회 평균 {mean * 1000:6.1f}ms/회 | 기록 파일 {mb(size)}")
                block = []
    except KeyboardInterrupt:
        print(f"\n  중단됨 — {done:,}회까지 실행한 결과로 검증합니다.")
    elapsed = time.perf_counter() - t_start
    n = done

    # ------------------------------------------------------------
    # 검증
    # ------------------------------------------------------------
    print("\n" + "-" * 62)
    print(f" 검증 중... (총 {n:,}회 실행, 걸린 시간 {elapsed:.1f}초)")
    checks = []

    size = os.path.getsize(at.HISTORY_FILE)
    try:
        with open(at.HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)                        # 파일 전체가 올바른 JSON 배열인지
        valid = isinstance(data, list)
    except ValueError:
        data, valid = [], False
    checks.append(("practice_history.json 이 올바른 JSON 배열", valid))
    checks.append((f"기록 개수가 실행 횟수와 같음 ({len(data):,} / {n:,})", len(data) == n))

    ids = [e.get("id") for e in data]
    checks.append(("id 가 1부터 끊김 없이 이어짐 (누락 없음)", ids == list(range(1, n + 1))))
    checks.append(("중복된 id 없음", len(set(ids)) == len(ids)))

    required = ("id", "timestamp", "original", "take", "tolerance_ms", "accuracy", "original_onsets", "user_onsets")
    checks.append(("모든 기록에 필수 항목이 들어 있음", all(all(k in e for k in required) for e in data)))
    checks.append(("저장 순서가 실행 순서와 같음", [e.get("take") for e in data] == [takes[i % len(takes)] for i in range(n)]))

    with open(at.RESULT_JSON, "r", encoding="utf-8") as f:
        result = json.load(f)
    shown = len(result["history"])
    checks.append((f"리포트에는 최근 {at.MAX_HISTORY_SHOWN}건만 담김 ({shown}건) — 리포트 용량이 일정", shown == min(n, at.MAX_HISTORY_SHOWN)))
    checks.append(("리포트가 누적된 전체 개수를 알고 있음", result["history_total"] == n))
    report_kb = os.path.getsize(at.REPORT_FILE) / 1024

    # 속도: 처음 구간과 마지막 구간의 평균 실행 시간 비교
    first_mean, last_mean = block_means[0], block_means[-1]
    checks.append((f"마지막 구간 실행 시간이 1초 이내 ({last_mean * 1000:.0f}ms)", last_mean < 1.0))

    # 기록 파일에 한 건 추가하는 데 걸리는 시간 (기록이 n건 쌓인 상태에서) — 복사본으로 측정해서 진짜 기록은 그대로 둠
    probe = os.path.join(STRESS_DIR, "probe.json")
    shutil.copyfile(at.HISTORY_FILE, probe)
    real_path = at.HISTORY_FILE
    at.HISTORY_FILE = probe
    sample = dict(data[-1]) if data else {}
    t = time.perf_counter()
    for k in range(20):
        sample["id"] = n + 1 + k
        at.append_history_entry(sample)
    append_ms = (time.perf_counter() - t) / 20 * 1000
    t = time.perf_counter()                            # 비교용: 파일 전체를 읽고 다시 쓰는 방식이었다면?
    with open(probe, "r", encoding="utf-8") as f:
        tmp_data = json.load(f)
    with open(probe + ".rewrite", "w", encoding="utf-8") as f:
        json.dump(tmp_data, f, ensure_ascii=False)
    rewrite_ms = (time.perf_counter() - t) * 1000
    at.HISTORY_FILE = real_path

    t = time.perf_counter()
    scan = at.scan_history(at.DEFAULT_ORIGINAL, at.MAX_HISTORY_SHOWN)
    scan_ms = (time.perf_counter() - t) * 1000

    ok_all = all(ok for _, ok in checks)
    for name, ok in checks:
        print(f"  [{'통과' if ok else '실패'}] {name}")

    print("\n" + "=" * 62)
    print(f" 결과: {'모든 검증 통과' if ok_all else '실패한 항목이 있습니다'}")
    print("=" * 62)
    print(f"  실행 횟수 / 누적된 기록      : {n:,}회 / {len(data):,}건")
    print(f"  기록 파일 크기               : {mb(size)}  (기록 1건 평균 {size / max(1, len(data)):,.0f}바이트)")
    print(f"  1회 실행 평균 시간           : 처음 구간 {first_mean * 1000:.1f}ms → 마지막 구간 {last_mean * 1000:.1f}ms")
    print(f"  기록 {len(data):,}건 상태에서 1건 추가 : {append_ms:.2f}ms  (파일 전체를 다시 쓰는 방식이었다면 약 {rewrite_ms:.0f}ms)")
    print(f"  최근 기록 {len(scan['recent'])}건 읽기 + 개수 집계 : {scan_ms:.0f}ms")
    print(f"  웹 리포트(accuracy_report.html) 크기 : {report_kb:.0f}KB  (기록이 늘어도 일정)")

    if keep:
        print(f"\n  검증용 폴더를 남겨두었습니다: {STRESS_DIR}")
    else:
        shutil.rmtree(STRESS_DIR, ignore_errors=True)
        print("\n  검증용 폴더(stress_data)는 삭제했습니다. (남기려면 --keep)")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
