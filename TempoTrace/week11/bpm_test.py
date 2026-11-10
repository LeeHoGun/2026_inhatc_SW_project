# bpm_test.py  —  BPM 모드(11주차)가 얼마나 믿을 만한지 숫자로 확인하는 시험
# ------------------------------------------------------------
# 목적: 실제 음원 없이, 무작위로 만든 "리듬"으로 BPM 모드의 핵심 기능을 시험한다. (몇 초면 끝남)
#       무작위 값의 시작점을 고정(seed)했기 때문에, 누가 실행해도 결과가 항상 같게 나온다.
#
# 시험 항목
#   1) BPM 자동 측정   : 리듬 모양별로, 측정한 BPM 이 정답과 맞는지 (정답이 2배/절반 후보에 들어 있는지도)
#   2) 점수 읽는 법    : 박자가 몇 ms 흔들리면 정확도가 몇 % 나오는지, 아무렇게나 쳐도 몇 %까지 나오는지
#   3) 눈금 위치 보정  : 녹음이 시작된 위치가 제각각이어도 같은 점수가 나오는지
#
# 사용법:  python bpm_test.py
# ------------------------------------------------------------

import os
import sys
import numpy as np

# accuracy_test 를 불러올 때 기록·설정 파일이 진짜 폴더에 생기지 않도록 임시 위치로 돌린다.
os.environ["TEMPOTRACE_DATA_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpm_test_tmp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import accuracy_test as at          # noqa: E402


def make_rhythm(rng, pattern, bpm, duration=30.0, jitter=0.015, extra=0.1):
    """
    무작위 리듬의 소리 시각(초)을 만든다.
      quarter     : 매 박마다 한 번 (4분음표)
      eighth_even : 박과 반 박(8분음표)에 항상 (고르게 이어진 8분음표)
      eighth_mix  : 박은 항상, 반 박은 절반 확률
      syncop      : 박은 70%, 반 박은 30% 확률 (엇박이 섞인 리듬)
    jitter = 타이밍 흔들림(초), extra = 엉뚱한 소리의 비율
    """
    beat = 60.0 / bpm
    start = rng.uniform(0.2, 1.5)
    times = []
    for k in range(int((duration - start) / beat)):
        b = start + k * beat
        if pattern == "quarter":
            times.append(b)
        elif pattern == "eighth_even":
            times += [b, b + beat / 2]
        elif pattern == "eighth_mix":
            times += [b] + ([b + beat / 2] if rng.random() < 0.5 else [])
        elif pattern == "syncop":
            if rng.random() < 0.7:
                times.append(b)
            if rng.random() < 0.3:
                times.append(b + beat / 2)
    times = np.array(times) + rng.normal(0, jitter, len(times))
    times = np.concatenate([times, rng.uniform(0, duration, int(extra * len(times)))])
    return np.sort(times[(times >= 0) & (times < duration)])


def near(a, b, tol=0.015):
    """두 BPM 이 1.5% 이내로 같은지"""
    return abs(a / b - 1) < tol


def test_bpm_estimate():
    rng = np.random.default_rng(5)
    print("1) BPM 자동 측정 (패턴마다 200번, 박자 60~180 BPM, 흔들림 15ms, 엉뚱한 소리 10%)")
    print("   리듬 모양          | 정확히 맞음 | 후보(2배·절반) 안에 정답")
    for pattern, label in (("quarter", "4분음표"), ("eighth_mix", "8분음표가 섞임"), ("syncop", "엇박이 섞임"), ("eighth_even", "고른 8분음표")):
        exact = listed = 0
        for _ in range(200):
            true_bpm = float(rng.uniform(60, 180))
            est = at.estimate_bpm(make_rhythm(rng, pattern, true_bpm))
            if est is None:
                continue
            exact += near(est["bpm"], true_bpm)
            listed += any(near(x, true_bpm) for x in [est["bpm"]] + est["alternatives"])
        print(f"   {label:<16} | {exact / 2:8.1f}%  | {listed / 2:8.1f}%")
    print("   (고르게 이어진 8분음표는 '4분음표를 2배 빠르게 친 것'과 구분이 안 돼서, 후보를 보고 사람이 골라야 합니다)")


def test_score_meaning():
    rng = np.random.default_rng(7)
    bpm, div, tol_ms = 100, 1, 50                       # 100 BPM, 4분음표 눈금, 기준값 50ms
    period = at.bpm_period(bpm, div)
    print(f"\n2) 점수 읽는 법 (BPM {bpm}, 4분음표 눈금, 기준값 {tol_ms}ms, 소리 24개, 200번 평균)")
    print(f"   우연히 맞을 확률(기준선) = {at.chance_percent(tol_ms, bpm, div):.0f}%")
    print("   박자 흔들림(표준편차) | 정확도 평균 | 눈금 집중도 평균")
    for jitter_ms in (10, 20, 30, 50, 80):
        accs, concs = [], []
        for _ in range(200):
            t = 0.4 + np.arange(24) * period + rng.normal(0, jitter_ms / 1000, 24)
            phase, conc = at.estimate_phase(t, period)
            devs = at.grid_deviations(t, bpm, div, round(phase, 3))
            accs.append(float((np.abs(devs) <= tol_ms / 1000).mean() * 100)); concs.append(conc)
        print(f"   {jitter_ms:>12}ms       | {np.mean(accs):8.1f}%   | {np.mean(concs):8.2f}")
    accs, concs = [], []
    for _ in range(200):                                # 아무렇게나 친 연주
        t = np.sort(rng.uniform(0.4, 0.4 + 24 * period, 24))
        phase, conc = at.estimate_phase(t, period)
        devs = at.grid_deviations(t, bpm, div, round(phase, 3))
        accs.append(float((np.abs(devs) <= tol_ms / 1000).mean() * 100)); concs.append(conc)
    print(f"   아무렇게나 침         | {np.mean(accs):8.1f}%   | {np.mean(concs):8.2f}   (정확도 상위 5%: {np.percentile(accs, 95):.0f}%)")
    print("   → 아무렇게나 쳐도 눈금 위치를 자동으로 맞추는 만큼 기준선보다 높게 나옵니다.")
    print("     그래서 정확도만 보지 말고 '눈금 집중도'(0.5 미만이면 눈금에 잘 안 모인 것)도 함께 봐야 합니다.")


def test_phase_independence():
    rng = np.random.default_rng(11)
    bpm, div, tol = 100, 1, 0.05
    period = at.bpm_period(bpm, div)
    base = 0.4 + np.arange(24) * period + rng.normal(0, 0.02, 24)       # 같은 연주
    print("\n3) 눈금 위치 보정 (같은 연주를 녹음 시작 위치만 바꿔서 분석)")
    print("   녹음이 일찍 시작한 시간 | 정확도")
    for shift in (0.0, 0.17, 0.33, 0.9, 1.7, 3.1):
        t = base + shift
        phase, _ = at.estimate_phase(t, period)
        devs = at.grid_deviations(t, bpm, div, round(phase, 3))
        print(f"   {shift:>14.2f}초        | {float((np.abs(devs) <= tol).mean() * 100):5.1f}%")
    print("   → 시작 위치가 달라도 같은 점수가 나와야 합니다. (눈금 위치를 자동으로 맞추기 때문)")


if __name__ == "__main__":
    test_bpm_estimate()
    test_score_meaning()
    test_phase_independence()
