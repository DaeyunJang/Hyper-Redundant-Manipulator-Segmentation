"""Rebuild the paper asset index from measured V3 CSVs; no model execution."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

from run_feasibility import labeled_tile


LABELS = {
    "clean": "원본",
    "base_near25": "Base 가까움 · 25°", "base_near45": "Base 가까움 · 45°",
    "base_near60": "Base 가까움 · 60°", "tip_near25": "Tip 가까움 · 25°",
    "tip_near45": "Tip 가까움 · 45°", "tip_near60": "Tip 가까움 · 60°",
    "bg_base_near45_multi3": "Base 가까움 45° + 배경·배치·세 줄 가림",
    "bg_tip_near45_multi3": "Tip 가까움 45° + 배경·배치·세 줄 가림",
    "multi3_30": "V2 회귀: 세 줄 30%", "bg_perspective_multi3": "V2 회귀: 배경·원근·세 줄 30%",
}


def generate(run):
    run = Path(run).resolve()
    evaluation = run / "segmentation_evaluation"
    with (evaluation / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    cases = json.loads((evaluation / "cases.json").read_text())
    configuration = json.loads((run / "unet_v3/config.json").read_text())
    completed = json.loads((run / "unet_v3/completed.json").read_text())
    by_key = {(r["condition"], r["method"]): r for r in rows}
    conditions = list(dict.fromkeys(r["condition"] for r in rows))
    table = []
    for condition in conditions:
        scores = [f"{100 * float(by_key[condition, method]['dice_mean']):.2f}%"
                  for method in ("opencv", "unet_v2", "unet_v3")]
        delta = 100 * (float(by_key[condition, "unet_v3"]["dice_mean"]) -
                       float(by_key[condition, "unet_v2"]["dice_mean"]))
        table.append("<tr><td>" + html.escape(LABELS.get(condition, condition)) + "</td>" +
                     "".join(f"<td>{score}</td>" for score in scores) + f"<td>{delta:+.2f}</td></tr>")
    sources = list(dict.fromkeys(c["source"]["rgb"] for c in cases))
    middle = sources[(len(sources) - 1) // 2]
    selected = {c["metadata"]["condition"]: c for c in cases if c["source"]["rgb"] == middle}
    display_conditions = ["clean", "base_near45", "base_near60", "tip_near45", "tip_near60"]
    sheets = [[], [], []]
    figure_sources = []
    for condition in display_conditions:
        item = selected[condition]
        rgb = cv2.cvtColor(cv2.imread(str(evaluation / item["rgb_path"])), cv2.COLOR_BGR2RGB)
        target = (cv2.imread(str(evaluation / item["mask_path"]), 0) > 0).astype(np.uint8)
        sheets[0].append(labeled_tile(rgb, np.zeros_like(target), condition))
        for line, method in enumerate(("unet_v2", "unet_v3"), start=1):
            pred = (cv2.imread(str(evaluation / item["prediction_paths"][method]), 0) > 0).astype(np.uint8)
            sheets[line].append(labeled_tile(rgb, pred, method, target))
        figure_sources.append(item)
    preview = np.vstack([np.hstack(line) for line in sheets])
    cv2.imwrite(str(run / "base_tip_perspective_comparison.png"), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    figure = dict(asset_path_base="segmentation_evaluation",
                  selection="fixed middle source from time-spaced test list; not chosen by performance",
                  path="base_tip_perspective_comparison.png", sources=figure_sources)
    (run / "figure_sources.json").write_text(json.dumps(figure, ensure_ascii=False, indent=2))
    links = []
    for condition in conditions:
        examples = sorted(evaluation.glob(condition + "_example_*.png"))
        links.append(f"<li>{html.escape(LABELS.get(condition, condition))}: " + " · ".join(
            f'<a href="{html.escape(str(path.relative_to(run)))}">{path.stem.rsplit("_", 1)[1]}</a>'
            for path in examples) + "</li>")
    timing = np.mean([float(r["latency_median_ms"]) for r in rows if r["method"] == "unet_v3"])
    errors = sum(int(r["errors"]) for r in rows)
    plain = [c for c in conditions if c.startswith(("base_near", "tip_near"))]
    composite = [c for c in conditions if c.startswith(("bg_base_near", "bg_tip_near"))]
    def score(condition, method="unet_v3"):
        return 100 * float(by_key[condition, method]["dice_mean"])
    interpretation = (f"원근 단독 조건의 V3 Dice는 {min(score(c) for c in plain):.2f}–"
                      f"{max(score(c) for c in plain):.2f}%, 배경·배치·가림 복합 조건은 "
                      f"{min(score(c) for c in composite):.2f}–{max(score(c) for c in composite):.2f}%였다. "
                      f"원근 단독에서 V2 대비 최대 향상은 {max(score(c)-score(c,'unet_v2') for c in plain):.2f}%p로 작았다. "
                      f"원본 조건은 V2 {score('clean','unet_v2'):.2f}% → V3 {score('clean'):.2f}%였다. "
                      "단순 원근 조건에서는 기존 OpenCV가 더 높은 pseudo-label Dice를 보였다. "
                      "학습 모델의 상대적 이점은 배경 변화와 다중 가림이 결합된 조건에서 확인되며, "
                      "모든 상황에서 더 정확하다고 해석하지 않는다.")
    output = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>HRM base/tip 원근 증강 및 논문용 결과 자료</title>
<style>body{{max-width:1400px;margin:32px auto;font:16px/1.65 sans-serif;padding:0 20px;color:#222}}
table{{border-collapse:collapse}}th,td{{padding:8px 14px;border:1px solid #ccc}}th{{background:#eef2f6}}
img{{max-width:100%;height:auto}}.note{{padding:15px;background:#fff1d5;border-left:5px solid #d89024}}</style>
<h1>HRM base/tip 원근 증강과 논문용 결과 기록</h1>
<div class="note">이 보고서는 합성 영상의 visible-mask 개발 평가이다. Pseudo-label은 OpenCV로 생성했으며
독립 수작업 정답이 아니다. 기울기 25/45/60°는 가상 평면의 기울기이며 실제 HRM 관절각이나 측정 시점이 아니다.
아래 RGB 변형에 원본 depth를 붙여 3D 곡선을 계산하지 않았다.</div>
<h2>추가한 변형</h2><p>원본 base→tip 방향을 기준으로 평면을 pinhole 투영하여 base-near와 tip-near를 각각 생성했다.
가까운 쪽이 크게, 먼 쪽이 작게 보이고 길이방향 투영도 짧아진다. RGB와 마스크는 동일한 homography를 사용한다.
이는 휘어진 원통의 새로운 3D 표면·자가가림·렌즈왜곡을 재현하는 물리 렌더링은 아니다.</p>
<p>원본 배경을 유지하는 조건은 원래 몸체 위치를 inpainting으로 지운 뒤 변형 몸체를 합성한다.
회색 띠 같은 합성 흔적이 일부 남아 있으며 실제 재촬영과 동일하지 않다. 배경 교체 조건도 별도로 보고한다.
<a href="planar_tilt_preview.png">base/tip 위치를 표시한 증강 미리보기</a>.</p>
<p>학습 이미지 {configuration['training_images']}장, {configuration['epochs']} epochs.
V2와 동일한 학습/검증 원본·예산·검증 조건·checkpoint 선택 지표를 유지했다. V2를 추가 학습한 것이 아니라
동일한 ImageNet 초기화 방식에서 V3를 별도 학습했다. 단일 seed이며 이미 검토한 개발용 구간이므로 최종 blind test가 아니다.</p>
<h2>동일한 영상에 대한 segmentation 비교</h2><p>{len(sources)}원본 × {len(conditions)}조건 × 3방법.
처리 예외 {errors}건. Dice는 이미지별 foreground Dice의 평균. V3 추론 중앙값의 조건별 평균은 약 {timing:.2f}ms
(카메라/ROS/3D 처리 제외). 모델은 검증 데이터로 선택했고 아래 시험 점수로 checkpoint를 고르지 않았다.</p>
<table><tr><th>조건</th><th>OpenCV</th><th>U-Net V2</th><th>U-Net V3</th><th>V3−V2 [%p]</th></tr>
{''.join(table)}</table><p>{interpretation}</p>
<h2>Base/tip 방향별 비교 이미지</h2><img src="base_tip_perspective_comparison.png">
<p>위: 입력 영상, 중간: V2, 아래: V3. 초록=의사정답과 중첩, 빨강=과검출(FP), 파랑=미검출(FN).
시간순 균등 표본의 가운데 프레임을 고정 선택했다.
<a href="figure_sources.json">원본 경로·timestamp·변환행렬·마스크/모델 예측 경로</a>.</p>
<h2>모든 조건의 고정 예시</h2><ul>{''.join(links)}</ul>
<h2>가림 이후 관절각·FK 자료는 별도 유지</h2>
<p>이미 측정한 원본 RGB-D에서 합성 가림만 적용한 결과를 별도로 재집계했다.
이 자료는 <b>U-Net V2의 원본 RGB-D 가림 시험</b>이며 V3 원근변형 이미지의 3D 결과가 아니다.
<a href="paper_pose_summary.html">각도·FK 논문용 요약과 수식</a> ·
<a href="paper_pose_summary.csv">mean/p95/max/1mm 이하 비율 CSV</a> ·
<a href="../occlusion_pose_20261024/report.html">기존 가림 RGB-D 전체 보고서</a>.</p>
<p>이 값은 각 방법의 가림 없는 같은 프레임 추정값 대비 변화량이다. 실제 정답에 대한 오차가 아니며,
평균이1mm 이하여도 모든 프레임이1mm 이하라는 뜻은 아니다. 가려진 곳은 관측점 없이 매끄러운 곡선으로
연결했으므로 숨은 실제 관절 형상을 복원했다는 증거로 쓰지 않는다. 합성 가림 영역의 depth는 알려진 마스크로 제외했다.</p>
<h2>파일 구조와 재현</h2><ul>
<li><a href="segmentation_evaluation/summary.csv">모델·조건별 요약</a> / <a href="segmentation_evaluation/per_frame.csv">프레임별 점수와 실패</a></li>
<li><a href="segmentation_evaluation/cases.json">전체 사례 색인</a>: 원본 RGB/마스크 SHA256, timestamp, seed, H, 가림, depth 불호환 여부.</li>
<li>segmentation_evaluation/assets/조건/원본이름/: rgb.png, visible_pseudo_mask.png, synthetic_occluder.png,
opencv_mask.png, unet_v2_mask.png, unet_v3_mask.png, provenance.json. 원본/기존 결과는 덮어쓰지 않았다.</li>
<li><a href="unet_v3/best.pt">V3 모델</a>, <a href="unet_v3/config.json">설정</a>, <a href="unet_v3/history.csv">학습 로그</a>.
검증 선택 Dice={completed['best_validation_dice']:.6f}; test 이미지별 평균과 집계 방식이 다르다.</li>
</ul><p>최종 논문 검증: 실제 종이/물체 가림, 수작업 visible-mask 검수, 보정된 marker tip reference 및
가림에 의한 유효하지만 잘못된 depth의 제거 성능을 별도로 평가해야 한다. 실제 로봇/ROS 모듈은 교체하지 않았다.</p>
</html>'''
    (run / "report.html").write_text(output, encoding="utf-8")
    print(run / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="runs/perspective_pose_followup_20261024")
    generate(parser.parse_args().run)
