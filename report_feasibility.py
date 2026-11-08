"""Render measured pilot results as plots and a local, self-contained HTML index.

No training, model inference, ROS or original-data modifications.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NAMES = {
    "opencv": "OpenCV",
    "unet_basic": "U-Net basic",
    "unet_robust": "U-Net robust",
    "mobile_sam_roi": "MobileSAM fixed ROI",
    "unet_robust_mobile_sam": "U-Net + MobileSAM",
}
COLORS = ["#626262", "#d18000", "#007c91", "#9e58a7", "#388843"]
CONDITIONS = {
    "clean": "가림 없음", "background": "배경 교체",
    "mid_occlusion_05": "중간 가림 5%", "mid_occlusion_10": "중간 가림 10%",
    "mid_occlusion_20": "중간 가림 20%", "mid_occlusion_30": "중간 가림 30%",
    "mid_occlusion_bg_10": "배경 + 가림 10%", "mid_occlusion_bg_20": "배경 + 가림 20%",
    "low_light": "저조도", "glare": "합성 반사광",
}


def read_json(path):
    return json.loads(path.read_text())


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(v))}</th>" for v in headers) + \
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) +
                                       "</tr>" for row in rows) + "</tbody></table>"


def report(root):
    data = read_json(root / "evaluation/summary.json")
    rows = data["results"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    lookup = {(r["method"], r["condition"]): r for r in rows}
    conditions = data["conditions"]
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    keys = ["clean", "mid_occlusion_05", "mid_occlusion_10", "mid_occlusion_20", "mid_occlusion_30"]
    for method, color in zip(methods, COLORS):
        for ax, field in zip(axes, ["dice_mean", "all_visible_components_recall_ge_80_rate"]):
            ax.plot([0, 5, 10, 20, 30], [lookup[method, k][field] * 100 for k in keys],
                    "o-", color=color, label=NAMES[method])
    for ax in axes:
        ax.set(xlabel="Synthetic centerline occlusion (%)", ylim=(-2, 103))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Visible pseudo-mask Dice (%)")
    axes[1].set_ylabel("All visible components recall >= 80% (%)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Held-out-candidate sequence: paired synthetic stress test (not manual ground truth)")
    fig.savefig(plots / "occlusion_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.7), layout="constrained")
    matrix = np.array([[lookup[m, c]["dice_mean"] * 100 for c in conditions] for m in methods])
    im = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
    ax.set_yticks(range(len(methods)), [NAMES[m] for m in methods])
    ax.set_xticks(range(len(conditions)), conditions, rotation=35, ha="right", fontsize=8)
    for i in range(len(methods)):
        for j in range(len(conditions)):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=9,
                    color="black" if matrix[i,j] > 65 else "white")
    fig.colorbar(im, ax=ax, label="Mean visible pseudo-mask Dice (%)", shrink=.8)
    fig.savefig(plots / "condition_matrix.png", dpi=160)
    plt.close(fig)

    timing = []
    with (root / "evaluation/per_frame.csv").open() as stream:
        per_frame = list(csv.DictReader(stream))
    for method in methods:
        values = [float(r["latency_ms"]) for r in per_frame if r["method"] == method]
        timing.append([NAMES[method], f"{np.median(values):.2f}", f"{np.percentile(values,95):.2f}"])
    metric_table = table(["조건"] + [NAMES[m] for m in methods],
                         [[CONDITIONS.get(c, c)] + [f"{lookup[m,c]['dice_mean']*100:.2f}%" for m in methods]
                          for c in conditions])
    detail_links = "".join(f'<li><a href="evaluation/{c}_overview.jpg">{CONDITIONS.get(c,c)}: 사전 선정 5프레임 비교</a></li>'
                           for c in conditions)
    live = read_json(root / "live_predictions/report.json") if (root / "live_predictions/report.json").exists() else None
    live_section = "<p>실제 카메라 추론 결과 미생성.</p>"
    if live:
        live_section = f"<p>새 RGB-D 촬영 {live['source_frames']}프레임, 원본 및 합성 가림 10·20%, 총 {live['inference_rows']}회 추론. " \
                       f"처리 예외 {len(live['errors'])}건. 직선 HRM이며 실제 가림은 가하지 않았다. " \
                       "수작업 정답이 없어 이 구간의 정확도 수치는 계산하지 않았다.</p>"
        for c in live["conditions"]:
            live_section += f'<h3>{CONDITIONS.get(c,c)}</h3><a href="live_predictions/{c}_overview.png">3프레임 전체 보기</a>' \
                            f'<img src="live_predictions/{c}_first_frame.png" alt="live {c}">'
    depth_section = ""
    depth_path = root / "depth_observed_preview/manifest.json"
    if depth_path.exists():
        depth_section = '<h2>Depth 연결 확인</h2><p>실제 촬영의 가림 없는 영상에 한해서 U-Net 마스크 → 보이는 영역의 skeleton → ' \
            '마스크 내부 유효 depth의 5×5 median → camera-frame XYZ를 저장했다. 합성 가림 RGB에 원래 depth를 붙이지 않았다. ' \
            '가려진 구간의 좌표를 복원한 결과가 아니며, 곡선·각도·FK 정량 검증은 이번 범위에 포함하지 않는다. ' \
            '마스크 자체가 가림 물체를 오검출하면 잘못된 depth가 들어갈 수 있으므로 실제 가림 실험이 필요하다.</p>' \
            '<p><b>실제 관찰:</b> 이번 20프레임에서도 밝은 주변 물체 오검출로 배경 depth가 포함되었다. ' \
            '3,978 skeleton 후보 중 3,881점이 유효 depth를 가졌지만, 전체 depth 범위는 0.113–0.663m였으며 ' \
            '이 점들이 모두 HRM에 속한다는 의미는 아니다. 결과는 수정하지 않고 보존했다. ' \
            '실시간 이식 전 여러 몸체 조각은 유지하면서 무관한 물체는 배제하는 대상 구분과 depth 이상치 검증이 필요하다.</p>' \
            '<a href="depth_observed_preview/manifest.json">Depth 결과 manifest</a>'
    baseline = lookup["opencv", "mid_occlusion_20"]["dice_mean"] * 100
    robust = lookup["unet_robust", "mid_occlusion_20"]["dice_mean"] * 100
    content = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<title>HRM segmentation feasibility — 2026-10-24</title>
<style>body{{font:16px/1.7 sans-serif;color:#152936;max-width:1200px;margin:36px auto;padding:0 24px}}
h1,h2{{line-height:1.3}}h2{{margin-top:36px}}img{{max-width:100%;margin:12px 0;border:1px solid #ddd}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #ddd;padding:7px 10px;text-align:right}}
th:first-child,td:first-child{{text-align:left}}th{{background:#edf3f5}}.note{{background:#fff4d9;padding:16px;border-left:5px solid #cf9200}}
code{{background:#eee;padding:2px 4px}}a{{color:#006b8a}}</style>
<h1>HRM 가림 대응 segmentation: feasibility 결과</h1>
<p>2026-10-24 · RTX 3060 · 오프라인 연구용 비교 · 로봇/모터 제어 변경 없음</p>
<div class="note"><b>해석 범위:</b> 기존 OpenCV 마스크를 기준으로 합성 가림 영역을 제거한 visible pseudo-label과 비교했다.
수작업 정답, 실제 물체 가림, 관절각 정확도 또는 임상 일반화의 검증 결과가 아니다.</div>
<h2>주요 관찰</h2>
<p>중간 20% 합성 가림에서 OpenCV의 평균 Dice는 {baseline:.2f}%, 증강 U-Net은 {robust:.2f}%였다.
가린 부분을 마스크로 메우는 것이 아니라 <b>가림 앞뒤에 실제로 보이는 몸체를 남기는 것</b>이 이 실험의 목표다.</p>
<p>이번 조건에서는 증강 U-Net 단독이 속도와 마스크 품질의 균형이 좋았다. 전체 ROI만 지정한 MobileSAM은
몸체 대신 배경을 선택했다. 이는 target prompt가 약한 이 구성의 결과이며 SAM 전체의 성능으로 일반화할 수 없다.
U-Net prompt를 사용한 SAM은 합성 배경에서 일치도를 조금 높였지만 처리 시간이 늘었다.</p>
<p><b>새 카메라 영상의 한계:</b> 증강 U-Net은 가림 전후의 몸체를 남겼지만 우측 상단의 밝은 주변 물체도
일부 오검출했다. 이 조각까지 SAM positive prompt로 전달되면서 hybrid가 배경을 크게 포함하는 사례가 관찰되었다.
따라서 이 모델을 검증 없이 실시간 형상/제어 파이프라인에 교체하지 않았다.</p>
<h2>동일 조건 비교</h2><p>별도 촬영 구간의 {data['records']}장 × {len(conditions)}조건 × {len(methods)}방법.
이미지별 foreground Dice를 평균했다. 배경 픽셀 정확도로 평가하지 않았다. 처리 예외 {len(data['errors'])}건.</p>
{metric_table}<img src="plots/occlusion_comparison.png"><img src="plots/condition_matrix.png">
<p>OpenCV 원본 조건의 100%는 같은 알고리즘으로 만든 pseudo-label을 재현한 결과다. 독립 정답에 대한 100% 정확도가 아니다.</p>
<p>오른쪽 그래프는 크기 32px 이상인 모든 visible component를 각각 80% 이상 회수했는지의 비율이다.
오검출은 이 지표만으로 판정할 수 없으므로 Dice·precision과 함께 봐야 하며, 중심선/3D 복원 성공률이 아니다.</p>
<h2>실험 방법</h2><ul>
<li>학습: 35deg_1 및 45deg_2에서 균등 추출 1,200장, 모델별 12 epoch. 동일 ResNet18 U-Net, ImageNet 초기화, 384 입력.</li>
<li>검증: 35deg_2의 64장 × 원본/배경+20% 가림. 검증 micro Dice로 checkpoint 선택. 테스트는 선택에 사용하지 않음.</li>
<li>테스트: 현재 split에 사용되지 않은 45deg_1의 96장. 같은 카메라/환경의 단일 구간이며 과거 개발 노출 여부는 미확인.</li>
<li>Basic: 약한 기하·조명 증강. Robust: 배경/배치/조명/가림/negative 등을 함께 사용. 가림 증강 단독의 인과적 효과를 분리한 실험은 아님.</li>
<li>가림 정도: pseudo-mask skeleton 중심선 길이의 5·10·20·30%. 양 끝을 보호하고 중간 몸체 폭을 가로지르는 불투명 가림을 합성. 실제 비율은 cases.json에 기록.</li>
<li>MobileSAM fixed ROI: 모든 프레임에 같은 전체 crop box를 제공. 대상 위치를 알려주는 강한 prompt가 아님.</li>
<li>U-Net + MobileSAM: 해당 관측 영상의 U-Net 예측으로 box와 내부 point prompt 생성. 정답이나 가림 mask를 prompt/결과 보정에 사용하지 않음.</li>
<li>SAM은 사전학습 모델을 추론만 했고, 별도 fine-tuning 및 YOLO segmentation 학습은 하지 않았다.</li>
</ul>
<h2>추론 시간</h2>{table(['방법','중앙값 (ms)','95백분위 (ms)'], timing)}
<p>RTX 3060, batch 1, CUDA 동기화, 전처리/추론/원본 크기 mask 복구 포함. 카메라·파일 I/O·ROS·3D 후처리 제외.
Hybrid는 U-Net 시간을 포함한다. 이 수치를 전체 ROS 파이프라인 FPS로 해석하면 안 된다.</p>
<h2>예측 비교 이미지</h2><p>초록=TP, 빨강=FP, 파랑=FN. 표본은 결과를 보기 전에 균등 선정했다.</p>
<ul>{detail_links}</ul><img src="evaluation/mid_occlusion_20_example_1.jpg">
<h2>새 카메라 영상 확인</h2>{live_section}{depth_section}
<h2>리뷰어 대응을 위한 다음 검증</h2><ol>
<li>중간 부분만 실제로 가리는 물체를 이용해 가림 없음/약함/중간/강함을 독립 촬영하고, 양 끝은 보이도록 유지한다.</li>
<li>각 조건에서 균등 표본을 수작업 visible mask로 검수해 Dice/IoU와 두 몸체 조각 유지 여부를 평가한다. 기저부/끝단 가림은 별도 실패 조건으로 보고한다.</li>
<li>정량 형상 평가에는 태그 기준 tip 위치·자세, angle/shape 오차와 결과 누락률을 함께 기록한다. hidden depth는 측정값으로 취급하지 않는다.</li>
<li>다른 시점/조명/배경/촬영 세션으로 일반화 확인. 합성 시험 성능만으로 실제 가림에 강건하다고 단정하지 않는다.</li>
</ol>
<h2>파일과 재현성</h2><ul>
<li><a href="evaluation/summary.csv">조건별 수치 CSV</a> / <a href="evaluation/per_frame.csv">프레임별 전체 수치</a> / <a href="evaluation/cases.json">동일 평가 조건</a></li>
<li><a href="unet_robust/config.json">Robust 학습 설정</a> / <a href="unet_robust/history.csv">학습 로그</a> / <a href="unet_robust/best.pt">선택 모델</a></li>
<li><a href="live_capture/manifest.json">새 RGB-D 촬영 시각·ROI·intrinsics</a></li>
<li>코드/환경/배경 출처 사본: provenance/. Endoscapes 배경의 CC BY-NC-SA 조건은 해당 출처 파일 참고.</li>
</ul>
<p>공식 구현: <a href="https://smp.readthedocs.io/en/latest/models.html">segmentation-models-pytorch</a>,
<a href="https://docs.ultralytics.com/models/mobile-sam/">Ultralytics MobileSAM</a>,
<a href="https://github.com/CAMMA-public/Endoscapes">Endoscapes</a>.</p></html>"""
    (root / "report.html").write_text(content, encoding="utf-8")
    print(root / "report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/feasibility_20261024"))
    report(parser.parse_args().run)
