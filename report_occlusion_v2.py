"""Create measured RGB-mask and paired RGB-D shape sensitivity report."""
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


NAMES = {"opencv":"OpenCV", "unet_v1":"U-Net V1", "unet_v2":"U-Net V2"}
COLORS = {"opencv":"#767676", "unet_v1":"#d17a00", "unet_v2":"#007d91"}
CONDITIONS = {"clean":"가림 없음", "thin":"얇은 선 약 2%", "thick10":"한 줄 10%",
              "thick20":"한 줄 20%", "thick30":"한 줄 30%", "multi3_15":"세 줄 합계 15%",
              "multi3_30":"세 줄 합계 30%", "perspective":"완만한 원근 변형",
              "bg_perspective":"배경 + 위치/회전/크기/원근", "bg_perspective_multi3":"위 복합 조건 + 세 줄 30%"}
DATASETS = {"live_black":"새 검정 배경·거의 직선", "archive_45deg_1":"저장된 굽힘 운동 구간"}


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    cells = lambda tag, values: "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values)
    return "<table><thead><tr>"+cells("th",headers)+"</tr></thead><tbody>"+"".join(
        "<tr>"+cells("td",r)+"</tr>" for r in rows)+"</tbody></table>"


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def report(root, pose_directory="pose_evaluation"):
    pose_root=root/pose_directory
    seg = read(root/"segmentation_evaluation/summary.json")
    pose = read(pose_root/"results.json")
    protocol = read(pose_root/"protocol.json")
    config = read(root/"unet_v2/config.json")
    methods = list(NAMES)
    lookup = {(r["method"],r["condition"]):r for r in seg["results"]}
    conditions = list(dict.fromkeys(r["condition"] for r in seg["results"]))
    plots=root/"plots";plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    matrix=np.array([[lookup[m,c]["dice_mean"]*100 for c in conditions] for m in methods])
    fig,ax=plt.subplots(figsize=(12,3.8),layout="constrained")
    im=ax.imshow(matrix,vmin=0,vmax=100,cmap="viridis",aspect="auto")
    ax.set_yticks(range(3),list(NAMES.values()));ax.set_xticks(range(len(conditions)),conditions,rotation=30,ha="right",fontsize=8)
    for i in range(3):
        for j in range(len(conditions)):
            ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",color="black" if matrix[i,j]>65 else "white")
    fig.colorbar(im,ax=ax,label="Visible pseudo-mask Dice (%)",shrink=.8)
    fig.savefig(plots/"segmentation.png",dpi=160);plt.close(fig)
    seg_table=table(["조건"]+list(NAMES.values()),[[CONDITIONS[c]]+[f"{lookup[m,c]['dice_mean']*100:.2f}%" for m in methods] for c in conditions])
    shape_sections=""
    for dataset in pose["datasets"]:
        group=[r for r in pose["summary"] if r["dataset"]==dataset and r["condition"]!="clean"]
        shape_table=table(["조건","방법","비교 성립/전체","q18 RMSE 평균 [°]","FK 위치차 평균 [mm]","FK 위치차 p95 [mm]","FK 자세차 평균 [°]"],
            [[CONDITIONS[r["condition"]],NAMES[r["method"]],f"{r['paired_valid_frames']}/{r['requested_frames']}",
              fmt(r["q_rmse_deg_mean"]),fmt(r["fk_tip_delta_mm_mean"]),fmt(r["fk_tip_delta_mm_p95"]),
              fmt(r["fk_orientation_delta_deg_mean"])] for r in group])
        selected=[c for c in pose["conditions"] if c!="clean"]
        fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout="constrained")
        for method in methods:
            g={r["condition"]:r for r in group if r["method"]==method}
            for ax,field in zip(axes,["q_rmse_deg_mean","fk_tip_delta_mm_mean","paired_fraction_all_requested"]):
                values=[g[c][field] if g[c][field] is not None else np.nan for c in selected]
                if field=="paired_fraction_all_requested":values=np.asarray(values)*100
                ax.plot(range(len(selected)),values,"o-",color=COLORS[method],label=NAMES[method])
        for ax,label in zip(axes,["Mean paired q18 RMSE (deg)","Mean paired FK tip difference (mm)","Paired output availability (%)"]):
            ax.set_ylabel(label);ax.set_xticks(range(len(selected)),selected,rotation=35,ha="right",fontsize=8);ax.grid(alpha=.2)
        axes[2].set_ylim(-3,103);axes[0].legend(fontsize=8)
        fig.suptitle(dataset+": deviations from unoccluded output, NOT ground-truth errors")
        fig.savefig(plots/f"pose_{dataset}.png",dpi=160);plt.close(fig)
        shape_sections+=f'<h3>{DATASETS.get(dataset,dataset)}</h3><img src="plots/pose_{dataset}.png">{shape_table}'
    # A fixed middle source, not selected based on performance. Failures are annotated.
    fig=plt.figure(figsize=(12,5.5),layout="constrained")
    for index,dataset in enumerate(pose["datasets"]):
        members=[r for r in protocol["records"] if r["dataset"]==dataset]
        record=members[len(members)//2]
        ax=fig.add_subplot(1,2,index+1,projection="3d")
        for condition,color in [("clean","#555555"),("thick30","#d17a00"),("multi3_30","#007d91")]:
            path=pose_root/f"shape_arrays/{dataset}/{condition}/unet_v2/{record['frame_id']}.npz"
            if not path.exists():
                ax.text2D(.02,.9-.06*list(["clean","thick30","multi3_30"]).index(condition),f"{condition}: no output",transform=ax.transAxes)
                continue
            with np.load(path,allow_pickle=False) as d:
                if "curve_xyz_m" not in d:
                    ax.text2D(.02,.9-.06*["clean","thick30","multi3_30"].index(condition),f"{condition}: no output",transform=ax.transAxes)
                    continue
                curve=d["curve_xyz_m"]*1000
                observed=d["observed_base_xyz_m"]*1000
                ax.plot(*curve.T,color=color,label=condition)
                if condition=="multi3_30":ax.scatter(*observed.T,color=color,s=4,alpha=.5)
                ax.scatter(*(d["fk_tip_xyz_m"]*1000),color=color,marker="x",s=50)
        ax.set(xlabel="Base X (mm)",ylabel="Base Y (mm)",zlabel="Base Z (mm)",title=dataset)
        ax.legend(fontsize=8)
    fig.savefig(plots/"v2_shape_examples.png",dpi=160);plt.close(fig)
    joint_rows=list(csv.DictReader((pose_root/"per_joint_delta.csv").open()))
    fig,axes=plt.subplots(2,1,figsize=(12,6),layout="constrained")
    for ax,dataset in zip(axes,pose["datasets"]):
        selected=[c for c in pose["conditions"] if c!="clean"]
        heat=[]
        for c in selected:
            line=[]
            for joint in range(1,19):
                values=[abs(float(r["wrapped_delta_deg"])) for r in joint_rows if r["dataset"]==dataset and r["method"]=="unet_v2" and r["condition"]==c and int(r["joint_id"])==joint and r["wrapped_delta_deg"]]
                line.append(np.mean(values) if values else np.nan)
            heat.append(line)
        im=ax.imshow(heat,cmap="magma",aspect="auto",vmin=0)
        ax.set_xticks(range(18),range(1,19));ax.set_yticks(range(len(selected)),selected,fontsize=8)
        ax.set_title(dataset+": V2 mean absolute per-joint deviation (valid pairs only)");ax.set_xlabel("Joint: tilt, pan, tilt, ...")
        fig.colorbar(im,ax=ax,label="degrees")
    fig.savefig(plots/"joint_deltas.png",dpi=160);plt.close(fig)
    seg_links="".join(f'<li><a href="segmentation_evaluation/{c}_example_000.jpg">{CONDITIONS[c]} 예측 예시</a></li>' for c in conditions)
    example_links=[]
    for dataset in pose["datasets"]:
        for condition in ("clean","thick30","multi3_30"):
            images=sorted((pose_root/f"examples/{dataset}/{condition}").glob("*.png"))
            if images:
                path=images[len(images)//2].relative_to(root)
                example_links.append(f'<h3>{DATASETS[dataset]} / {CONDITIONS[condition]}</h3><img src="{path}">')
    refinement=""
    if pose_directory!="pose_evaluation":
        old=read(root/"pose_evaluation/results.json")
        comparison=[]
        for dataset in pose["datasets"]:
            before=next(r for r in old["summary"] if r["dataset"]==dataset and r["method"]=="unet_v2" and r["condition"]=="multi3_30")
            after=next(r for r in pose["summary"] if r["dataset"]==dataset and r["method"]=="unet_v2" and r["condition"]=="multi3_30")
            comparison.append([DATASETS[dataset],f"{before['paired_valid_frames']}/{before['requested_frames']}",f"{after['paired_valid_frames']}/{after['requested_frames']}"])
        refinement='<h2>짧은 visible 조각 처리 수정</h2><p>초기 frontend는 skeleton이8픽셀보다 짧은 조각을 버렸다. 실제 결과를 확인하니 base가 보이고 마스크도 올바른데, 3픽셀짜리 base skeleton이 삭제되어 원점 검사가 실패했다. '
        refinement+='따라서 최소1픽셀 경로도 보존하는 별도 frontend를 만들었다. singleton은 이미 관측된 한 픽셀을 유지할 뿐 숨은 점을 생성하지 않는다. 최소 마스크 면적16px·depth 검사·고정 원점 검사 및 모델 가중치는 동일하다. '
        refinement+='아래 본문은 수정 버전이며 초기 결과도 별도 보존했다. 이는 실패 원인 확인 후의 개발 수정으로, 새 블라인드 검증 결과가 아니다.</p>'
        refinement+=table(["구간 / 세 줄30% / U-Net V2","초기8px 출력","수정1px 출력"],comparison)
        refinement+='<p><a href="pose_evaluation/summary.csv">초기 frontend 결과</a></p>'
    body=f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>HRM V2 가림·형상 민감도</title>
<style>body{{max-width:1350px;margin:32px auto;padding:0 25px;font:16px/1.7 sans-serif;color:#18303c}}img{{max-width:100%;border:1px solid #ddd;margin:12px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #ddd;padding:6px 9px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf3f5}}.note{{background:#fff2d4;border-left:5px solid #c58a00;padding:16px}}h2{{margin-top:40px}}a{{color:#00677c}}</style>
<h1>HRM V2: 다양한 가림·원근 증강과 관절각/FK 민감도</h1>
<p>2026-10-24 · RTX 3060 · 카메라 수집/오프라인 연구 · 모터 제어 및 production ROS 코드는 변경하지 않음</p>
<div class="note"><b>두 평가를 분리한다.</b> Segmentation은 OpenCV pseudo-label과의 Dice/IoU 일치도.
관절각·FK는 동일 프레임/방법의 가림 없는 결과 대비 변화량이며 실제 자세 정답 오차가 아니다.
3D 결과에는 가림 위치를 알고 그곳의 depth를 제외한 <b>missing-depth simulation</b> 가정이 있다.
실물 종이/물체의 유효하지만 잘못된 depth를 자동으로 제거한 성능은 아직 평가하지 않았다.</div>
<h2>추가한 증강과 학습</h2><ul>
<li>한 줄 약2·10·20·30%, 세 줄 합계15·30% 가림. 검정/회색/흰색/유색 불투명 줄. 가림 앞뒤 양 끝 중심선 구간 보호.</li>
<li>RGB와 마스크에 동일한 homography를 적용해 원근 taper, 크기, 위치, 회전을 바꿈. 실제 3D 시점·광각 렌즈 왜곡을 합성한 것은 아님.</li>
<li>별도 split 배경 합성, 조명/연기/blur/noise, 배경의 밝은 물체와 negative 장면 추가.</li>
<li>ResNet18 U-Net V2: 학습 {config['training_images']}장, {config['epochs']} epoch, 검증 {config['validation_images']}장×3조건. 검증으로만 checkpoint 선택.</li>
<li>V1은 이전 모델 그대로. 학습량/증강이 함께 달라졌으므로 증강 하나의 인과 효과만 분리한 ablation이 아님.</li>
<li>평가 archive는 이전 파일럿에서 이미 검토한 개발용 stress set. 새 블라인드 테스트나 임상 일반화 결과로 부르지 않음.</li>
<li>MobileSAM은 이번 본 비교/후처리에 넣지 않음. 이전 파일럿의 선택·정제 도구로 남김.</li>
</ul>
<h2>Segmentation: {seg['records']}프레임 × 10조건 × 3방법</h2>
<p>Dice = 2|예측∩기준|/(|예측|+|기준|). 보이는 몸체의 겹침 정도이며 픽셀 정확도 또는 각도 정확도가 아니다.
OpenCV 원본100%는 같은 OpenCV pseudo-label을 재현했기 때문이다. 실제 가림 정답을 수작업 검수해야 최종 정확도를 말할 수 있다.</p>
{seg_table}<img src="plots/segmentation.png"><ul>{seg_links}</ul>
<h2>관측점만으로 곡선·관절각·FK 계산</h2><ol>
<li>예측 mask에서 여러 visible component의 Lee skeleton과 경로를 유지. 원본/정답 mask로 예측을 자르지 않는다.</li>
<li>알려진 합성 가림 부분의 depth는 사용하지 않음. 주변 median도 예측 몸체 내부의 관측 가능한 depth만 사용.</li>
<li>아래쪽 base endpoint부터 visible 경로를 순서대로 배치, crop K로 역투영. 가려진 구간에 측정점을 생성하지 않음.</li>
<li>관측점 사이의 누적3D chord 길이로 매개화. gap에서는 실제 arc가 아닌 chord 근사이며, 남은 점에 원점제약4차 곡선을 피팅.</li>
<li>production의 물리길이 정규화→20경계점/19중심점→경계점 사이19방향벡터→순차 DH 평면 투영→18관절각(tilt부터) 계산을 그대로 재사용.</li>
<li>robot_control_pkg와 같은 modified-DH chain으로 FK tip 위치/회전 계산. 원본C++함수와 별도컴파일 수치 일치 검증.</li>
</ol><p>시간 Kalman/base평균은 비활성화. 각 방법의 clean base 위치만 같은 프레임의 가림 조건에 고정하며 clean shape 전체를 피팅에 전달하지 않는다.
원점에서 첫 관측점이 하드웨어길이10%보다 떨어지면 실패 처리. 2D extrapolation은 사용하지 않는 새 offline front-end이므로 unchanged production end-to-end의 성능은 아니다.</p>
{refinement}
<h2>관절각과 tip 변화 결과</h2><p>q18 RMSE는 프레임마다18개 실제 관절축의 wrapped 각도 차이를 계산한 뒤 프레임별RMSE를 평균한 값. FK 위치는 유클리드 거리, FK 자세는 두 회전행렬 간 SO(3) 회전각이다.
아래 평균/p95는 <b>비교가 성립한 프레임만</b>의 값이며 실패는 분모와 CSV에 남긴다. 방법별 own-clean 기준이 달라 절대정확도 순위로 해석하면 안 된다.</p>
{shape_sections}<img src="plots/joint_deltas.png"><img src="plots/v2_shape_examples.png">
<p>3D 예시는 각 촬영 구간의 미리 정한 중간 프레임. 곡선은 물리길이 정규화된 추정 결과, 점은 정규화 전 남은 실제 관측점, ×는 FK tip.
가림 영역을 통과하는 곡선은 보간 모델이지 관측된 숨은 HRM이 아니다. 출력 실패도 숨기지 않는다.</p>
<h2>검정 배경 및 굽힘 자세 예시</h2>{''.join(example_links)}
<h2>실제 실험과 논문 해석</h2><ul>
<li>검정 배경에서80 RGB-D쌍을 새로 수집, 그중40프레임과 저장된 굽힘 구간64프레임을 분리 평가했다. 실제 물체 가림은 아직 없음.</li>
<li>실물 가림 시험에서는 가림 정도별 visible mask를 검수하고, tag 보정 후 tip GT/실패율을 동시에 비교한다. 작은 clean 대비 변화만으로 정확한 형상을 복원했다고 주장하지 않는다.</li>
<li>base/tip 가림, 자기겹침, 큰 배경 오검출은 이번 순서정렬 가정 밖이다. 모든 visible 조각을 남기되 무관한 조각은 배제하는 검증이 추가로 필요하다.</li>
<li>추후 제공할 합성사진/광각 tip영상은 별도 외부 평가 set으로 추가할 수 있다. homography RGB에 원래 depth를 재사용하지 않는다.</li>
</ul>
<h2>저장 파일</h2><ul>
<li><a href="segmentation_evaluation/summary.csv">Segmentation 조건별 CSV</a></li>
<li><a href="pose_evaluation/summary.csv">각도/FK 요약 CSV</a> · <a href="pose_evaluation/per_frame.csv">모든 프레임/실패</a> · <a href="pose_evaluation/per_joint_delta.csv">관절별 clean/가림 각도와 차이</a></li>
<li><a href="pose_evaluation/baseline_pose.csv">기준 q18 및 FK 위치·회전행렬</a> · <a href="pose_evaluation/protocol.json">파이프라인/좌표계/필터 설정</a></li>
<li><a href="unet_v2/best.pt">V2 checkpoint</a> · <a href="unet_v2/history.csv">학습 로그</a></li>
</ul><p>원근 변환 참고: <a href="https://github.com/opencv/opencv/blob/4.x/doc/tutorials/features2d/homography/homography.markdown">OpenCV homography 공식 설명</a>.</p></html>"""
    body=body.replace('href="pose_evaluation/',f'href="{pose_directory}/')
    # Preserve the explicit initial-front-end comparison link.
    body=body.replace(f'href="{pose_directory}/summary.csv">초기 frontend', 'href="pose_evaluation/summary.csv">초기 frontend')
    (root/"report.html").write_text(body,encoding="utf-8")
    print(root/"report.html")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=Path("runs/occlusion_pose_20261024"))
    parser.add_argument("--pose-directory",default="pose_evaluation")
    args=parser.parse_args()
    report(args.run,args.pose_directory)
