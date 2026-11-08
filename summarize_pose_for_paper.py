#!/usr/bin/env python3
"""Describe paired occlusion sensitivity without claiming absolute pose accuracy.

Reads frozen evaluation artifacts only. No model fitting, camera, ROS, or motor
operations. Recomputes all U-Net-v2 pose metrics from NPZ arrays and independently
checks every saved observed pixel against regenerated synthetic occlusion maps.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
import zlib

import cv2
import numpy as np

from feasibility_augmentation_v2 import make_case_v2
from feasibility_backends import OpenCVBaseline

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / 'runs/occlusion_pose_20261024/pose_evaluation_short_fragments'
DEFAULT_OUTPUT = ROOT / 'runs/perspective_pose_followup_20261024'
DATASETS = {
    'live_black': '새 검정 배경 · 정지 직선 자세 (40프레임)',
    'archive_45deg_1': '기존 동적·굽힘 자세 기록 (64프레임)',
}
CONDITIONS = {
    'clean': '무가림 자기 비교', 'thin': '단일 얇은 가림 (~2%)',
    'thick10': '단일 가림 10%', 'thick20': '단일 가림 20%',
    'thick30': '단일 가림 30%', 'multi3_15': '3개 가림 · 합계 15%',
    'multi3_30': '3개 가림 · 합계 30%',
}
METRICS = ('fk_tip_delta_mm', 'q18_rmse_deg', 'fk_orientation_delta_deg', 'curve_tip_delta_mm')
CSV_NAMES = {'q18_rmse_deg': 'q_rmse_deg'}
LIMITATIONS = [
    '실제 정답(ground truth) 대비 정확도 실험이 아니다. 각 방법의 동일 원본 프레임·무가림 추정값과 가림 추정값을 비교한다.',
    '무가림 depth 곡선과 그 각도로 계산한 FK 팁도 측정·모델 오차가 있는 추정값이며 정답이 아니다. 둘이 일치하거나 가림 전후 변화가 작아도 실제 1mm 정확도가 증명되지는 않는다.',
    '합성 가림의 정확한 픽셀 지도를 알고 depth 질의점과 median 이웃을 제외했다. 실제 가림체가 내는 유효 depth를 자동으로 찾아 제거하는 성능은 검증하지 않았다.',
    '예측 mask 자체는 정답 가림 지도로 잘라주지 않는다. 가림 내부의 숨은 원본 HRM depth도 읽지 않는다. 가림 사이 곡선은 남은 관측점과 모델 가정으로 보간한다.',
    'base와 tip이 보이는 조건을 의도했고 각 방법·프레임의 clean base 위치를 가림 변형에도 제공했다. 실제 동작에서는 안정적인 base 검출 또는 보정이 별도로 필요하다.',
    '새 촬영은 정지 직선 자세이고 다양한 굽힘은 기존 기록에서 가져왔다. 두 데이터셋은 분리 보고하며 외부 독립 검증이나 임상 검증을 의미하지 않는다.',
    '원근·homography 등 기하 변환된 RGB 이미지는 이 pose 시험에 포함하지 않았다. 그 이미지에 원래 depth를 붙여 3D 정확도를 계산하지 않았다.',
    'minimum_path_points=1은 최초 min8 결과에서 짧은 관측 조각 제거 문제를 발견한 뒤 도입한 탐색적 frontend 수정이다. 최초 평가를 보존했고 untouched test 성능으로 주장하지 않는다.',
    '시간 필터·Kalman·base 평균 없이 프레임별 독립 계산이다. 관측 표면과 실제 중심축의 반경 차이 및 카메라/마커 보정 오차는 여기서 제거하지 않았다.',
    '실패는 전체 요청 분모에 유지한다. 변화량의 평균·중앙값·p95·최댓값은 유효한 paired 프레임에 조건부로 계산한다. 출력 성공은 올바른 복원이나 안전한 제어의 보장이 아니다.',
]
EQUATIONS = {
    'joint_difference': r'\Delta q_{t,j}=\operatorname{atan2}(\sin(q^{\mathrm{occ}}_{t,j}-q^{\mathrm{clean}}_{t,j}),\cos(q^{\mathrm{occ}}_{t,j}-q^{\mathrm{clean}}_{t,j})),\quad j=1,\ldots,18',
    'joint_rmse': r'e_{q,t}=\frac{180}{\pi}\sqrt{\frac{1}{18}\sum_{j=1}^{18}(\Delta q_{t,j})^2}',
    'position_difference': r'e_{p,t}=1000\left\|\mathbf p^{\mathrm{occ}}_{t,\mathrm{FK}}-\mathbf p^{\mathrm{clean}}_{t,\mathrm{FK}}\right\|_2\quad[\mathrm{mm}]',
    'rotation_difference': r'e_{R,t}=\frac{180}{\pi}\cos^{-1}\!\left[\operatorname{clip}\!\left(\frac{\operatorname{tr}((R^{\mathrm{clean}}_{t,\mathrm{FK}})^T R^{\mathrm{occ}}_{t,\mathrm{FK}})-1}{2},-1,1\right)\right]',
    'one_mm_fraction': r'P_{\le1\mathrm{mm}}=\frac{\sum_{t\in\mathcal V}\mathbf1[e_{p,t}\le1\mathrm{mm}]}{|\mathcal V|},\qquad A=\frac{|\mathcal V|}{N_{\mathrm{requested}}}',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(value):
    if value in (True, 'True'):
        return True
    if value in (False, 'False'):
        return False
    raise ValueError(f'Unexpected Boolean value: {value!r}')


def recompute_metrics(arrays):
    q, ref = np.asarray(arrays['q_rad']), np.asarray(arrays['clean_q_rad'])
    if q.shape != (18,) or ref.shape != (18,) or not np.isfinite([q, ref]).all():
        raise ValueError('Require 18 finite observed and clean angles.')
    delta = np.arctan2(np.sin(q-ref), np.cos(q-ref))
    rotation, clean_rotation = arrays['fk_tip_rotation'], arrays['clean_fk_tip_rotation']
    for matrix in (rotation, clean_rotation):
        if matrix.shape != (3,3) or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(matrix), 1.):
            raise ValueError('Invalid proper rotation matrix.')
    angle = 0. if np.allclose(rotation, clean_rotation, rtol=0, atol=1e-12) else np.rad2deg(np.arccos(np.clip((np.trace(clean_rotation.T @ rotation)-1)/2, -1, 1)))
    result = dict(fk_tip_delta_mm=float(np.linalg.norm(arrays['fk_tip_xyz_m']-arrays['clean_fk_tip_xyz_m'])*1000),
                  q18_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(delta**2)))),
                  fk_orientation_delta_deg=float(angle),
                  curve_tip_delta_mm=float(np.linalg.norm(arrays['curve_tip_xyz_m']-arrays['clean_curve_tip_xyz_m'])*1000))
    if not np.isfinite(list(result.values())).all():
        raise ValueError('Nonfinite pose metric.')
    return result


def summarize_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['dataset'], row['condition']].append(row)
    result = []
    for (dataset, condition), group in groups.items():
        paired = [row for row in group if row['paired_valid']]
        count = sum(row['fk_tip_delta_mm'] <= 1. for row in paired)
        entry = dict(dataset=dataset, condition=condition, method='unet_v2',
                     requested_frames=len(group), valid_paired_frames=len(paired),
                     invalid_frames=len(group)-len(paired), availability=len(paired)/len(group),
                     fk_tip_delta_le_1mm_count=count,
                     fk_tip_delta_le_1mm_fraction_paired=count/len(paired) if paired else None,
                     fk_tip_delta_le_1mm_fraction_all_requested=count/len(group))
        for field in METRICS:
            values = np.array([row[field] for row in paired])
            for name, fn in (('mean',np.mean),('median',np.median),('p95',lambda x:np.percentile(x,95)),('max',np.max)):
                entry[f'{field}_{name}'] = float(fn(values)) if len(values) else None
        # Different from mean of per-frame RMSE, and explicitly named.
        entry['q18_pooled_rmse_deg'] = float(np.sqrt(np.mean([row['q18_rmse_deg']**2 for row in paired]))) if paired else None
        ratios = [row['actual_centerline_occlusion_fraction'] for row in group]
        entry['placement_reference_occlusion_fraction_mean'] = float(np.mean(ratios))
        entry['placement_reference_occlusion_fraction_min'] = float(np.min(ratios))
        entry['placement_reference_occlusion_fraction_max'] = float(np.max(ratios))
        result.append(entry)
    return result


def csv_write(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def relative_link(path, output):
    return quote(os.path.relpath(Path(path), Path(output)), safe='/')


def render_html(report, output):
    esc = html.escape
    fmt = lambda x: '—' if x is None else f'{x:.3f}'
    sections = []
    for dataset, title in DATASETS.items():
        rows = [row for row in report['summary'] if row['dataset']==dataset]
        tr = []
        for row in rows:
            cells = [CONDITIONS[row['condition']], f"{row['valid_paired_frames']}/{row['requested_frames']}"]
            cells += [fmt(row[f'fk_tip_delta_mm_{stat}']) for stat in ('mean','median','p95','max')]
            cells += ['—' if row['fk_tip_delta_le_1mm_fraction_paired'] is None else f"{100*row['fk_tip_delta_le_1mm_fraction_paired']:.1f}%"]
            cells += [fmt(row['q18_rmse_deg_mean']),fmt(row['q18_rmse_deg_p95']),fmt(row['fk_orientation_delta_deg_mean']),fmt(row['fk_orientation_delta_deg_p95'])]
            tr.append('<tr>'+''.join(f'<td>{esc(cell)}</td>' for cell in cells)+'</tr>')
        heads = ['가림 조건','유효/전체','팁 평균','팁 중앙값','팁 p95','팁 최대','변화 ≤1mm','관절 평균','관절 p95','자세 평균','자세 p95']
        sections.append(f'<h2>{esc(title)}</h2><p>팁 위치 변화: mm · 18관절 RMSE 및 회전 지오데식 변화: °. 각 평균은 프레임별 지표의 평균입니다.</p><div class="scroll"><table><thead><tr>'+''.join(f'<th>{esc(h)}</th>' for h in heads)+'</tr></thead><tbody>'+''.join(tr)+'</tbody></table></div>')
    source_rows=[]
    for record in report['source_index']:
        links=' · '.join(f'<a href="{relative_link(record[key],output)}">{label}</a>' for key,label in [('rgb','RGB'),('depth','Depth'),('source_manifest','촬영 메타데이터')])
        source_rows.append(f'<tr><td>{esc(record["dataset"])}</td><td>{esc(record["frame_id"])}</td><td>{record["source_stamp_ns"]}</td><td>{links}</td></tr>')
    latex='\n\n'.join(report['equations_latex'].values())
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HRM 가림 후 관절·팁 변화량</title><style>
body{font:16px/1.65 system-ui,sans-serif;color:#173147;background:#f3f6f9;margin:0}main{max-width:1400px;margin:30px auto;padding:28px;background:white;border-radius:14px}h1{line-height:1.3}h2{margin-top:40px}a{color:#1754a0}.notice{padding:18px 22px;background:#fff3d6;border-left:5px solid #e2a52d;border-radius:7px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dbe3eb;padding:10px;text-align:right;white-space:nowrap}th{background:#edf3f8}td:first-child,th:first-child{text-align:left}tr:hover{background:#f6f9fb}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eff3f6;padding:16px;border-radius:8px}small{color:#536b7a}li{margin:8px 0}.links{margin:20px 0}
</style><main><h1>가림 후 18관절 각도 및 FK 팁 변화량</h1><p>U-Net v2 · 동일 프레임 paired 비교 · 기존 결과 재계산</p><div class="notice"><strong>“실제 위치 오차 1mm 이하”라는 결과가 아닙니다.</strong><br>표의 ≤1mm는 <b>가림 전의 추정 팁</b>과 <b>가림 후의 추정 팁</b> 사이 거리입니다. 실제 HRM 팁 정답과의 거리도, 마커 검증 오차도 아닙니다. 무가림 곡선/FK 자체도 정답이 아닌 추정값입니다.</div><div class="links"><a href="paper_pose_summary.csv">요약 CSV</a> · <a href="paper_pose_summary.json">전체 수치·정의 JSON</a> · <a href="paper_pose_frames.csv">프레임별 재계산 CSV</a> · <a href="paper_pose_source_index.json">원본/깊이/보정 index</a></div>'''+''.join(sections)+f'''<h2>계산 정의</h2><p>q₁~q₁₈은 tilt–pan 교대 관절의 상대각입니다. 각도 차이를 −π~π로 감싼 뒤 18개 관절에 대한 RMSE를 계산합니다. 위치는 동일 hrm_base 기준 FK 팁의 유클리드 거리, 자세는 전체 3×3 회전행렬의 SO(3) 지오데식 각도입니다. 오일러각 성분의 차이를 더한 값이 아닙니다.</p><p>Δqⱼ = atan2(sin(qⱼ,occ − qⱼ,clean), cos(qⱼ,occ − qⱼ,clean))<br>e_q = √(Σⱼ Δqⱼ² / 18) · 180/π<br>e_p = 1000 ‖p_FK,occ − p_FK,clean‖₂ [mm]<br>e_R = acos(clip((tr(R_cleanᵀ R_occ) − 1)/2, −1, 1)) · 180/π</p><details><summary>논문에 붙여넣을 LaTeX 수식</summary><pre>{esc(latex)}</pre></details><h2>해석과 한계</h2><ul>'''+''.join('<li>'+esc(text)+'</li>' for text in LIMITATIONS)+f'''</ul><h2>검증</h2><p>{report['verification']['npz_recomputed_frames']}개 유효 NPZ에서 지표를 다시 계산해 저장 CSV와 일치함을 확인했습니다. U-Net v2 관측점 {report['verification']['observed_points']:,}개 중 알려진 합성 가림 내부 점은 {report['verification']['observed_points_in_known_occlusion']}개였습니다. 이는 실제 가림체를 자동 검출했다는 뜻은 아닙니다.</p><p>원본 평가 및 independent_audit.json은 변경하지 않았으며 입력 파일 hash는 JSON에 기록했습니다.</p><h2>원본 RGB-D 및 메타데이터</h2><p>RGB와 depth는 촬영 timestamp로 연결했습니다. 저장 파일 번호가 다른 경우에도 timestamp가 동일한 짝을 사용합니다. crop-local K와 depth 단위, 평가 NPZ/예시 링크는 source index JSON에 있습니다. 아래 링크는 로컬 HTML을 열 때 사용할 수 있습니다.</p><div class="scroll"><table><thead><tr><th>데이터셋</th><th>프레임</th><th>촬영 timestamp(ns)</th><th>원본 링크</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></div><p><small>자료 목적: 합성 가림 민감도와 pipeline 실행 가능성. 실제 3D 위치·관절각 정확도 보장 및 실시간 폐루프 제어 안전성 검증은 별도 실험이 필요합니다.</small></p></main></html>'''


def run(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    products = ['paper_pose_summary.csv','paper_pose_summary.json','paper_pose_summary.html','paper_pose_frames.csv','paper_pose_source_index.json']
    if any((output/name).exists() for name in products):
        raise FileExistsError('Refusing to overwrite an existing paper summary.')
    names = ['protocol.json','per_frame.csv','cases.json','summary.csv','independent_audit.json']
    hashes = {name:digest(source/name) for name in names}
    protocol = json.loads((source/'protocol.json').read_text())
    audit = json.loads((source/'independent_audit.json').read_text())
    if audit['audit_status'] != 'PASS':
        raise ValueError('Source independent audit did not pass.')
    for name, expected in audit['original_artifact_sha256'].items():
        if digest(source/name) != expected:
            raise ValueError(f'Source artifact changed since audit: {name}')
    cases = {(x['dataset'],x['frame_id'],x['condition']):x for x in json.loads((source/'cases.json').read_text()) if x['method']=='unet_v2'}
    records = {(x['dataset'],x['frame_id']):x for x in protocol['records']}
    with (source/'per_frame.csv').open() as stream:
        original_rows = [row for row in csv.DictReader(stream) if row['method']=='unet_v2']
    verification = dict(npz_recomputed_frames=0, observed_points=0, observed_points_in_known_occlusion=0,
                        source_artifact_hashes_verified=True, source_artifacts_preserved=True)
    rows=[]; linked=defaultdict(list)
    for original in original_rows:
        dataset,frame,condition = original['dataset'],original['frame_id'],original['condition']
        record=records[dataset,frame]; case=cases[dataset,frame,condition]
        row=dict(dataset=dataset,frame_id=frame,source_stamp_ns=int(original['source_stamp_ns']),condition=condition,
                 paired_valid=truth(original['paired_valid']),error=original['error'],
                 actual_centerline_occlusion_fraction=float(case['augmentation']['actual_centerline_occlusion_fraction']))
        row.update({metric:None for metric in METRICS})
        if row['paired_valid']:
            with np.load(source/original['shape_npz'],allow_pickle=False) as loaded:
                arrays={key:loaded[key] for key in loaded.files}
            metrics=recompute_metrics(arrays)
            for field,value in metrics.items():
                if not np.isclose(value,float(original[CSV_NAMES.get(field,field)]),rtol=1e-9,atol=1e-9):
                    raise ValueError(f'NPZ/CSV metric mismatch: {frame}/{condition}/{field}')
            row.update(metrics); verification['npz_recomputed_frames']+=1
            rgb=cv2.cvtColor(cv2.imread(record['rgb']),cv2.COLOR_BGR2RGB)
            seed=protocol['seed']+zlib.crc32(f"{dataset}:{record['source_stamp_ns']}".encode())
            _,_,meta,occlusion=make_case_v2(rgb,OpenCVBaseline().predict_rgb(rgb),seed,condition)
            if int(occlusion.sum()) != case['augmentation']['occluder_pixels']:
                raise ValueError('Occlusion regeneration disagrees with saved case.')
            pixels=arrays['observed_pixels_yx']; hidden=int(occlusion[pixels[:,0],pixels[:,1]].sum())
            verification['observed_points']+=len(pixels); verification['observed_points_in_known_occlusion']+=hidden
            if hidden:
                raise ValueError('Observed point lies inside known occluder.')
        rows.append(row)
        linked[dataset,frame].append(dict(condition=condition,npz=str(source/original['shape_npz']) if original['shape_npz'] else None,
            comparison_image=str(source/'examples'/dataset/condition/(frame+'.png')) if (source/'examples'/dataset/condition/(frame+'.png')).exists() else None,
            predicted_mask=str(source/'example_masks'/dataset/condition/'unet_v2'/(frame+'.png')) if (source/'example_masks'/dataset/condition/'unet_v2'/(frame+'.png')).exists() else None))
    index=[dict(record,evaluation_outputs=linked[key]) for key,record in records.items()]
    report=dict(title='U-Net v2 paired occlusion sensitivity; NOT absolute pose accuracy',
                source_directory=str(source),method='unet_v2',source_sha256=hashes,
                reference='Same method, same source RGB-D frame, unoccluded estimate; not independent ground truth.',
                unit_notes={'position':'mm','q18_rmse':'degrees; mean is average of per-frame 18-joint RMSE, pooled RMSE separately named',
                            'rotation':'degrees; full SO(3) geodesic','fractions':'0 to 1; <=1mm is clean-reference displacement, NOT absolute accuracy'},
                equations_latex=EQUATIONS,limitations=LIMITATIONS,verification=verification,
                summary=summarize_rows(rows),source_index=index)
    if hashes != {name:digest(source/name) for name in names}:
        raise ValueError('Original source changed while summarizing.')
    output.mkdir(parents=True,exist_ok=True)
    csv_write(output/products[0],report['summary']);json_write(output/products[1],report)
    (output/products[2]).write_text(render_html(report,output))
    csv_write(output/products[3],rows);json_write(output/products[4],index)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();result=run(args.source,args.output)
    print(json.dumps(dict(output=str(args.output),verification=result['verification']),indent=2))
