from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from defusedxml import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_algorithm_evidence import (  # noqa: E402
    csv_write,
    evaluate,
)  # noqa: E402

OUTPUT = ROOT / 'outputs/algorithm_evidence_20261022'
GEOMETRY_TESTS = [
    'test_normalise_bbox_extra_fields',
    'test_detect_polygon_from_cones_no_cones',
    'test_calculate_people_in_controlled_area_no_polygons',
    'test_calculate_people_in_controlled_area',
    'test_no_cones',
    'test_person_inside_polygon',
    'test_detect_polygon_from_cones_empty_list',
    'test_detect_polygon_from_cones_with_noise',
    'test_calculate_people_in_controlled_area_no_datas',
    'test_count_people_in_polygon',
    'test_polygons_to_coords',
]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def collect():
    records, inputs = {}, []
    for batch in [
        'patent_figures_20261021',
        'aerial_cone_zones_20261021',
        'aerial_people_20261021',
    ]:
        path = ROOT / 'outputs' / batch / 'data/results.json'
        m = json.loads(path.read_text())
        inputs.append(
            dict(
                path=str(path.relative_to(ROOT)),
                sha256=sha(path),
                model=m['model'],
            ),
        )
        shutil.copy2(path, OUTPUT / 'inputs' / f"{batch}.json")
        for sid, r in m['sources'].items():
            h = r['source_sha256']
            if h in records:
                assert records[h]['detections'] == r['detections']
            else:
                records[h] = {**r, 'formal_image': True, 'aliases': []}
            records[h]['aliases'].append(dict(batch=batch, source_id=sid))
    path = ROOT / (
        'outputs/supplement_compilation_20261021_v2/research/camera_refine'
        'd.json'
    )
    inputs.append(
        dict(
            path=str(path.relative_to(ROOT)),
            sha256=sha(
                path,
            ),
            model='YOLO26m, imgsz=1280, conf=0.25',
        ),
    )
    shutil.copy2(path, OUTPUT / 'inputs/camera_refined.json')
    for r in json.loads(path.read_text()):
        h = r['source_sha256']
        if h in records:
            assert records[h]['detections'] == r['detections']
        else:
            records[h] = {**r, 'formal_image': False, 'aliases': []}
        records[h]['aliases'].append(
            dict(batch='same_camera_57', source_id=r['source_id']),
        )
    for i, r in enumerate(records.values()):
        r['scene_id'] = f"S{i + 1:03}"
    dump(
        OUTPUT / 'inputs/replay_dataset.json',
        dict(
            records=list(records.values()),
            input_files=inputs,
            conditions=dict(
                model='YOLO26m',
                imgsz=1280,
                conf=0.25,
                iou=0.7,
                device='cpu',
                deduplication=(
                    'Exact original-image SHA256; one image appears only '
                    'once.'
                ),
                formal_subset_images=11,
                expanded_images=64,
                repeated_camera_frames_are_not_independent_people=True,
                independent_ground_truth=(
                    'Not available for region intrusion or occupant '
                    'identity.'
                ),
                no_new_image_inference=(
                    'Replay starts from archived detection boxes; it does not '
                    'measure missed YOLO '
                    'detections.'
                ),
            ),
        ),
    )


def snapshot():
    for name in [
        '__init__.py',
        'geometry.py',
        'danger_detector.py',
        'auth_tokens.py',
        'redis_client.py',
        'runtime_utils.py',
    ]:
        shutil.copy2(ROOT / 'src' / name, OUTPUT / 'source/src' / name)
    for name in ['danger_detector_test.py', 'utils_test.py']:
        shutil.copy2(
            ROOT / 'tests/src' / name,
            OUTPUT / 'source/tests/src' / name,
        )
    for folder in ['source/tests', 'source/tests/src']:
        (OUTPUT / folder / '__init__.py').write_text('')
    shutil.copy2(
        ROOT / 'scripts/evaluate_algorithm_evidence.py',
        OUTPUT / 'reproduce/evaluate_algorithm_evidence.py',
    )
    shutil.copy2(
        Path(__file__),
        OUTPUT / 'reproduce/build_algorithm_evidence.py',
    )
    selected = {
        'normalise_bbox',
        'detect_polygon_from_cones',
        'calculate_people_in_controlled_area',
        'polygons_to_coords',
        '__init__',
        'detect_danger',
        'check_cone_restricted_area',
        'check_proximity_violations',
        '_person_proximity_flags',
        '_is_driver_detection',
        '_is_dangerously_close_detection',
        '_build_spatial_index',
        '_nearby_machinery_vehicles',
        '_proximity_query_bbox',
        '_filter_and_normalise_static_machinery',
    }
    index, snippets = [], []
    for filename in ['geometry.py', 'danger_detector.py']:
        path = ROOT / 'src' / filename
        source = path.read_text()
        lines = source.splitlines()
        for node in ast.walk(ast.parse(source)):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in selected
            ):
                code = '\n'.join(lines[node.lineno - 1: node.end_lineno])
                qualified = (
                    'DangerDetector.'
                    if filename == 'danger_detector.py'
                    else ''
                ) + node.name
                index.append(
                    dict(
                        function=qualified,
                        package_file=f"source/src/{filename}",
                        original_repo_path=f"src/{filename}",
                        start_line=node.lineno,
                        end_line=node.end_lineno,
                        source_file_sha256=sha(path),
                        excerpt_sha256=hashlib.sha256(
                            code.encode(),
                        ).hexdigest(),
                    ),
                )
                snippets.append(
                    (
                        '===== src/'
                        f"{filename}"
                        ':'
                        f"{node.lineno}"
                        '  '
                        f"{qualified}"
                        ' =====\n'
                        f"{code}"
                        '\n'
                    ),
                )
    csv_write(OUTPUT / 'docs/function_index.csv', index)
    (OUTPUT / 'docs/related_functions.txt').write_text('\n'.join(snippets))


CONTRACT_TESTS = (
    '"""Declared synthetic contracts and explicit observations of known '
    'limitations."""\nfrom pathlib import Path\nimport sys\nimport '
    'unittest\n\nBASE = Path(__file__).resolve().parents[2]\nsys.path.insert(0'
    ", str(BASE / 'source'))\nsys.path.insert(0, str(BASE / "
    "'reproduce'))\nfrom evaluate_algorithm_evidence import box, flags, "
    'NoOccupantExclusion, synthetic_experiments\nfrom src.danger_detector '
    'import DangerDetector\nfrom src.geometry import '
    'calculate_people_in_controlled_area, detect_polygon_from_cones, '
    'polygons_to_coords\n\n\nclass AlgorithmContractTests(unittest.TestCase):'
    '\n    @classmethod\n    def setUpClass(cls):\n        cls.cones, '
    'cls.proximity, cls.observations = synthetic_experiments()\n\n    def '
    'test_all_25_declared_center_rule_cases(self):\n        '
    'self.assertEqual(len(self.cones), 25)\n        for row in self.cones:\n  '
    "          with self.subTest(case=row['case']):\n                "
    "self.assertEqual(row['current_alert'], "
    "row['declared_expected_alert'])\n\n    def "
    'test_grouping_preserves_gap(self):\n        for row in self.cones:\n     '
    "       if row['case'].endswith('between_groups'):\n                "
    "self.assertFalse(row['current_alert'])\n                "
    "self.assertTrue(row['single_clustered_points_hull_alert'])\n\n    def "
    'test_strict_boundary_and_center_instead_of_feet(self):\n        for row '
    "in self.cones:\n            if 'exact_boundary' in row['case'] or "
    "'feet_inside_center_outside' in row['case']:\n                "
    "self.assertFalse(row['current_alert'])\n            if "
    "'center_inside_feet_outside' in row['case']:\n                "
    "self.assertTrue(row['current_alert'])\n\n    def "
    'test_outlier_does_not_bridge_regions(self):\n        row = next(r for r '
    "in self.cones if r['case'] == 'noise_outlier_bridge')\n        "
    "self.assertTrue(row['single_all_cones_hull_alert'])\n        "
    "self.assertFalse(row['single_clustered_points_hull_alert'])\n        "
    "self.assertFalse(row['current_alert'])\n\n    def "
    'test_duplicate_centers_are_counted_once(self):\n        cones = '
    '[box(x,y) for ox in [0,1000] for x,y in '
    '[(ox,0),(ox,100),(ox+100,0),(ox+100,100)]]\n        regions = '
    'detect_polygon_from_cones(cones, DangerDetector().clusterer)\n        '
    'people = [box(50,50,5,20,20), box(50,50,5,30,30)]\n        '
    'self.assertEqual(calculate_people_in_controlled_area(regions, '
    'cones+people), 1)\n\n    def test_empty_and_insufficient_cones(self):\n  '
    '      for data in [[], [box(0,0)], [box(0,0),box(100,100)]]:\n           '
    ' self.assertEqual(detect_polygon_from_cones(data, '
    'DangerDetector().clusterer), [])\n\n    def '
    'test_single_cluster_all_noise_is_observed_not_fixed(self):\n        row '
    "= next(r for r in self.observations if r['probe']=='single_square')\n    "
    "    self.assertEqual(row['labels'], [-1,-1,-1,-1])\n        "
    "self.assertEqual(row['returned_types'], [])\n\n    def "
    'test_collinear_count_export_mismatch_is_observed_not_fixed(self):\n      '
    '  row = next(r for r in self.observations if '
    "r['probe']=='two_collinear_groups')\n        "
    "self.assertEqual(row['returned_types'], ['LineString','LineString'])\n   "
    "     self.assertEqual(row['counted_people_with_center_15_0'], 1)\n       "
    " self.assertEqual(row['exported_coordinate_regions'], 0)\n\n    def "
    'test_real_six_condition_method_can_cancel_synthetic_occupant_alert(self):'
    '\n        row = next(r for r in self.proximity if '
    "r['case']=='occupant_inside')\n        "
    "self.assertTrue(row['occupant_excluded'])\n        "
    "self.assertFalse(row['current_alert'])\n        "
    "self.assertTrue(row['no_occupant_exclusion_alert'])\n\n    def "
    'test_synthetic_exterior_alert_survives_exclusion(self):\n        row = '
    "next(r for r in self.proximity if r['case']=='outside_near_vehicle')\n   "
    "     self.assertTrue(row['current_alert'])\n        "
    "self.assertTrue(row['no_occupant_exclusion_alert'])\n\n    def "
    'test_projection_ambiguity_false_negative_is_explicit(self):\n        row '
    "= next(r for r in self.proximity if r['case']=='exterior_with_occupant_li"
    "ke_projection')\n        self.assertTrue(row['declared_expected_alert'])"
    "\n        self.assertFalse(row['current_alert'])\n        "
    "self.assertTrue(row['no_occupant_exclusion_alert'])\n"
)


def run_tests():
    (OUTPUT / 'source/tests/test_algorithm_contract.py').write_text(
        CONTRACT_TESTS,
    )
    nodes = (
        ['source/tests/src/danger_detector_test.py']
        + [
            f"source/tests/src/utils_test.py::TestUtils::{name}"
            for name in GEOMETRY_TESTS
        ]
        + [
            'source/tests/test_algorithm_contract.py',
        ]
    )
    command = [
        sys.executable,
        '-m',
        'pytest',
        '--import-mode=importlib',
        '-q',
        '--tb=short',
        '--junitxml=tests/snapshot_tests.xml',
        *nodes,
    ]
    env = {**os.environ, 'PYTHONPATH': str(OUTPUT / 'source')}
    result = subprocess.run(
        command,
        cwd=OUTPUT,
        env=env,
        capture_output=True,
        text=True,
    )
    (OUTPUT / 'tests/snapshot_tests.log').write_text(
        result.stdout + result.stderr,
    )
    dump(
        OUTPUT / 'tests/test_commands.json',
        dict(
            command=command,
            cwd=str(OUTPUT),
            source_import_root='source/',
            exit_code=result.returncode,
            original_test_nodes=['tests/src/danger_detector_test.py']
            + [
                f"tests/src/utils_test.py::TestUtils::{n}"
                for n in GEOMETRY_TESTS
            ],
        ),
    )
    rows = []
    for run in ['existing_tests', 'snapshot_tests']:
        xml = ET.parse(OUTPUT / 'tests' / f"{run}.xml")
        for test in xml.iter('testcase'):
            status = (
                'failed'
                if test.find('failure') is not None
                else (
                    'error'
                    if test.find(
                        'error',
                    )
                    is not None
                    else (
                        'skipped'
                        if test.find('skipped') is not None
                        else 'passed'
                    )
                )
            )
            name = test.get('name', '')
            caveat = (
                (
                    'Synthetic contract or documented limitation; not field '
                    'accuracy.'
                )
                if 'AlgorithmContractTests'
                in test.get(
                    'classname',
                    '',
                )
                else 'Unit test; not field accuracy.'
            )
            if name == 'test_driver_filtering':
                caveat = (
                    'Smoke test without a direct warning assertion; not '
                    'evidence of filtering '
                    'efficacy.'
                )
            if name in {
                'test_proximity_skips_driver_and_handles_degenerate_inputs',
                'test_detect_polygon_from_cones_with_noise',
            }:
                caveat = (
                    'Uses mocked occupant or clustering output; does not '
                    'validate real recognition '
                    'accuracy.'
                )
            rows.append(
                dict(
                    run=run,
                    classname=test.get('classname', ''),
                    name=name,
                    status=status,
                    elapsed_seconds=test.get('time', ''),
                    interpretation=caveat,
                ),
            )
    csv_write(OUTPUT / 'tests/test_results.csv', rows)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = {
        run: dict(
            tests=sum(r['run'] == run for r in rows),
            passed=sum(
                r['run'] == run and r['status'] == 'passed' for r in rows
            ),
        )
        for run in ['existing_tests', 'snapshot_tests']
    }
    dump(OUTPUT / 'tests/test_summary.json', summary)
    (OUTPUT / 'reproduce/run_tests.py').write_text(
        (
            'from pathlib import Path\nimport json, '
            'os, subprocess, sys\nroot = '
            'Path(__file__).resolve().parents[1]\nsaved'
            " = json.loads((root / 'tests/test_commands"
            ".json').read_text())\ncommand = "
            "[sys.executable, *saved['command'][1:]]\nr"
            'aise SystemExit(subprocess.run(command, '
            "cwd=root, env={**os.environ,'PYTHONPATH':s"
            "tr(root/'source')}).returncode)\n"
        ),
    )
    return summary


def write_docs(report, tests):
    formal = [
        r for r in report['actual_comparisons'] if r['scope'] == 'formal_11'
    ]
    expanded = [
        r
        for r in report['actual_comparisons']
        if r['scope'] == 'expanded_64_includes_formal'
    ]
    csv_write(
        OUTPUT / 'docs/test_conditions.csv',
        [
            dict(
                scope='formal_11',
                inputs=(
                    'Previously delivered formal figures, deduplicated by '
                    'original '
                    'SHA256'
                ),
                images=11,
                person_boxes=27,
                label_source='No independent intrusion/occupant labels',
                ground_truth_available=False,
            ),
            dict(
                scope='expanded_64_includes_formal',
                inputs=(
                    'Formal 11 plus 53 additional images from the 57-frame '
                    'same-camera '
                    'archive'
                ),
                images=64,
                person_boxes=121,
                label_source='No independent intrusion/occupant labels',
                ground_truth_available=False,
            ),
            dict(
                scope='synthetic_cone_25',
                inputs=(
                    '24 declared centre-rule cases in 3 layouts plus one '
                    'noise-outlier '
                    'case'
                ),
                images=0,
                person_boxes=25,
                label_source='Declared synthetic expectations',
                ground_truth_available='synthetic_only',
            ),
            dict(
                scope='synthetic_proximity_core',
                inputs='Occupant, exterior nearby person, exterior far person',
                images=0,
                person_boxes=3,
                label_source='Declared synthetic expectations',
                ground_truth_available='synthetic_only',
            ),
            dict(
                scope='synthetic_projection_ambiguity',
                inputs=(
                    'Hypothetical exterior person with occupant-like 2D '
                    'projection'
                ),
                images=0,
                person_boxes=1,
                label_source='Declared synthetic limitation scenario',
                ground_truth_available='synthetic_only',
            ),
        ],
    )
    (OUTPUT / 'docs/01_algorithm_explanation.txt').write_text(
        (
            '錐筒分群與禁區判斷：對照現有實作\n\n一、輸入與中心點\n偵測列格式為 '
            '[x1,y1,x2,y2,confidence,class_id,...]，使用原圖像素座標；x 向右，y '
            '向下。\nDangerDetector.detect_danger 入口先以 normalise_bbox '
            '整理左右、上下順序，並依追蹤／移動欄位篩選機具。\nclass_id=6 為錐筒、5 為人員、8 為機具、10 '
            '為車輛。geometry 函式本身假設輸入框座標已有效。\n錐筒中心 '
            'cx=(x1+x2)/2、cy=(y1+y2)/2；人員判斷點也採同一個框中心公式。\n不是腳底點、框底部中心或身體分割重心；人員'
            '框與區域重疊本身也不是進入條件。\n\n二、分群\n使用 sklearn.cluster.HDBSCAN(min_samples='
            '3,min_cluster_size=2,copy=True)。\n輸入為本張圖中所有錐筒中心的二維像素座標，以預設 '
            'Euclidean 距離運算。\n少於三個錐筒直接回傳空列表；標籤 -1 為雜訊並略過。\nmin_cluster_size=2 '
            '不代表兩點會形成管制區，後續仍要求每群至少三點。\n群號只代表該次結果，沒有跨影格持續追蹤群身分的機制。\n實際完整參數記於 '
            'environment.json，含 allow_single_cluster=False。\n\n三、邊界\n每個非雜訊群至少三'
            '個中心時，分別取 MultiPoint(cluster_points).convex_hull。\n分群後分別取凸包，可以保留不同'
            '群間的空隙；凸包本身會填滿同群內的凹陷。\n沒有把所有群自動連成單一大凸包，也沒有使用人工場域邊界或地面公尺座標。\npolygo'
            'ns_to_coords 再把 Polygon/MultiPolygon 邊界轉為前端頂點座標。\n\n四、人員進入\n針對 '
            'class=5 的人員取框中心，對各群凸包呼叫 contains(Point(cx,cy))。\n任一區 contains 為 '
            'True 即列入警示；恰在邊界上不算內部。\n人員以中心座標 tuple 放入 set '
            '去重，同一中心的多個框只計一次；不是用追蹤 ID 去重。\n結果大於零時輸出 '
            'warning_people_in_controlled_area.count。\n這是每張圖的幾何位置判斷，沒有連續入侵／離開事'
            '件或同一人軌跡的證明。\n\n五、現行邊界行為\n至少三點是必要條件，並不保證 HDBSCAN '
            '產生可用群。四角單群案例在目前預設下全為雜訊。\n現行程式未檢查凸包的幾何型別與面積。共線點可產生 '
            'LineString，contains 仍可能計入人員，但匯出座標會略過線段。\n本包 '
            'edge_case_observations.csv/JSON 已實際重現上述兩項限制；沒有為整理資料而修改生產程式。\n正式圖片'
            '所用的幾何都已在先前產圖時核對為有效 Polygon。\n\n六、乘員排除與接近規則\n依空間索引逐一處理候選機具／車輛，先檢查六'
            '項框幾何條件。\n任一候選符合乘員條件，整位人員的接近機具／車輛旗標即被排除，即使先前候選曾設為 '
            'True。\n不符合乘員條件，才對該候選比較人框／物件框面積比：車輛 <=0.10、機具 <=0.05。\n再比較 '
            'dx<=5*人寬 且 dy<=1.5*人高。\ndx=min(|p_right-v_left|,|p_left-v_right|)'
            '，dy=min(|p_bottom-v_top|,|p_top-v_bottom|)。\n兩項是像素框邊差指標，並非地面最近距離；'
            '乘員判斷也不是獨立駕駛身分辨識模型。\n乘員排除僅作用於接近規則，不會排除錐筒禁區警示。\n'
        ),
    )
    (OUTPUT / 'docs/02_pseudocode.txt').write_text(
        (
            '偽程式碼（依現行行為整理，完整來源見 source/src/）\n\nNORMALISE(row):\n    '
            'left=min(x1,x2); right=max(x1,x2)\n    top=min(y1,y2); '
            'bottom=max(y1,y2)\n    return [left,top,right,bottom,...remaining'
            '_fields]\n\nCENTRE(row):\n    return ((row.x1+row.x2)/2, '
            '(row.y1+row.y2)/2)\n\nCONE_REGIONS(detections):\n    '
            'points=[CENTRE(row) for row in detections if row.class_id==6]\n  '
            '  if len(points)<3: return []\n    '
            'labels=HDBSCAN(min_samples=3,min_cluster_size=2,copy=True).fit_pr'
            'edict(points)\n    groups=group points by label, skipping '
            'label==-1\n    regions=[]\n    for group in groups:\n        if '
            'len(group)>=3:\n            regions.append(convex_hull(group))\n '
            '           # Actual implementation does NOT reject '
            'Point/LineString/zero area here.\n    return '
            'regions\n\nCOUNT_PEOPLE_IN_REGIONS(regions,detections):\n    '
            'unique_centres=set()\n    for person in detections where '
            'class_id==5:\n        centre=CENTRE(person)\n        for region '
            'in regions:\n            if region.contains(Point(centre)):\n    '
            '            unique_centres.add(centre)\n                break\n  '
            '  return len(unique_centres)\n\nALARM_CONE_AREA(detections):\n   '
            ' regions=CONE_REGIONS(detections)\n    '
            'count=COUNT_PEOPLE_IN_REGIONS(regions,detections)\n    if '
            "count>0: warnings['warning_people_in_controlled_area']={'count':c"
            'ount}\n\nIS_OCCUPANT(p,v):\n    pw=p.right-p.left; '
            'ph=p.bottom-p.top; vh=v.bottom-v.top\n    return '
            '(p.bottom<v.bottom\n        and v.bottom-p.bottom>=ph/2\n        '
            'and p.left>=v.left-pw/2\n        and p.right<=v.right+pw/2\n     '
            '   and p.top>v.top\n        and ph<=vh/2)\n\nPER_PERSON_PROXIMITY'
            '(p,candidates):\n    machinery_flag=False; vehicle_flag=False\n  '
            '  for v in spatial_index_candidates(p):\n        if '
            'IS_OCCUPANT(p,v):\n            return '
            '(occupant=True,machinery=False,vehicle=False)\n        if '
            'area(v)<=0: continue\n        ratio_limit=0.10 if v.class_id==10 '
            'else 0.05\n        if area(p)/area(v)>ratio_limit: continue\n    '
            '    dx=min(abs(p.right-v.left),abs(p.left-v.right))\n        '
            'dy=min(abs(p.bottom-v.top),abs(p.top-v.bottom))\n        if '
            'dx<=5*width(p) and dy<=1.5*height(p):\n            if '
            'v.class_id==8: machinery_flag=True\n            if '
            'v.class_id==10: vehicle_flag=True\n    return '
            '(occupant=False,machinery_flag,vehicle_flag)\n\nCOMPARISON_BASELI'
            'NES:\n    B1=one convex hull of ALL detected cone centres '
            '(includes noise).\n    B2=one convex hull of centres retained in '
            'valid clustered polygons.\n       B2 versus current per-cluster '
            'regions isolates the filling of inter-cluster gaps.\n    '
            'P0=unchanged production proximity code with IS_OCCUPANT always '
            'False.\n       P0 changes only the exclusion predicate, '
            'retaining spatial indexing and thresholds.\n\nThese baselines '
            'are experiments, not alternative production '
            'features.\n'
        ),
    )
    lines = [
        '測試條件、比較結果與可支持的範圍',
        '',
        '一、實際影像框回放（不是獨立效能測試）',
        '資料來自先前圖片及研究篩選；已按原圖 SHA256 去重。正式 11 張是擴充 64 張的一部分，不可相加為 75 張。',
        '所有回放均使用既有 YOLO26m、conf=0.25、imgsz=1280 的框；沒有重新偵測圖片。',
        '計數單位為逐圖偵測人員框，不是獨立受試者或跨影格不重複人數。',
        '',
    ]
    names = {
        'current_clustered_regions': '現行各群凸包',
        'single_all_cones_hull': '全部錐筒單一凸包（含雜訊）',
        'single_clustered_points_hull': '保留群點後合成單一凸包（控制雜訊因素）',
        'current_proximity_with_exclusion': '現行人車／人機接近（含乘員排除）',
        'proximity_without_exclusion': '同規則關閉乘員排除',
    }
    for title, selected in [
        ('正式 11 張／27 個人員框', formal),
        ('擴充 64 張／121 個人員框（包含正式資料）', expanded),
    ]:
        lines.append(title)
        lines.extend(
            f"  {names[r['method']]}：{r['alert_person_boxes']} 個人員框有警示"
            for r in selected
        )
        lines.append('')
    lines += [
        '擴充集有 14 個人員框符合乘員排除條件，但只有 1 個在關閉排除後新增接近警示；不能把 14 個都算成避免誤報。',
        '正式 11 張中的接近警示為 5 對 5；沒有在這個子集中觀察到警示數降低。',
        '上述減少的警示可能是正確排除，也可能包含漏警；沒有獨立正解就不能稱為誤報改善。',
        '',
        '二、實際誤報／漏報狀態',
        '現有類別框標註不含獨立的禁區進入、駕駛／乘員身分或應警示真值，並且本批包含訓練素材與選樣案例。',
        '因此實際影像 FP、FN、precision、recall 皆列 NA，不填零、不以目前演算法輸出反過來作正解。',
        '回放從既有偵測框開始，也不能量到 YOLO 漏掉整個人員或錐筒的端到端漏報。',
        '',
        '三、合成受控情境（僅證明機制／限制，不是實景準確率）',
    ]
    for row in report['synthetic_comparisons']:
        lines.append(
            f"{
                row[(
                    'scope'
                )]
            } / {
                row[(
                    'method'
                )]
            }: n={
                row[(
                    'samples'
                )]
            }, TP={
                row[(
                    'tp'
                )]
            }, FP={
                row[(
                    'fp'
                )]
            }, TN={
                row[(
                    'tn'
                )]
            }, FN={
                row[(
                    'fn'
                )]
            }",
        )
    lines += [
        '',
        '錐筒的 25 個判斷點包含 8 種中心規則情境的 3 個平移／縮放版本，以及 1 個離群點案例；不是 25 張實拍照片。',
        '乘員核心情境是 3 個預先宣告的示例；另外 1 個假設外部人員投影在車框內的限制案例，現行排除會產生 FN。',
        '合成情境的 TP/FP/TN/FN 只相對於預先宣告的情境目標，不得引用為現場誤報率或漏報率。',
        '',
        '四、程式測試紀錄',
    ]
    for name, row in tests.items():
        lines.append(f"{name}: {row['passed']}/{row['tests']} passed")
    lines += [
        'snapshot_tests 包含同一批 30 個既有測試，加上 11 個合成契約／限制觀察測試；兩次測試數不可相加當作獨立樣本。',
        (
            '既有 test_driver_filtering 沒有直接斷言排除結果；另有使用 mock '
            '的測試。test_results.csv '
            '已逐列標示解讀限制。'
        ),
        '共線及單群限制測試是確認現有行為的觀察測試；測試通過不代表這些限制已修正。',
    ]
    (OUTPUT / 'docs/03_test_results_and_effects.txt').write_text(
        '\n'.join(lines) + '\n',
    )
    (OUTPUT / 'docs/04_ground_truth_and_followup.txt').write_text(
        """如何補足實際誤報／漏報證據

manual_label_template.csv 留白，未用目前警示當正解；inference 比較結果在其他 CSV 中分開提供。
建議由獨立標註者依原圖及事先定義的場域規則標註：真實人員框、應有管制區邊界、是否進區、是否乘員、是否應發接近警示。
原圖可依 source_path 及 SHA256 在先前完整圖片資料包中查找；本 ZIP 不含圖片或權重。
靜態投影難以判斷者應標為 unknown，不要強制歸類。人員是否在車內不應只以本程式六條件作正解。
先制定相同警示政策，再比較分群／單凸包、開啟／關閉乘員排除，其他框與門檻保持相同。
若要量到模型漏檢，須從完整原圖標註全部人員與錐筒，不能只標已有偵測框。
按攝影機、工地與日期切分獨立測試集，避免同機位近時刻重複畫面被當成獨立樣本。
報告每種情境的正負例數、未知／排除例數、TP/FP/TN/FN、precision/recall，以及場域與取樣方式。
若聲稱地面距離門檻，需另有相機標定及實測距離；本次只提供像素框邊指標。
目前資料可支持程式處理步驟、可重現警示差異及受控案例的機制解釋；尚不足以支持現場誤報／漏報改善率。
""",
    )


def main():
    for folder in [
        'docs',
        'inputs',
        'results',
        'tests',
        'source/src',
        'source/tests/src',
        'reproduce',
    ]:
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
    collect()
    snapshot()
    report = evaluate(
        OUTPUT / 'inputs/replay_dataset.json',
        OUTPUT / 'results',
    )
    tests = run_tests()
    write_docs(report, tests)
    packages = [
        'numpy',
        'scipy',
        'scikit-learn',
        'shapely',
        'pytest',
        'aiohttp',
        'PyJWT',
        'redis',
    ]
    dump(
        OUTPUT / 'environment.json',
        dict(
            created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
            python=sys.version,
            packages={p: importlib.metadata.version(p) for p in packages},
            clusterer_defaults=__import__(
                'sklearn.cluster',
                fromlist=['HDBSCAN'],
            )
            .HDBSCAN(
                min_samples=3,
                min_cluster_size=2,
                copy=True,
            )
            .get_params(),
            source_sha256={
                str(p.relative_to(ROOT)): sha(p)
                for p in [
                    ROOT / 'src/geometry.py',
                    ROOT / 'src/danger_detector.py',
                ]
            },
        ),
    )
    (OUTPUT / 'reproduce/requirements_observed.txt').write_text(
        ''.join(f"{p}=={importlib.metadata.version(p)}\n" for p in packages),
    )
    (OUTPUT / 'README.txt').write_text(
        (
            '錐筒分群、禁區與乘員排除：非圖片補充資料（2026-10-22）\n\n此 ZIP '
            '與之前圖片包分開，內容為文字、程式、JSON、CSV、測試日誌及 JUnit '
            'XML；不含照片、圖片或模型權重。\n\n閱讀入口\ndocs/01_algorithm_explanation.txt：中心點、'
            'HDBSCAN、逐群凸包、contains、人員去重與乘員規則。\ndocs/02_pseudocode.txt：與現行行為對應的'
            '完整偽程式碼。\ndocs/related_functions.txt、function_index.csv：精確函式摘錄、原檔行'
            '號及雜湊；完整來源在 source/src/。\ndocs/03_test_results_and_effects.txt：條件、'
            '樣本數、比較結果、誤報漏報狀態與解讀範圍。\ndocs/04_ground_truth_and_followup.txt：後續實景'
            '效能驗證需要的標註與方法。\n\nCSV 與測試\nresults/real_comparison_summary.csv：正式 '
            '11 張／27 框及擴充 64 張／121 框的五種方法比較；真實 FP/FN 列 '
            'NA。\nresults/real_person_comparisons.csv：121 '
            '筆逐人框判斷；real_scene_conditions.csv 為 64 '
            '張條件與來源。\nresults/changed_decisions.csv：警示有差異的資料；不是誤報標註。\nresults/'
            'synthetic_cone_cases.csv、synthetic_proximity_cases.csv、synthetic_'
            'confusion_counts.csv：合成情境及其宣告標籤的 TP/FP/TN/FN。\nresults/edge_case_'
            'observations.csv/JSON：單群全雜訊、共線線段與匯出不一致的實際觀察。\nresults/manual_labe'
            'l_template.csv：待獨立人工標註的空白表，未把程式輸出當正解。\ntests/existing_tests.log/X'
            'ML：30 個既有相關測試；snapshot_tests.log/XML：30 個既有加 11 '
            '個合成契約／限制觀察測試。\ntests/test_results.csv、test_summary.json、test_comm'
            'ands.json：逐項結果、耗時、命令與解讀限制。\n\n可重算\n在與 environment.json '
            '相容的環境中，從解壓後目錄執行：\n  python reproduce/evaluate_algorithm_evidence.'
            'py\n輸出在 recomputed/，不需要圖片、YOLO 權重、攝影機、資料庫或網路服務。\n  python '
            'reproduce/run_tests.py\n可重跑來源快照與合成契約測試。requirements_observed.txt '
            '是本次實際依賴版本紀錄。\nreproduce/build_algorithm_evidence.py '
            '是本次打包程式快照；重建整包需原專案及歷次輸出，純數值重算不需要。\n\n重要範圍\n64 張包含正式 11 張；121 '
            '是逐圖偵測框數，不是 121 個獨立人。\n現有資料沒有獨立的禁區進入或乘員真值，因此實景誤報、漏報與 '
            'precision/recall 不可計算，CSV 中為 NA。\n分群／乘員比較是保持同一批框的警示差異；警示減少不等於誤報減少'
            '。\n擴充集乘員排除為 22→21 個有接近警示的框；正式子集是 5→5，並非所有排除都降低警示數。\n合成情境的誤報／漏報僅相對'
            '預先宣告的示例目標，不是現場效能數據；限制示例有一個現行 FN，已明列。\n沒有修改原始生產演算法；已如實記錄單一群全雜訊及共線凸'
            '包退化的行為。\n所有原圖與權重仍在先前的完整整合圖片包 v2，可用來源 SHA256 '
            '對照。\n\n完整性：MANIFEST.json 及 SHA256SUMS.txt '
            '提供逐檔雜湊。\n'
        ),
    )
    # Replay the portable evaluator against the packaged source snapshot.
    command = [
        sys.executable,
        str(
            OUTPUT / 'reproduce/evaluate_algorithm_evidence.py',
        ),
    ]
    replay = subprocess.run(
        command,
        cwd=OUTPUT,
        capture_output=True,
        text=True,
    )
    (OUTPUT / 'tests/portable_replay.log').write_text(
        replay.stdout + replay.stderr,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    replay_checks = []
    for p in sorted((OUTPUT / 'results').glob('*')):
        other = OUTPUT / 'recomputed' / p.name
        assert sha(p) == sha(other), p.name
        replay_checks.append(
            dict(
                file=p.name,
                sha256=sha(
                    p,
                ),
                snapshot_recomputed_exact_match=True,
            ),
        )
    dump(OUTPUT / 'tests/portable_replay_validation.json', replay_checks)
    shutil.rmtree(OUTPUT / 'recomputed')
    for path in OUTPUT.rglob('__pycache__'):
        shutil.rmtree(path)
    # Exclude transient pytest caches from the delivered files.
    for path in OUTPUT.rglob('.pytest_cache'):
        shutil.rmtree(path)
    assert not any(
        p.suffix.lower()
        in {
            '.png',
            '.jpg',
            '.jpeg',
            '.pdf',
            '.pt',
            '.onnx',
            '.mp4',
            '.gif',
        }
        for p in OUTPUT.rglob('*')
        if p.is_file()
    )
    files = [
        dict(
            path=str(p.relative_to(OUTPUT)),
            bytes=p.stat().st_size,
            sha256=sha(p),
        )
        for p in sorted(OUTPUT.rglob('*'))
        if p.is_file() and p.name not in {'MANIFEST.json', 'SHA256SUMS.txt'}
    ]
    dump(
        OUTPUT / 'MANIFEST.json',
        dict(
            created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
            contents='Non-image algorithm/evaluation evidence only',
            real_images_replayed=report['unique_images'],
            detected_person_boxes=report['detected_person_boxes'],
            real_world_fp_fn_available=False,
            tests=tests,
            portable_replay_exact_match=True,
            files=files,
        ),
    )
    sums = [(r['sha256'], r['path']) for r in files] + [
        (sha(OUTPUT / 'MANIFEST.json'), 'MANIFEST.json'),
    ]
    (OUTPUT / 'SHA256SUMS.txt').write_text(
        ''.join(f"{h}  {p}\n" for h, p in sums),
    )
    archive = OUTPUT.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file():
                zipped.write(p, p.relative_to(OUTPUT.parent))
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
    print(
        json.dumps(
            dict(
                archive=str(archive),
                bytes=archive.stat().st_size,
                files=len(sums) + 1,
                real_images=report['unique_images'],
                person_boxes=report['detected_person_boxes'],
                tests=tests,
                portable_replay_exact_match=True,
            ),
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == '__main__':
    main()
