from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from PIL import Image
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_patent_figures import (  # noqa: E402
    INK,
    MUTED,
    save_csv,
    sha,
    text,
)
from scripts.generate_aerial_cone_figures import wrap  # noqa: E402

BATCHES = [
    (
        '01_original_examples',
        'patent_figures_20261021',
        '第一批：流程、人物中心及乘員排除',
        'patent_supplement_figures.pdf',
    ),
    (
        '02_elevated_views',
        'aerial_cone_zones_20261021',
        '第二批：工地俯瞰與錐筒管制區',
        'aerial_cone_zones.pdf',
    ),
    (
        '03_elevated_people',
        'aerial_people_20261021',
        '新增：俯瞰同畫面管制區內／外人員',
        'aerial_people.pdf',
    ),
    (
        '04_proximity_process',
        'driver_proximity_20261021',
        '新增：乘員排除與人車接近判定過程',
        'driver_proximity.pdf',
    ),
]
COMBINED_PAGES = 39


def cover(title, paragraphs):
    im = Image.new('RGB', (2480, 1754), 'white')
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 2480, 16), fill=INK)
    text(d, (90, 72), title, 62, bold=True)
    y = 205
    for value, heading in paragraphs:
        size = 38 if heading else 29
        if heading:
            y += 18
        for line in wrap(d, value, 2280, size):
            text(
                d,
                (95, y),
                line,
                size,
                INK if heading else MUTED,
                bold=heading,
            )
            y += size + 18
        y += 15
    assert y < 1690, f"Cover text overflows: {y}"
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'outputs/supplement_compilation_20261021_v2',
    )
    args = parser.parse_args()
    out = args.output.resolve()
    workspace = out / 'workspace'
    for folder in [
        'data',
        'index',
        'workspace/scripts',
        'workspace/src',
        'workspace/models/pt',
        'research',
    ]:
        (out / folder).mkdir(parents=True, exist_ok=True)
    (
        source_catalog,
        detections,
        cones,
        people,
        vertices,
        pairs,
        warnings,
        figures,
    ) = ([], [], [], [], [], [], [], [])
    all_source_paths = set()
    source_count, figure_count = 0, 0
    batchinfo = []
    for target, source, description, pdf in BATCHES:
        src = ROOT / 'outputs' / source
        dst = out / target
        shutil.copytree(src, dst, dirs_exist_ok=True)
        m = json.loads((src / 'data/results.json').read_text())
        model = m['model']
        model_sha = (
            model['sha256']
            if isinstance(
                model,
                dict,
            )
            else m['model_sha256']
        )
        batchinfo.append(
            dict(
                id=target,
                title=description,
                source_directory=str(
                    src.relative_to(ROOT),
                ),
                original_pdf=f"{target}/{pdf}",
                sources=len(m['sources']),
                png_count=len(list((dst / 'figures').glob('*.png'))),
            ),
        )
        for sid, r in m['sources'].items():
            source_count += 1
            all_source_paths.add(r['source']['path'])
            source_catalog.append(
                dict(
                    batch=target,
                    source=sid,
                    description=r['source']['description'],
                    original_file=f"{target}/{r['original_copy']}",
                    original_repo_path=r['source']['path'],
                    source_sha256=r['source_sha256'],
                    width=r['size'][0],
                    height=r['size'][1],
                    model_sha256=model_sha,
                    capture_time_visible=r['source'].get(
                        'capture_time_visible',
                        '',
                    ),
                    cones=len(r['cones']),
                    zones=len(r['zones']),
                    people=len(r['people']),
                ),
            )
            assert sha(dst / r['original_copy']) == r['source_sha256']
            for i, row in enumerate(r['detections']):
                detections.append(
                    dict(
                        batch=target,
                        source=sid,
                        detection=i + 1,
                        x1=row[0],
                        y1=row[1],
                        x2=row[2],
                        y2=row[3],
                        confidence=row[4],
                        class_id=int(row[5]),
                    ),
                )
            for co in r['cones']:
                cones.append(
                    dict(
                        batch=target,
                        source=sid,
                        cone=co['id'],
                        cx=co['center'][0],
                        cy=co['center'][1],
                        confidence=co['confidence'],
                        cluster_label=co['cluster_label'],
                        zone=co['zone_id'],
                    ),
                )
            for p in r['people']:
                people.append(
                    dict(
                        batch=target,
                        source=sid,
                        person=p['id'],
                        cx=p['center'][0],
                        cy=p['center'][1],
                        confidence=p['confidence'],
                        inside_zones=';'.join(p['inside_zones']),
                        between_zone_pairs=json.dumps(p['between_zone_pairs']),
                        cone_alert=p['cone_alert'],
                        occupant_excluded=p['occupant_excluded'],
                        near_vehicle=p['near_vehicle_alert'],
                        near_machinery=p['near_machinery_alert'],
                    ),
                )
            for zone in r['zones']:
                for i, v in enumerate(zone['vertices']):
                    vertices.append(
                        dict(
                            batch=target,
                            source=sid,
                            zone=zone['id'],
                            cluster_label=zone['cluster_label'],
                            vertex_index=i,
                            x=v[0],
                            y=v[1],
                            area_px2=zone['area_px2'],
                        ),
                    )
            pairs.extend(
                dict(batch=target, source=sid, **pair)
                for pair in r['proximity_pairs']
            )
            warnings.append(
                dict(batch=target, source=sid, warnings=r['warnings']),
            )
        for im in sorted((dst / 'figures').glob('*.png')):
            with Image.open(im) as picture:
                picture.verify()
            figures.append(
                dict(
                    batch=target,
                    figure=im.stem,
                    path=str(im.relative_to(out)),
                    sha256=sha(im),
                ),
            )
            figure_count += 1
        print(
            'Preserved',
            target,
            batchinfo[-1]['png_count'],
            'PNG',
            flush=True,
        )
    # Preserve exploratory outputs and all images referenced by scans.
    scanfiles = [
        out / '01_original_examples/scout.json',
        out / '01_original_examples/refined.json',
        out / 'research/aerial_people_scan.json',
    ]
    for file in scanfiles:
        for row in json.loads(file.read_text()):
            all_source_paths.add(row['path'])
    for file in ['camera_refined.json', 'aerial_analysis.json']:
        for row in json.loads((out / 'research' / file).read_text()):
            all_source_paths.add(row['source']['path'])
    for file in [
        'hazard_sample_paths.json',
        'hazard_date_paths.json',
        'refine_paths.json',
        'inside_paths.json',
        'aerial_candidates.json',
    ]:
        for path in json.loads((out / 'research' / file).read_text()):
            all_source_paths.add(path)
    for row in json.loads(
        (out / 'research/aerial_raw_paths.json').read_text(),
    ):
        all_source_paths.add(row[2])
    raw_catalog = []
    annotation_count = 0
    class_maps = []
    for name in [
        'examples/YOLO_train/cv_dataset/data.yaml',
        'examples/YOLO_train/datasets/data.yaml',
        'tests/cv_dataset/data.yaml',
        'tests/dataset/data.yaml',
    ]:
        src = ROOT / name
        dst = workspace / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        parsed = yaml.safe_load(src.read_text())
        class_maps.append(
            dict(
                path=f"workspace/{name}",
                sha256=sha(src),
                nc=parsed['nc'],
                names=parsed['names'],
            ),
        )
    (out / 'data/class_maps.json').write_text(
        json.dumps(
            dict(
                datasets=class_maps,
                note=(
                    'Existing annotations use their original dataset class '
                    'map. Historical test datasets have 10 classes; the final '
                    'model uses 11. Preserved annotations were not '
                    'substituted for '
                    'predictions.'
                ),
            ),
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
    )
    for i, name in enumerate(sorted(all_source_paths)):
        src = ROOT / name
        assert src.is_file(), name
        dst = workspace / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        original_sha = sha(src)
        assert sha(dst) == original_sha
        annotation = ''
        class_map = ''
        if src.parent.name == 'images':
            label = src.parent.parent / 'labels' / (src.stem + '.txt')
            if label.is_file():
                annotation = str(label.relative_to(ROOT))
                destlabel = workspace / annotation
                destlabel.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label, destlabel)
                annotation_count += 1
                for parent in [src.parent.parent, *src.parent.parent.parents]:
                    if parent == ROOT.parent:
                        break
                    if (parent / 'data.yaml').is_file():
                        class_map = str(
                            (parent / 'data.yaml').relative_to(ROOT),
                        )
                        break
        raw_catalog.append(
            dict(
                original_repo_path=name,
                archive_path=str(
                    dst.relative_to(out),
                ),
                sha256=original_sha,
                bytes=src.stat().st_size,
                existing_label_path=(
                    f"workspace/{annotation}" if annotation else ''
                ),
                annotation_class_map=(
                    f"workspace/{class_map}" if class_map else ''
                ),
            ),
        )
        if i % 200 == 0:
            print(
                'Copied source evidence',
                i,
                '/',
                len(all_source_paths),
                flush=True,
            )
    save_csv(out / 'data/source_catalog.csv', source_catalog)
    save_csv(out / 'data/all_detections.csv', detections)
    save_csv(out / 'data/all_cones.csv', cones)
    save_csv(out / 'data/all_people.csv', people)
    save_csv(out / 'data/all_zone_vertices.csv', vertices)
    save_csv(out / 'data/all_proximity_pairs.csv', pairs)
    save_csv(out / 'index/figure_catalog.csv', figures)
    save_csv(out / 'research/all_source_files.csv', raw_catalog)
    (out / 'data/all_warnings.json').write_text(
        json.dumps(
            warnings,
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
    )
    for name in ['occupant_conditions.csv', 'proximity_metrics.csv']:
        shutil.copy2(
            out / '04_proximity_process/data' / name,
            out / 'data' / name,
        )
    process = json.loads(
        (out / '04_proximity_process/data/results.json').read_text(),
    )
    (out / 'data/proximity_process_trace.json').write_text(
        json.dumps(
            dict(
                batch='04_proximity_process',
                cases=process['cases'],
                validation=process['validation'],
            ),
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
    )
    # Bundle weights with their exact hashes, plus a small runnable workspace.
    models = []
    for name, role in [
        ('best_yolo26m.pt', 'All final figures'),
        ('best_yolo26n.pt', 'Exploratory candidate screening'),
    ]:
        src = ROOT / 'models/pt' / name
        dst = workspace / 'models/pt' / name
        shutil.copy2(src, dst)
        assert sha(src) == sha(dst)
        models.append(
            dict(
                path=str(dst.relative_to(out)),
                role=role,
                sha256=sha(dst),
                bytes=dst.stat().st_size,
            ),
        )
    for row in source_catalog:
        assert row['model_sha256'] == models[0]['sha256']
    for name in [
        'generate_patent_figures.py',
        'generate_aerial_cone_figures.py',
        'generate_aerial_people_figures.py',
        'generate_driver_proximity_figures.py',
        'build_supplement_bundle.py',
        'patent_figure_sources.json',
        'aerial_cone_sources.json',
        'aerial_people_sources.json',
        '__init__.py',
    ]:
        shutil.copy2(ROOT / 'scripts' / name, workspace / 'scripts' / name)
    for name in ['__init__.py', 'geometry.py', 'danger_detector.py']:
        shutil.copy2(ROOT / 'src' / name, workspace / 'src' / name)
    for name in ['pyproject.toml', 'requirements.txt', 'uv.lock']:
        shutil.copy2(ROOT / name, workspace / name)
    environment = dict(
        python=sys.version,
        executable_used=str(sys.executable),
        packages={
            d.metadata['Name']: d.version
            for d in importlib.metadata.distributions()
            if d.metadata['Name']
        },
        note=(
            'Observed runtime; repository dependency files are preserved as '
            'separate references.'
        ),
    )
    (workspace / 'environment.json').write_text(
        json.dumps(environment, indent=2) + '\n',
    )
    (workspace / 'run_figures.py').write_text(
        (
            'from pathlib import Path\nimport '
            'subprocess\nimport sys\nroot = '
            'Path(__file__).resolve().parent\nfor '
            "script in ['generate_patent_figures.py', "
            "'generate_aerial_cone_figures.py', "
            "'generate_aerial_people_figures.py', "
            "'generate_driver_proximity_figures.py']:\n"
            '    subprocess.run([sys.executable, '
            "str(root / 'scripts' / script)], "
            "cwd=root, check=True)\nprint('Reproduced "
            'all four batches under '
            "workspace/outputs/.')\n"
        ),
    )
    (workspace / 'README_REPRODUCE.txt').write_text(
        """正式交付圖之可重跑工作目錄

在與 environment.json 相容的 Python 環境執行：
  python run_figures.py

本次實際環境為 Python 3.14；套件精確版本列於 environment.json。
所有正式圖使用的原始圖片、YOLO26m 權重、三個來源清單與程式均已包含。
第四節的判定過程使用第一批已產生的偵測結果回放；run_figures.py 會依正確順序執行。
不需要連接攝影機、資料庫、通知服務或取得外部圖片。
字型依既有系統路徑使用 NotoSansCJK-Regular.ttc / Bold.ttc。
另附 YOLO26n 權重供研究篩選核對，正式圖一律使用 YOLO26m。
pyproject.toml、requirements.txt、uv.lock 為專案參考檔；實際產圖環境以 environment.json 為準。
research/ 的研究腳本保留原始路徑及選樣過程，部分需要完整訓練資料集；正式圖的重跑不依賴它們。
""",
    )
    research_metadata = dict(
        scan_1010=dict(
            file='research/aerial_people_scan.json',
            model=models[1],
            imgsz=960,
            conf=0.25,
            device='cpu',
            candidates=1010,
        ),
        same_camera_refinement=dict(
            file='research/camera_refined.json',
            model=models[0],
            imgsz=1280,
            conf=0.25,
            candidates=57,
        ),
        provenance=(
            'Screening records include augmented training examples and '
            'unrelated rejected candidates; final evidence is identified '
            'separately in data/source_catalog.csv.'
        ),
        ocr=(
            'OCR was used for screening only. Final E1–E4 timestamps were '
            'visually verified against '
            'originals.'
        ),
    )
    (out / 'research/screening_metadata.json').write_text(
        json.dumps(research_metadata, ensure_ascii=False, indent=2) + '\n',
    )
    pages = [
        cover(
            '施工安全影像補充資料｜整合版 v2',
            [
                ('閱讀順序與重點', True),
                (
                    (
                        '共 '
                        f"{figure_count}"
                        ' 張正式 PNG、'
                        f"{source_count}"
                        ' 筆來源紀錄（跨章可重複引用原圖）；原有資料全部保留。'
                    ),
                    False,
                ),
                ('本次新增：乘員排除與靠近車輛人員的判定過程', True),
                (
                    (
                        '04_proximity_process：六項乘員幾何條件、逐候選判定流程、面積比、水平與垂直框邊指標，附'
                        '三種實際結果。'
                    ),
                    False,
                ),
                (
                    '包含操作室內人員、車外接近警示，以及車外人員因面積比未通過而不警示的案例。',
                    False,
                ),
                ('原有資料', True),
                (
                    '01_original_examples：六步驟流程、禁區內／外／兩區之間、人車接近、乘員幾何排除。',
                    False,
                ),
                (
                    '02_elevated_views：四張工地俯瞰原圖、錐筒分群管制區與逐步流程。',
                    False,
                ),
                (
                    '03_elevated_people：同一俯瞰畫面管制區內／外人員及同機位不同時間的獨立取樣。',
                    False,
                ),
                ('資料與程式', True),
                (
                    (
                        'data/ 為合併 CSV、完整警示 JSON 與來源索引；workspace/ '
                        '含原圖、既有標註、兩個實際模型權重及重跑程式。'
                    ),
                    False,
                ),
                (
                    'research/ 保留候選篩選、57 張同機位複核與 OCR 研究紀錄；所有取樣原檔也已備存。',
                    False,
                ),
            ],
        ),
        cover(
            '判讀規則、來源及檔案索引',
            [
                ('人員中心與管制區', True),
                (
                    '錐筒框中心 → HDBSCAN → 各群凸包 → 人員框中心 contains；邊界點不算內部，不跨群自動填滿。',
                    False,
                ),
                (
                    (
                        '每張圖使用自己的偵測框與分群結果。Z1/P1 等編號僅在該批／該張來源中有效；合併資料含 batch 與 '
                        'source '
                        '欄位。'
                    ),
                    False,
                ),
                ('俯瞰「進出」的證據範圍', True),
                (
                    '現有照片支持同機位靜態內／外判斷，未證明同一人的連續進出軌跡。不可把四張取樣稱為已追蹤到的同一人。',
                    False,
                ),
                (
                    '新俯瞰原圖 640×360；標示圖與局部圖僅等比例放大，沒有合成人物或增加原始細節。',
                    False,
                ),
                ('乘員與距離', True),
                (
                    '乘員排除是既有框幾何規則，並非獨立身分類別；人車距離是像素框邊指標，未標定為公尺。',
                    False,
                ),
                ('檔案核對', True),
                (
                    (
                        'index/figure_catalog.csv：全部正式圖片；data/source_catalog.c'
                        'sv：正式原圖、日期及模型雜湊。'
                    ),
                    False,
                ),
                (
                    (
                        'data/all_cones.csv、all_people.csv、all_detections.csv、'
                        'all_zone_vertices.csv、all_proximity_pairs.csv：統一數據。'
                    ),
                    False,
                ),
                (
                    (
                        'MANIFEST.json、SHA256SUMS.txt、verify_bundle.py：逐檔檢查；wo'
                        'rkspace/environment.json：本次實際軟體版本。'
                    ),
                    False,
                ),
            ],
        ),
    ]
    pages[0].save(
        out / 'index/reading_guide.pdf',
        save_all=True,
        append_images=pages[1:],
        resolution=212.1,
        quality=95,
    )
    subprocess.run(
        [
            'pdfunite',
            str(out / 'index/reading_guide.pdf'),
            str(out / '04_proximity_process/driver_proximity.pdf'),
            str(out / '03_elevated_people/aerial_people.pdf'),
            str(out / '01_original_examples/patent_supplement_figures.pdf'),
            str(out / '02_elevated_views/aerial_cone_zones.pdf'),
            str(out / '00_combined_figures.pdf'),
        ],
        check=True,
    )
    readme = (
        '施工安全影像補充資料｜完整整合版 v2（2026-10-21）\n\n先看：00_combined_'
        'figures.pdf（'
        f"{COMBINED_PAGES}"
        ' 頁）\n本包完整保留先前各批，新增操作室內人員與靠近車輛人員的逐項判定過程。\n正式圖片共 '
        f"{figure_count}"
        ' 張 PNG，正式來源紀錄 '
        f"{source_count}"
        ' 筆。\n\n01_original_examples/：第一批全部原圖、15 張 PNG、15 '
        '頁 PDF、CSV/JSON、重跑程式及原篩選結果。\n02_elevated_views/：第二批'
        '全部原圖、17 張 PNG、10 頁 PDF、CSV/JSON、重跑程式。\n03_elevated'
        '_people/：新增 4 張原圖、17 張 PNG、5 頁 '
        'PDF、人物中心、局部放大與同機位不同時間取樣。\n04_proximity_process/：7 '
        '張 PNG、7 頁 PDF、六項乘員條件、三種判定結果與逐候選回放資料。\n\n建議採用的主圖\n0'
        '4_proximity_process/figures/00_three_case_comparis'
        'on.png：乘員排除、接近車輛警示、面積比不通過三案例。\n04_proximity_proces'
        's/figures/01_decision_flow.png：逐人、逐候選的完整判定流程。\n03_'
        'elevated_people/figures/00_same_frame_inside_outsi'
        'de.png\n同一張俯瞰照片 E4 同時有 P1 中心位於管制區內、P2 '
        '中心位於管制區外，附座標及局部放大。\n同機位另取 E1（08:16:03 '
        '外）、E2（08:21:23 內）、E3（09:09:40 外），E4 為 '
        '09:37:21。\n這些照片不構成同一人的連續進出軌跡。P 編號不是跨影格追蹤 '
        'ID。\n\n完整資料\ndata/source_catalog.csv：每筆正式原圖的來源、原檔位'
        '置、寬高、模型雜湊與偵測數量。\ndata/all_detections.csv：四節引用的完整模型'
        '框、類別、信心值。\ndata/all_cones.csv：錐筒編號、中心、群別、管制區。\ndat'
        'a/all_people.csv：人物中心、區內／區外／區間、乘員排除、接近警示。\ndata/al'
        'l_zone_vertices.csv：全部管制區頂點及面積；data/all_proximity_'
        'pairs.csv：逐對人車判斷數值。\ndata/all_warnings.json：全部原始警示'
        '，不只列出圖上重點規則。\ndata/occupant_conditions.csv、proximi'
        'ty_metrics.csv、proximity_process_trace.json：六條件、距離'
        '指標與逐候選分支。\nindex/figure_catalog.csv：正式圖片清單與雜湊。\n同名'
        '來源 A/B/C、人物 P1、管制區 Z1 必須結合 batch 與 source '
        '欄位識別。\n\n原始資料、研究過程及模型\nworkspace/ 保留 '
        f"{len(raw_catalog)}"
        ' 個已使用或取樣的原始影像檔與 '
        f"{annotation_count}"
        ' 份對應既有標註（如原資料有提供）。\n正式圖片的框與判斷來自模型推論，既有標註只是備存資料，沒有用'
        '來人工修改推論框。\ndata/class_maps.json 與各資料集原始 data.yaml '
        '保留標註類別；舊測試集為 10 類，本次模型為 11 '
        '類，應依各自對照表解讀。\nresearch/ 保留研究 JSON、候選來源清單、1010 '
        '張影像的 YOLO26n 初篩、57 張同機位的 YOLO26m 複核及時間 OCR '
        '紀錄。\n研究候選含資料增強影像及未採用素材；正式採用來源請看 '
        'data/source_catalog.csv。\nworkspace/models/pt/ '
        '含本次實際使用的 YOLO26m、研究初篩 YOLO26n 原始權重，均附 '
        'SHA256。\nworkspace/scripts/ 與 src/ 保留實際程式；各批 '
        'reproduce/ 保留該批快照。\nworkspace/environment.json '
        '保留本次 Python 與所有套件版本；專案依賴檔另作參考保留。\n\n重跑正式圖（使用相容環境）'
        '\n  python workspace/run_figures.py\n結果寫入 '
        'workspace/outputs/，不需連接攝影機、資料庫或通知服務。\n研究腳本保留原始操作路徑'
        '；部分篩選步驟需完整訓練資料集，正式圖重跑不依賴這些研究腳本。\n\n檢查資料完整性\n  '
        'python verify_bundle.py\nSHA256SUMS.txt '
        '列出每個交付檔案的雜湊；MANIFEST.json '
        '列出檔案、來源批次、模型與數量。\n\n判讀範圍\n現行程式使用影像座標、框中心、HDBSCAN '
        '各群凸包與 contains 判斷；恰在邊界不算進入。\n錐筒多邊形是程式依偵測框建立的影像管制區，'
        '不等同人工標註的完整實際作業範圍。\n前三批圖片逐張推論；第四節回放第一批的真實偵測框。模型漏檢與原'
        '始分群結果均保留。\n乘員排除是幾何啟發式；距離指標為像素，不是地面公尺距離。\n影像來自現有本機資'
        '料，包含訓練素材；這些為可核對的實作展示，不是獨立模型效能測試。\n本包是補充技術圖與完整資料，不是'
        '原始美國申請圖 1–3 '
        '的正式替換圖頁。\n'
    )
    (out / 'README.txt').write_text(readme)
    (out / 'verify_bundle.py').write_text("""from pathlib import Path
import hashlib
root = Path(__file__).resolve().parent
rows = (root / 'SHA256SUMS.txt').read_text().splitlines()
for i, line in enumerate(rows):
    expected, name = line.split('  ', 1)
    with (root / name).open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != expected:
        raise SystemExit('HASH MISMATCH: ' + name)
print('Verified', len(rows), 'files. All SHA256 hashes match.')
""")
    files = [
        dict(
            path=str(p.relative_to(out)),
            bytes=p.stat().st_size,
            sha256=sha(p),
        )
        for p in sorted(out.rglob('*'))
        if p.is_file() and p.name not in {'MANIFEST.json', 'SHA256SUMS.txt'}
    ]
    manifest = dict(
        created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        batches=batchinfo,
        formal_png_count=figure_count,
        formal_source_records=source_count,
        preserved_candidate_image_files=len(raw_catalog),
        existing_annotation_files=annotation_count,
        combined_pdf_pages=COMBINED_PAGES,
        models=models,
        files=files,
        integrity_note=(
            'MANIFEST.json describes content files. SHA256SUMS.txt '
            'additionally hashes MANIFEST.json; the checksum file does not '
            'hash itself.'
        ),
        new_evidence=(
            'Occupant six-condition checks, per-candidate execution trace and '
            'three real proximity outcomes; original elevated-view evidence '
            'is preserved.'
        ),
    )
    (out / 'MANIFEST.json').write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + '\n',
    )
    sums = [(f['sha256'], f['path']) for f in files] + [
        (sha(out / 'MANIFEST.json'), 'MANIFEST.json'),
    ]
    (out / 'SHA256SUMS.txt').write_text(
        ''.join(f"{h}  {name}\n" for h, name in sums),
    )
    subprocess.run([sys.executable, str(out / 'verify_bundle.py')], check=True)
    archive = out.parent / f"{out.name}.zip"
    print('Creating complete ZIP', flush=True)
    with zipfile.ZipFile(
        archive,
        'w',
        zipfile.ZIP_DEFLATED,
        compresslevel=3,
    ) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(
        json.dumps(
            dict(
                output=str(out),
                archive=str(archive),
                archive_bytes=archive.stat().st_size,
                formal_png_count=figure_count,
                formal_source_records=source_count,
                preserved_source_files=len(raw_catalog),
                file_count=len(sums) + 1,
                combined_pdf_pages=COMBINED_PAGES,
            ),
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == '__main__':
    main()
