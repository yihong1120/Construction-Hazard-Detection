from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_patent_figures import FigureExport  # noqa: E402
from scripts.generate_aerial_cone_figures import (  # noqa: E402
    draw_stage,
    frame,
)  # noqa: E402
from scripts.generate_patent_figures import (  # noqa: E402
    MUTED,
    analyse,
    board,
    cross,
    save_csv,
    sha,
    tag,
    text,
)
from src.danger_detector import DangerDetector  # noqa: E402


def annotated(record):
    im = draw_stage(record, 'zones')
    d = ImageDraw.Draw(im)
    for p in record['people']:
        color = '#e32e40' if p['cone_alert'] else '#008875'
        cross(d, p['center'], color, radius=7)
        tag(
            d,
            (p['bbox'][0] - 30, p['bbox'][3] + 6),
            f"{p['id']} ({p['center'][0]:.1f},{p['center'][1]:.1f})",
            color=color,
            size=13,
            bounds=im.size,
        )
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'outputs/aerial_people_20261021',
    )
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ['sources', 'figures', 'data', 'reproduce']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    configpath = ROOT / 'scripts/aerial_people_sources.json'
    configs = json.loads(configpath.read_text())
    modelpath = ROOT / 'models/pt/best_yolo26m.pt'
    torch.set_num_threads(6)
    model = YOLO(str(modelpath))
    detector = DangerDetector()
    records = {
        sid: analyse(
            sid,
            cfg,
            model,
            detector,
            argparse.Namespace(imgsz=1280, conf=0.25),
            out,
        )
        for sid, cfg in configs.items()
    }
    assert [
        sum(p['cone_alert'] for p in records[s]['people']) for s in records
    ] == [0, 1, 0, 1]
    assert len(records['E4']['people']) == 2
    assert (
        records['E4']['people'][0]['cone_alert']
        and not records['E4']['people'][1]['cone_alert']
    )
    images = {}

    export = FigureExport(out, images).export

    for sid, r in records.items():
        original = Image.open(ROOT / r['source']['path']).convert('RGB')
        export(
            f'{sid}_original',
            frame(
                original,
                r['source']['description'] + '｜原圖',
                r,
                [
                    (
                        '畫面時間：'
                        f"{r['source']['capture_time_visible']}"
                        '；保留原始 640×360 像素。請以資料夾編號區分來源。 '
                    ),
                ],
            ),
        )
        lines = [
            (
                f"{p['id']}"
                ' 中心=('
                f"{p['center'][0]:.1f}"
                ','
                f"{p['center'][1]:.1f}"
                ') px；所在管制區='
                f"{'、'.join(p['inside_zones']) or '無'}"
                '；錐筒禁區警示='
                f"{('是' if p['cone_alert'] else '否')}"
                '。'
            )
            for p in r['people']
        ]
        lines += [
            '紅色為中心在區內；綠色僅表示未觸發錐筒禁區規則，其他警示另行判定。',
            '顯示圖等比例放大 2 倍；所有座標、框與多邊形運算仍使用原圖像素座標。',
        ]
        export(
            f'{sid}_center_decision',
            frame(
                annotated(r).resize((1280, 720), Image.Resampling.LANCZOS),
                r['source']['description'],
                r,
                [f"畫面時間：{r['source']['capture_time_visible']}"] + lines,
            ),
        )
    # Crops are explicitly located in the original image coordinate system.
    for sid, roi, name in [
        ('E2', (385, 180, 575, 340), 'E2_inside_detail'),
        ('E4', (145, 110, 355, 240), 'E4_inside_outside_detail'),
    ]:
        r = records[sid]
        detail = (
            annotated(r)
            .crop(roi)
            .resize(
                ((roi[2] - roi[0]) * 5, (roi[3] - roi[1]) * 5),
                Image.Resampling.LANCZOS,
            )
        )
        export(
            name,
            frame(
                detail,
                '俯瞰人員中心判斷｜局部放大',
                r,
                [
                    f"原圖裁切座標：{roi} px；等比例放大 5 倍，未重建或補畫人物細節。",
                    '中心標記及邊界完全來自同一張原圖之模型框與現行分群凸包。',
                ],
            ),
        )
    export(
        '00_same_frame_inside_outside',
        board(
            [images['E4_center_decision'], images['E4_inside_outside_detail']],
            2,
            cell=(1150, 940),
            heading='俯瞰同一畫面：管制區內／外人員中心對照',
        ),
    )
    export(
        '01_same_camera_samples',
        board(
            [images[f'{sid}_center_decision'] for sid in records],
            2,
            cell=(1150, 940),
            heading='同機位不同時間取樣｜逐張偵測，不代表同一人的進出軌跡',
        ),
    )
    cards = []
    r = records['E4']
    for stage, title, note in [
        (
            'detections',
            '01 錐筒偵測',
            '錐筒框直接來自 YOLO26m；數字為信心值。',
        ),
        (
            'centers',
            '02 錐筒框中心',
            '使用框中心座標進行分群；圖上座標四捨五入，計算保留完整精度。',
        ),
        (
            'clusters',
            '03 HDBSCAN 分群',
            'min_samples=3，min_cluster_size=2；雜訊不形成管制區。',
        ),
        (
            'zones',
            '04 管制區與人員中心',
            '每群各取凸包，逐一判斷人員框中心是否位於任一區內。',
        ),
    ]:
        card = frame(
            draw_stage(r, stage).resize((1280, 720), Image.Resampling.LANCZOS),
            title,
            r,
            [note],
        )
        cards.append(card)
        export('E4_step_' + stage, card)
    export(
        '02_overhead_processing',
        board(
            cards,
            2,
            cell=(1150, 900),
            heading='俯瞰人員案例｜偵測、中心、分群、管制區判斷',
        ),
    )
    manifest = dict(
        created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        model=dict(
            path=str(modelpath.relative_to(ROOT)),
            sha256=sha(modelpath),
        ),
        inference=dict(
            device='cpu',
            imgsz=1280,
            conf=0.25,
            iou=0.7,
            max_det=300,
            seed=0,
        ),
        software={
            p: importlib.metadata.version(p)
            for p in [
                'torch',
                'ultralytics',
                'numpy',
                'Pillow',
                'shapely',
                'scikit-learn',
            ]
        },
        sources=records,
        temporal_claim=dict(
            continuous_entry_exit_proven=False,
            tracked_identity_proven=False,
            person_ids=(
                'P numbers are local to each image, not cross-frame '
                'identities.'
            ),
            timestamps=(
                'Manually verified against text in original images; '
                'OCR screening is not authoritative.'
            ),
        ),
        selected_scenarios=dict(
            same_frame_inside='E4/P1',
            same_frame_outside='E4/P2',
            additional_inside='E2/P1',
            outside_samples=['E1/P1', 'E3/P1'],
        ),
        crop_rois=dict(E2=[385, 180, 575, 340], E4=[145, 110, 355, 240]),
        validation=dict(
            production_polygons_and_warning_counts_match=True,
            same_frame_inside_outside_assertions_passed=True,
        ),
        code_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                ROOT / 'scripts/generate_patent_figures.py',
                ROOT / 'scripts/generate_aerial_cone_figures.py',
                ROOT / 'src/geometry.py',
                ROOT / 'src/danger_detector.py',
            ]
        },
        figures=[
            dict(
                path=f'figures/{name}.png',
                sha256=sha(out / 'figures' / f'{name}.png'),
            )
            for name in images
        ],
    )
    (out / 'data/results.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
    )
    rows = [
        dict(
            source=sid,
            capture_time=r['source']['capture_time_visible'],
            person=p['id'],
            cx=p['center'][0],
            cy=p['center'][1],
            confidence=p['confidence'],
            inside_zones=';'.join(p['inside_zones']),
            cone_alert=p['cone_alert'],
            near_vehicle=p['near_vehicle_alert'],
            near_machinery=p['near_machinery_alert'],
        )
        for sid, r in records.items()
        for p in r['people']
    ]
    save_csv(out / 'data/people.csv', rows)
    names = [
        '00_same_frame_inside_outside',
        '01_same_camera_samples',
        '02_overhead_processing',
        'E2_inside_detail',
        'E4_inside_outside_detail',
    ]
    sheets = []
    for name in names:
        art = ImageOps.contain(
            images[name],
            (2440, 1660),
            Image.Resampling.LANCZOS,
        )
        sheet = Image.new('RGB', (2480, 1754), 'white')
        sheet.paste(art, ((2480 - art.width) // 2, (1754 - art.height) // 2))
        text(ImageDraw.Draw(sheet), (30, 1710), name, 20, MUTED)
        sheets.append(sheet)
    sheets[0].save(
        out / 'aerial_people.pdf',
        save_all=True,
        append_images=sheets[1:],
        resolution=212.1,
        quality=95,
    )
    (out / 'README.txt').write_text(
        '''俯瞰人員中心判斷補充（第三批）

E1 2025-05-10 08:16:03：人員中心位於錐筒管制區外。
E2 2025-05-10 08:21:23：人員中心位於錐筒管制區內。
E3 2025-05-10 09:09:40：人員中心位於錐筒管制區外。
E4 2025-05-10 09:37:21：同一畫面 P1 在區內、P2 在區外，為本批建議主圖。

這些是同機位不同時刻的獨立靜態照片，未證明跨影格為同一人，也沒有連續進出軌跡。
每張的 P1/P2 是該張局部識別編號，不是追蹤 ID。時間由原圖人工辨讀核對。
原始圖為 640×360，顯示圖與局部裁切僅等比例放大，不增加原始細節。
所有框來自本次 YOLO26m 推論，imgsz=1280、conf=0.25，模型與前兩批相同。
錐筒中心分群、每群凸包及逐人 contains 判斷均使用現有 DangerDetector，已核對頂點與警示人數。
00 為同畫面內外對照，01 為同機位不同時間取樣，02 為俯瞰逐步流程。
sources/ 保留四張原檔；data/results.json 保留所有偵測、中心、群別、多邊形、警示與雜湊。
其他類型警示保留於 JSON；綠框只表示未觸發錐筒禁區規則。

重跑：.venv/bin/python scripts/generate_aerial_people_figures.py
''',
    )
    for p in [
        Path(__file__),
        ROOT / 'scripts/generate_patent_figures.py',
        ROOT / 'scripts/generate_aerial_cone_figures.py',
        configpath,
        ROOT / 'src/geometry.py',
        ROOT / 'src/danger_detector.py',
    ]:
        shutil.copy2(p, out / 'reproduce' / p.name)
    print(
        json.dumps(
            dict(
                output=str(out),
                png_count=len(images),
                pdf_pages=len(sheets),
                scenarios=manifest['selected_scenarios'],
            ),
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == '__main__':
    main()
