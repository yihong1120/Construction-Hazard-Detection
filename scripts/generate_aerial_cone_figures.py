from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import sys
import zipfile
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
from scripts.generate_patent_figures import (  # noqa: E402
    INK,
    MUTED,
    PALETTE,
    analyse,
    board,
    cross,
    font,
    save_csv,
    sha,
    tag,
    text,
)
from src.danger_detector import DangerDetector  # noqa: E402


def wrap(draw, value, max_width, size):
    lines, current = [], ''
    for char in value:
        if (
            draw.textlength(current + char, font=font(size)) > max_width
            and current
        ):
            lines.append(current)
            current = char
        else:
            current += char
    return lines + ([current] if current else [])


def frame(im, title, record, notes):
    width = max(1400, im.width)
    size = max(24, round(width / 67))
    scratch = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    lines = [
        line
        for note in notes
        for line in wrap(scratch, note, width - 70, size)
    ]
    header, line_height = 135, size + 13
    out = Image.new(
        'RGB',
        (width, im.height + header + len(lines) * line_height + 30),
        'white',
    )
    out.paste(im, ((width - im.width) // 2, header))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, width, 8), fill=INK)
    text(d, (30, 23), title, max(35, round(width / 48)), bold=True)
    text(
        d,
        (30, 83),
        (
            '來源 '
            f"{record['source_id']}"
            '｜原圖 '
            f"{record['size'][0]}"
            '×'
            f"{record['size'][1]}"
            ' px｜YOLO26m｜conf=0.25'
        ),
        size,
        MUTED,
    )
    for i, line in enumerate(lines):
        text(d, (30, header + im.height + 14 + i * line_height), line, size)
    return out


def draw_stage(record, stage):
    im = Image.open(ROOT / record['source']['path']).convert('RGB')
    colors = {
        z['cluster_label']: PALETTE[i % len(PALETTE)]
        for i, z in enumerate(record['zones'])
    }
    if stage == 'zones':
        layer = Image.new('RGBA', im.size)
        d = ImageDraw.Draw(layer)
        for z in record['zones']:
            color = colors[z['cluster_label']]
            rgb = tuple(int(color[i: i + 2], 16) for i in (1, 3, 5))
            pts = [tuple(p) for p in z['vertices']]
            d.polygon(pts, fill=(*rgb, 48))
            d.line(pts, fill=(*rgb, 255), width=max(3, im.width // 430))
        im = Image.alpha_composite(im.convert('RGBA'), layer).convert('RGB')
    d = ImageDraw.Draw(im)
    labelsize = max(15, round(im.width / 90))
    for c in record['cones'] if stage != 'original' else []:
        color = (
            colors.get(c['cluster_label'], '#687a86')
            if stage in ('clusters', 'zones')
            else '#b56100'
        )
        if stage in ('detections', 'centers'):
            d.rectangle(
                c['bbox'],
                outline=color,
                width=max(2, im.width // 700),
            )
        if stage != 'detections':
            cross(d, c['center'], color, radius=max(7, im.width // 200))
        if stage == 'detections':
            label = f"{c['id']} {c['confidence']:.2f}"
        elif stage == 'centers':
            label = f"{c['id']} ({c['center'][0]:.0f},{c['center'][1]:.0f})"
        elif stage == 'clusters':
            label = f"{c['id']} / {c['zone_id'] or 'noise'}"
        else:
            continue
        tag(
            d,
            (c['bbox'][0], c['bbox'][1] - labelsize - 14),
            label,
            color=color,
            size=labelsize,
            bounds=im.size,
        )
    if stage == 'zones':
        from shapely.geometry import Polygon

        for z in record['zones']:
            point = Polygon(z['vertices']).representative_point()
            tag(
                d,
                (point.x, point.y),
                z['id'],
                colors[z['cluster_label']],
                size=labelsize + 2,
                bounds=im.size,
            )
        for p in record['people']:
            color = '#e32e40' if p['cone_alert'] else '#008875'
            d.rectangle(
                p['bbox'],
                outline=color,
                width=max(2, im.width // 650),
            )
            cross(d, p['center'], color)
            tag(
                d,
                (p['bbox'][0], p['bbox'][1] - labelsize - 12),
                p['id'],
                color,
                labelsize,
                bounds=im.size,
            )
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'outputs/aerial_cone_zones_20261021',
    )
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ['sources', 'figures', 'data', 'reproduce']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    configpath = ROOT / 'scripts/aerial_cone_sources.json'
    configs = json.loads(configpath.read_text())
    modelpath = ROOT / 'models/pt/best_yolo26m.pt'
    torch.set_num_threads(6)
    model = YOLO(str(modelpath))
    detector = DangerDetector()
    inference_args = argparse.Namespace(imgsz=1280, conf=0.25)
    records = {
        sid: analyse(sid, config, model, detector, inference_args, out)
        for sid, config in configs.items()
    }
    assert all(r['zones'] for r in records.values())
    images = {}

    export = FigureExport(out, images).export

    for sid, r in records.items():
        export(
            f'{sid}_original',
            frame(
                draw_stage(r, 'original'),
                r['source']['description'] + '｜原圖',
                r,
                ['保留既有實際影像；未移動人物或錐筒，未增加或補畫照片內容。'],
            ),
        )
        zones = frame(
            draw_stage(r, 'zones'),
            r['source']['description'] + '｜管制區',
            r,
            [
                (
                    '模型偵測 '
                    f"{len(r['cones'])}"
                    ' 個錐筒，HDBSCAN 分群後形成 '
                    f"{len(r['zones'])}"
                    ' 個影像管制區。'
                ),
                '各色邊界為該群錐筒框中心之凸包；顯示現行程式輸出，非人工指定的作業區邊界。',
            ],
        )
        export(f'{sid}_cone_zones', zones)
        cards = []
        for stage, title, note in [
            (
                'detections',
                '01 錐筒偵測',
                '方框為模型輸出；C 編號後數字為偵測信心值。',
            ),
            (
                'centers',
                '02 錐筒框中心座標',
                '中心=(框左上+框右下)/2；顯示座標四捨五入，運算保留完整精度。',
            ),
            (
                'clusters',
                '03 HDBSCAN 分群',
                'min_samples=3，min_cluster_size=2；同色為同群，雜訊不納入管制區。',
            ),
            (
                'zones',
                '04 各群凸包管制區',
                '每群至少三點各別取凸包；保持各群邊界，不跨群連成同一區域。',
            ),
        ]:
            card = frame(draw_stage(r, stage), title, r, [note])
            cards.append(card)
            if sid == 'A':
                export(f'A_step_{stage}', card)
        export(
            f'{sid}_processing_steps',
            board(
                cards,
                2,
                cell=(1200, 860),
                heading=r['source']['description'] + '｜逐步處理',
            ),
        )
        print(
            sid,
            ('cones'),
            len(r[('cones')]),
            ('zones'),
            len(r[('zones')]),
            flush=True,
        )
    export(
        '00_aerial_selection',
        board(
            [images[f'{sid}_cone_zones'] for sid in records],
            2,
            cell=(1250, 970),
            heading='工地俯視影像｜安全錐管制區候選圖',
        ),
    )
    rows = [
        dict(
            source=sid,
            cone=c['id'],
            cx=c['center'][0],
            cy=c['center'][1],
            confidence=c['confidence'],
            cluster_label=c['cluster_label'],
            zone=c['zone_id'],
        )
        for sid, r in records.items()
        for c in r['cones']
    ]
    save_csv(out / 'data/cones.csv', rows)
    manifest = dict(
        created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        model=str(modelpath.relative_to(ROOT)),
        model_sha256=sha(modelpath),
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
                'scikit-learn',
                'shapely',
                'Pillow',
            ]
        },
        notes=[
            (
                'Local real elevated/oblique construction imagery; '
                'not new aerial photography.'
            ),
            (
                'Existing training directory; illustrative examples, '
                'not held-out evaluation.'
            ),
            (
                'Detection boxes and polygons match production '
                'DangerDetector outputs.'
            ),
            (
                'No person centre lies inside these selected cone '
                'polygons; no inside-alert example is claimed.'
            ),
            (
                'Source pixels are preserved; small images are '
                'padded for captions without upsampling.'
            ),
            (
                'Cone IDs, centres, cluster labels and zone vertices '
                'use source-image pixel coordinates.'
            ),
        ],
        sources=records,
        code_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                ROOT / 'scripts/generate_patent_figures.py',
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
    pdf_names = ['00_aerial_selection', 'A_processing_steps'] + [
        f'{sid}_{suffix}'
        for sid in records
        for suffix in ['original', 'cone_zones']
    ]
    sheets = []
    for name in pdf_names:
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
        out / 'aerial_cone_zones.pdf',
        save_all=True,
        append_images=sheets[1:],
        resolution=212.1,
        quality=95,
    )
    (out / 'README.txt').write_text(
        '''工地俯瞰安全錐管制區補充圖片

00_aerial_selection.png 為四張候選管制區圖總覽。
A：橋梁工區俯瞰，原圖 2532×1170；兩組影像管制區，建議作為逐步流程示例。
B：工地全景，原圖 1366×768；多組錐筒與作業動線。
C：模板施工區俯瞰，原圖 1366×768；可見施工人員及錐筒配置。
D：開挖工區高角度俯視，原圖 720×576；俯視角度最陡。

各例包含原圖、管制區圖、四步驟流程圖；A 另附各步驟獨立 PNG。
這些均是既有本機實際工地影像，部分含原攝影／監視系統介面，沒有合成人員或補畫錐筒。
所有來源原檔在 sources/。小圖以留白容納標題，未插值放大原影像。
model=best_yolo26m.pt, CPU, imgsz=1280, conf=0.25, iou=0.7。
使用現有 DangerDetector 的 HDBSCAN 與每群凸包規則，已逐一核對頂點及警示數。
本批各圖均無人員中心進入所產生的錐筒凸包，未宣稱有禁區內人員警示案例。
影像中的實際施工範圍與程式依錐筒中心產生的多邊形可能不同。
未將車道、施工圍籬或兩群之間的空隙人工改成管制區；模型漏檢與分群輸出均保留。
data/results.json 與 cones.csv 提供來源、模型雜湊、偵測框、中心、群別與管制區頂點。
資料來自本機訓練素材資料夾，僅用於實作展示，非獨立效能測試。

重跑（於原專案根目錄）：
  .venv/bin/python scripts/generate_aerial_cone_figures.py
模型權重依本機現有路徑載入；reproduce/ 收錄本次腳本與使用的程式快照。
''',
    )
    for path in [
        Path(__file__),
        ROOT / 'scripts/generate_patent_figures.py',
        configpath,
        ROOT / 'src/geometry.py',
        ROOT / 'src/danger_detector.py',
    ]:
        shutil.copy2(path, out / 'reproduce' / path.name)
    archive = out.parent / f'{out.name}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(out.parent))
    print(
        json.dumps(
            dict(
                output=str(out),
                archive=str(archive),
                png_count=len(images),
                pdf_pages=len(sheets),
            ),
            ensure_ascii=False,
            indent=2,
        ),
    )


if __name__ == '__main__':
    main()
