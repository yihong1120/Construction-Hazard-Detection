from __future__ import annotations

import argparse
import csv
import hashlib
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
from PIL import ImageFont
from PIL import ImageOps
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.danger_detector import DangerDetector  # noqa: E402
from src.geometry import calculate_people_in_controlled_area  # noqa: E402
from src.geometry import detect_polygon_from_cones  # noqa: E402

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK = '#142d42'
MUTED = '#466075'
RED = '#e32e40'
TEAL = '#008875'
GOLD = '#b26b00'
PALETTE = ['#0769e0', '#c025a9', '#13966b', '#df6a14', '#764ac4']


def font(size=24, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size, index=3)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def text(draw, xy, value, size=24, fill=INK, bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=fill)


def tag(draw, xy, value, color=INK, size=21, bounds=None):
    f = font(size, True)
    box = draw.textbbox((0, 0), value, font=f)
    w, h = box[2] + 14, box[3] - box[1] + 12
    x, y = xy
    if bounds:
        x, y = min(max(0, x), bounds[0] - w), min(max(0, y), bounds[1] - h)
    draw.rounded_rectangle((x, y, x + w, y + h), radius=4, fill=color)
    draw.text((x + 7, y + 3 - box[1]), value, font=f, fill='white')


def center(bbox):
    return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]


def person(record, person_id):
    return next(p for p in record['people'] if p['id'] == person_id)


def pair_stats(p, obj, detector):
    b, v = p['bbox'], obj['bbox']
    pw, ph = b[2] - b[0], b[3] - b[1]
    area = (v[2] - v[0]) * (v[3] - v[1])
    ratio = pw * ph / area
    dx = min(abs(b[2] - v[0]), abs(b[0] - v[2]))
    dy = min(abs(b[3] - v[1]), abs(b[1] - v[3]))
    kind = 'machinery' if obj['class_id'] == 8 else 'vehicle'
    row = [*b, p['confidence'], 5]
    machine = [*v, obj['confidence'], obj['class_id']]
    excluded = detector._is_driver_detection(row, machine)
    near = detector._is_dangerously_close_detection(row, machine, kind)
    return dict(
        person_id=p['id'],
        object_id=obj['id'],
        object_kind=kind,
        area_ratio=ratio,
        area_ratio_limit=0.05 if kind == 'machinery' else 0.1,
        dx=dx,
        dy=dy,
        dx_limit=5 * pw,
        dy_limit=1.5 * ph,
        occupant_excluded=excluded,
        proximity_before_exclusion=near,
    )


def analyse(source_id, config, model, detector, args, out):
    path = ROOT / config['path']
    result = model.predict(
        str(path),
        device='cpu',
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.7,
        max_det=300,
        augment=False,
        half=False,
        seed=0,
        verbose=False,
    )[0]
    data = result.boxes.data.cpu().numpy().tolist()
    cones = [d for d in data if int(d[5]) == 6]
    polygons = detect_polygon_from_cones(data, detector.clusterer)
    # Fail rather than silently substitute another boundary for a degenerate
    # hull.
    assert all(p.geom_type == 'Polygon' and p.is_valid for p in polygons)
    labels = detector.clusterer.labels_.tolist() if len(cones) >= 3 else []
    cluster_labels = [
        int(k)
        for k in dict.fromkeys(labels)
        if k >= 0 and labels.count(k) >= 3
    ]
    zones = [
        dict(
            id=f'Z{i + 1}',
            cluster_label=k,
            vertices=[list(v) for v in poly.exterior.coords],
            area_px2=poly.area,
        )
        for i, (k, poly) in enumerate(
            zip(cluster_labels, polygons, strict=True),
        )
    ]
    points = [center(d) for d in cones]
    conedata = [
        dict(
            id=f'C{i + 1:02}',
            bbox=d[:4],
            confidence=d[4],
            center=c,
            cluster_label=label,
            zone_id=next(
                (z['id'] for z in zones if z['cluster_label'] == label),
                None,
            ),
        )
        for i, (d, c, label) in enumerate(
            zip(cones, points, labels, strict=True),
        )
    ]
    people = []
    for i, d in enumerate(x for x in data if int(x[5]) == 5):
        c = center(d)
        hit = [
            z['id']
            for z, poly in zip(zones, polygons, strict=True)
            if poly.contains(Point(c))
        ]
        between = []
        if not hit:
            for left in range(len(polygons)):
                for right in range(left + 1, len(polygons)):
                    a, b = polygons[left], polygons[right]
                    if a.disjoint(b) and unary_union(
                        [a, b],
                    ).convex_hull.contains(Point(c)):
                        between.append([zones[left]['id'], zones[right]['id']])
        people.append(
            dict(
                id=f'P{i + 1}',
                bbox=d[:4],
                confidence=d[4],
                center=c,
                inside_zones=hit,
                between_zone_pairs=between,
                cone_alert=bool(hit),
            ),
        )
    objects = [
        dict(id=f'M{i + 1}', class_id=int(d[5]), bbox=d[:4], confidence=d[4])
        for i, d in enumerate(x for x in data if int(x[5]) in (8, 10))
    ]
    pairs = [pair_stats(p, v, detector) for p in people for v in objects]
    for p in people:
        relevant = [r for r in pairs if r['person_id'] == p['id']]
        p['occupant_excluded'] = any(r['occupant_excluded'] for r in relevant)
        p['near_vehicle_alert'] = not p['occupant_excluded'] and any(
            r['object_kind'] == 'vehicle' and r['proximity_before_exclusion']
            for r in relevant
        )
        p['near_machinery_alert'] = not p['occupant_excluded'] and any(
            r['object_kind'] == 'machinery' and r['proximity_before_exclusion']
            for r in relevant
        )
    warnings, production_polygons, _ = detector.detect_danger(data)
    assert production_polygons == [z['vertices'] for z in zones]
    assert calculate_people_in_controlled_area(polygons, data) == sum(
        p['cone_alert'] for p in people
    )
    assert warnings.get('warning_people_in_controlled_area', {}).get(
        'count',
        0,
    ) == sum(p['cone_alert'] for p in people)
    for key, flag in [
        ('warning_close_to_vehicle', 'near_vehicle_alert'),
        ('warning_close_to_machinery', 'near_machinery_alert'),
    ]:
        assert warnings.get(key, {}).get('count', 0) == sum(
            p[flag] for p in people
        )
    original_name = f'{source_id}_{path.name}'
    shutil.copy2(path, out / 'sources' / original_name)
    return dict(
        source_id=source_id,
        source=config,
        original_copy=f'sources/{original_name}',
        source_sha256=sha(path),
        size=list(result.orig_shape[::-1]),
        detections=data,
        cones=conedata,
        zones=zones,
        people=people,
        machinery_and_vehicles=objects,
        proximity_pairs=pairs,
        warnings=warnings,
    )


def image_for(record):
    return Image.open(ROOT / record['source']['path']).convert('RGB')


def zone_overlay(im, record):
    layer = Image.new('RGBA', im.size)
    d = ImageDraw.Draw(layer)
    for i, z in enumerate(record['zones']):
        rgb = tuple(
            int(PALETTE[i % len(PALETTE)][k: k + 2], 16) for k in (1, 3, 5)
        )
        verts = [tuple(v) for v in z['vertices']]
        d.polygon(verts, fill=(*rgb, 36))
        d.line(verts, fill=(*rgb, 255), width=5)
    im = Image.alpha_composite(im.convert('RGBA'), layer).convert('RGB')
    d = ImageDraw.Draw(im)
    for i, z in enumerate(record['zones']):
        poly = Polygon(z['vertices'])
        p = poly.representative_point()
        tag(d, (p.x, p.y), z['id'], PALETTE[i % len(PALETTE)], bounds=im.size)
    return im


def cross(draw, xy, color, radius=9):
    x, y = xy
    draw.ellipse(
        (x - 5, y - 5, x + 5, y + 5),
        fill=color,
        outline='white',
        width=1,
    )
    draw.line((x - radius, y, x + radius, y), fill=color, width=2)
    draw.line((x, y - radius, x, y + radius), fill=color, width=2)


def overlay(record, stage, focus=None, objects=None):
    im = image_for(record)
    if stage in ('boundaries', 'warning', 'position'):
        im = zone_overlay(im, record)
    d = ImageDraw.Draw(im)
    if stage in ('cones', 'centers', 'clusters', 'boundaries'):
        for c in record['cones']:
            color = '#d27800'
            if stage in ('clusters', 'boundaries'):
                color = (
                    '#717d85'
                    if c['cluster_label'] < 0
                    else PALETTE[c['cluster_label'] % len(PALETTE)]
                )
            if stage in ('cones', 'centers'):
                d.rectangle(c['bbox'], outline=color, width=3)
            if stage != 'cones':
                cross(d, c['center'], color)
            label = c['id']
            if stage == 'cones':
                label += f" {c['confidence']:.2f}"
            elif stage == 'centers':
                label += f" ({c['center'][0]:.0f},{c['center'][1]:.0f})"
            elif stage == 'clusters':
                label += (
                    ' noise'
                    if c['cluster_label'] < 0
                    else f" / G{c['cluster_label'] + 1}"
                )
            # Alternate rows to reduce overlapping labels around neighbouring
            # cones.
            offset = -32 if int(c['id'][1:]) % 2 else 10
            tag(
                d,
                (c['center'][0] - 20, c['bbox'][1] + offset),
                label,
                color,
                size=17,
                bounds=im.size,
            )
    if objects:
        for obj in record['machinery_and_vehicles']:
            if obj['id'] in objects:
                d.rectangle(obj['bbox'], outline=GOLD, width=5)
                label = '機具' if obj['class_id'] == 8 else '車輛'
                tag(
                    d,
                    (obj['bbox'][0] + 8, obj['bbox'][1] + 8),
                    f"{obj['id']} {label}",
                    GOLD,
                    bounds=im.size,
                )
    if stage in ('warning', 'position', 'proximity'):
        for p in record['people']:
            if focus and p['id'] not in focus:
                continue
            color = RED if p['cone_alert'] else TEAL
            if stage == 'proximity':
                color = (
                    GOLD
                    if p['occupant_excluded']
                    else (
                        RED
                        if p['near_vehicle_alert'] or p['near_machinery_alert']
                        else TEAL
                    )
                )
            d.rectangle(p['bbox'], outline=color, width=5)
            cross(d, p['center'], color, radius=13)
            tag(
                d,
                (p['bbox'][0], p['bbox'][1] - 35),
                f"{p['id']} {p['confidence']:.2f}",
                color,
                bounds=im.size,
            )
            if focus:
                x, y = p['center']
                d.ellipse(
                    (x - 23, y - 23, x + 23, y + 23),
                    outline=color,
                    width=4,
                )
    return im


def page(im, title, subtitle, notes):
    w, h = im.size
    header, footer = 128, 46 + 37 * len(notes)
    canvas = Image.new('RGB', (w, h + header + footer), 'white')
    canvas.paste(im, (0, header))
    d = ImageDraw.Draw(canvas)
    d.rectangle((0, 0, w, 9), fill=INK)
    text(d, (32, 22), title, 37, bold=True)
    text(d, (32, 80), subtitle, 23, MUTED)
    for i, line in enumerate(notes):
        text(d, (32, h + header + 18 + 37 * i), line, 24)
    return canvas


def board(cards, columns, cell=(1020, 780), heading=None):
    gap = 24
    rows = (len(cards) + columns - 1) // columns
    top = 100 if heading else gap
    out = Image.new(
        'RGB',
        (
            columns * cell[0] + (columns + 1) * gap,
            rows * cell[1] + (rows + 1) * gap + top,
        ),
        '#edf2f6',
    )
    d = ImageDraw.Draw(out)
    if heading:
        text(d, (gap, 23), heading, 38, bold=True)
    for i, card in enumerate(cards):
        tile = ImageOps.contain(card, cell, Image.Resampling.LANCZOS)
        x = gap + i % columns * (cell[0] + gap)
        y = top + gap + i // columns * (cell[1] + gap)
        d.rectangle((x, y, x + cell[0], y + cell[1]), fill='white')
        out.paste(
            tile,
            (
                x + (cell[0] - tile.width) // 2,
                y + (cell[1] - tile.height) // 2,
            ),
        )
    return out


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        ('--output'),
        type=Path,
        default=ROOT / ('outputs/patent_figures_20261021'),
    )
    parser.add_argument(
        '--sources',
        type=Path,
        default=ROOT / 'scripts/patent_figure_sources.json',
    )
    parser.add_argument(
        '--model',
        type=Path,
        default=ROOT / 'models/pt/best_yolo26m.pt',
    )
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ('sources', 'figures', 'data', 'reproduce'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    configs = json.loads(args.sources.read_text())
    detector = DangerDetector()
    model = YOLO(str(args.model))
    assert {
        5: 'Person',
        6: 'Safety Cone',
        8: 'machinery',
        10: 'vehicle',
    }.items() <= model.names.items()
    records = {
        sid: analyse(sid, cfg, model, detector, args, out)
        for sid, cfg in configs.items()
    }
    a, b, c = (records[s] for s in 'ABC')
    pin = person(a, a['source']['inside_person'])
    pout = person(a, a['source']['outside_person'])
    pbetween = person(b, b['source']['between_person'])
    occupant = person(c, c['source']['occupant_person'])
    exterior = person(c, c['source']['exterior_person'])
    near = person(a, a['source']['proximity_person'])
    assert pin['cone_alert'] and not pout['cone_alert']
    assert not pbetween['cone_alert'] and pbetween['between_zone_pairs']
    assert len(b['zones']) == 2
    assert occupant['occupant_excluded'] and not exterior['occupant_excluded']
    assert near['near_vehicle_alert'] and not near['occupant_excluded']
    images = {}

    export = FigureExport(out, images).export

    common = (
        '來源 A｜YOLO26m｜conf='
        f'{args.conf}'
        '｜imgsz='
        f'{args.imgsz}'
        '｜原圖座標 '
        f"{a['size'][0]}"
        '×'
        f"{a['size'][1]}"
        ' px'
    )
    steps = [
        (
            '01_original',
            '01 原始現場影像',
            'original',
            ['保留原始像素與視角；僅在影像外加上標題及說明。'],
        ),
        (
            '02_cone_detection',
            '02 錐筒偵測',
            'cones',
            [
                f"模型實際偵測 {len(a['cones'])} 個錐筒；C01…為識別編號，數字為信心值。",
            ],
        ),
        (
            '03_cone_centers',
            '03 計算錐筒框中心',
            'centers',
            [
                'cx=(x1+x2)/2，cy=(y1+y2)/2；圖中座標顯示至整數，運算使用未捨入值。',
            ],
        ),
        (
            '04_hdbscan_clusters',
            '04 HDBSCAN 分群',
            'clusters',
            [
                'min_samples=3，min_cluster_size=2；灰色點代表雜訊，不參與禁區建構。',
                '同色點為同一群；各群至少 3 點才產生凸包。',
            ],
        ),
        (
            '05_zone_boundaries',
            '05 建立各群禁區邊界',
            'boundaries',
            [
                f"各群錐筒中心分別取凸包；本圖產生 {len(a['zones'])} 個禁區，不跨群連成單一禁區。",
            ],
        ),
        (
            '06_center_based_warning',
            '06 人員中心與警示判斷',
            'warning',
            [
                (
                    '紅框：中心位於禁區內；本圖錐筒禁區警示人數 = '
                    f"{sum(p['cone_alert'] for p in a['people'])}"
                    '。'
                ),
                '綠框：中心未被任一禁區 contains；這只表示未觸發錐筒禁區規則。',
            ],
        ),
    ]
    for name, title, stage, notes in steps:
        export(name, page(overlay(a, stage), title, common, notes))
    export(
        '00_processing_overview',
        board(
            [images[s[0]] for s in steps],
            2,
            heading='錐筒偵測 → 中心座標 → 分群 → 禁區邊界 → 人員警示',
        ),
    )

    export(
        '07_person_inside',
        _position_card(
            a,
            pin,
            '07 人員中心位於禁區內',
            '即使人員框跨越邊界，現行程式仍以框中心作為判斷點。',
        ),
    )
    export(
        '08_person_outside',
        _position_card(
            a,
            pout,
            '08 人員中心位於禁區外',
            '錐筒禁區警示為逐人判定；本圖其他人員仍可能觸發警示。',
        ),
    )
    export(
        '09_person_between_zones',
        _position_card(
            b,
            pbetween,
            '09 人員中心位於兩個禁區之間',
            'Z1、Z2 分別判斷；不把兩群之間的空隙自動填成禁區。',
        ),
    )
    export(
        '10_position_comparison',
        board(
            [
                images[n]
                for n in [
                    '07_person_inside',
                    '08_person_outside',
                    '09_person_between_zones',
                ]
            ],
            3,
            cell=(900, 710),
            heading='人員中心位置與錐筒禁區警示：三種實際畫面',
        ),
    )

    next(
        r
        for r in c['proximity_pairs']
        if r['person_id'] == occupant['id']
        and r['object_id'] == c['source']['occupant_object']
    )
    exterior_pair = next(
        r
        for r in c['proximity_pairs']
        if r['person_id'] == exterior['id']
        and r['object_id'] == c['source']['occupant_object']
    )
    im = overlay(
        c,
        'proximity',
        focus={occupant['id'], exterior['id']},
        objects={c['source']['occupant_object']},
    )
    export(
        '11_occupant_and_exterior',
        page(
            im,
            '11 操作室內與車外人員偵測對照',
            f"來源 C｜黃色 {occupant['id']}：符合乘員排除條件｜綠色 {exterior['id']}：不符合乘員排除條件",
            [
                f"{occupant['id']}：人員框位於機具框內，符合現行幾何規則，排除人機接近警示。",
                (
                    f"{exterior['id']}"
                    '：不排除，但人員／機具框面積比='
                    f"{exterior_pair['area_ratio']:.3f}"
                    ' > 0.05，故本例不發接近警示。'
                ),
                '乘員排除是幾何啟發式，並非獨立駕駛／乘員身分類別；僅作用於人車／人機接近規則。',
            ],
        ),
    )

    v = next(
        o
        for o in a['machinery_and_vehicles']
        if o['id'] == a['source']['proximity_object']
    )
    near_pair = next(
        r
        for r in a['proximity_pairs']
        if r['person_id'] == near['id'] and r['object_id'] == v['id']
    )
    im = overlay(a, 'proximity', focus={near['id']}, objects={v['id']})
    d = ImageDraw.Draw(im)
    pbox, vbox = near['bbox'], v['bbox']
    hx = min([(pbox[2], vbox[0]), (pbox[0], vbox[2])], key=_coordinate_gap)
    vy = min([(pbox[3], vbox[1]), (pbox[1], vbox[3])], key=_coordinate_gap)
    yy, xx = pbox[3] + 24, pbox[2] + 28
    d.line((hx[0], yy, hx[1], yy), fill=RED, width=5)
    d.line((xx, vy[0], xx, vy[1]), fill=RED, width=5)
    tag(
        d,
        (vbox[2], yy + 8),
        f"dx={near_pair['dx']:.1f} px",
        RED,
        bounds=im.size,
    )
    tag(
        d,
        (xx + 8, vy[0]),
        ('dy=' f"{near_pair['dy']:.1f}" ' px'),
        RED,
        bounds=im.size,
    )
    export(
        '12_exterior_proximity_warning',
        page(
            im,
            '12 車外人員實際觸發人車接近警示',
            f"來源 A｜{near['id']} 與 {v['id']}｜人員不符合乘員排除條件",
            [
                (
                    '框面積比 '
                    f"{near_pair['area_ratio']:.3f}"
                    ' ≤ 0.10；水平指標 '
                    f"{near_pair['dx']:.1f}"
                    ' ≤ 5×人寬='
                    f"{near_pair['dx_limit']:.1f}"
                    ' px。'
                ),
                (
                    '垂直指標 '
                    f"{near_pair['dy']:.1f}"
                    ' ≤ 1.5×人高='
                    f"{near_pair['dy_limit']:.1f}"
                    ' px；此人觸發接近車輛警示。'
                ),
                'dx、dy 依現行程式取兩組相對框邊差的最小值；這是影像像素指標，未換算實際公尺。',
            ],
        ),
    )
    # A close-up from the unmodified source makes the cabin occupant visible.
    crop = ImageOps.contain(
        image_for(c).crop((1280, 110, 1660, 440)),
        (900, 780),
        Image.Resampling.LANCZOS,
    )
    export(
        '13_cabin_detail',
        page(
            crop,
            '13 操作室局部放大',
            '來源 C｜裁切座標 (1280,110)–(1660,440) px',
            ['僅裁切及等比例放大；保留原始照片內容。'],
        ),
    )
    export(
        '14_backup_comparison',
        board(
            [
                images['11_occupant_and_exterior'],
                images['12_exterior_proximity_warning'],
            ],
            2,
            cell=(1100, 850),
            heading='備案對照：乘員幾何排除與車外人員接近警示',
        ),
    )

    manifest = dict(
        created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        model=dict(
            path=str(args.model.relative_to(ROOT)),
            sha256=sha(args.model),
            classes=model.names,
        ),
        inference=dict(
            device='cpu',
            imgsz=args.imgsz,
            conf=args.conf,
            iou=0.7,
            max_det=300,
            augment=False,
            half=False,
            seed=0,
            threads=6,
        ),
        software={
            pkg: importlib.metadata.version(pkg)
            for pkg in [
                'torch',
                'ultralytics',
                'numpy',
                'Pillow',
                'shapely',
                'scikit-learn',
            ]
        },
        code_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                ROOT / 'src/geometry.py',
                ROOT / 'src/danger_detector.py',
            ]
        },
        rules=dict(
            cluster='HDBSCAN(min_samples=3,min_cluster_size=2,copy=True)',
            hull=(
                'Each non-noise cluster with >=3 points gets its own '
                'convex hull.'
            ),
            person=(
                'Bounding-box centre; Polygon.contains; exact '
                'boundary is excluded.'
            ),
            between=(
                'Descriptive only: outside both disjoint zones and '
                'inside their combined convex hull; not a production '
                'warning rule.'
            ),
            occupant=(
                'Production bbox heuristic, not a trained driver '
                'class; only excludes proximity warnings.'
            ),
            still_images=(
                'No tracking or motion inference; raw six-column '
                'detections accepted by production DangerDetector.'
            ),
            provenance=(
                'Existing local training-directory images; not a '
                'held-out accuracy evaluation.'
            ),
        ),
        selected_cases=dict(
            inside='A/P1',
            outside='A/P2',
            between='B/P3',
            occupant='C/P2',
            exterior='C/P1',
            proximity='A/P3/M1',
        ),
        sources=records,
        figures=[
            dict(
                file=f'figures/{n}.png',
                width=im.width,
                height=im.height,
                sha256=sha(out / 'figures' / f'{n}.png'),
            )
            for n, im in images.items()
        ],
        validation=dict(
            production_polygon_coordinates_match=True,
            production_cone_warning_counts_match=True,
            production_proximity_warning_counts_match=True,
            selected_scenario_assertions_passed=True,
        ),
    )
    (out / 'data/results.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
    )
    personrows, conerows, pairrows, detrows = [], [], [], []
    for sid, record in records.items():
        for p in record['people']:
            personrows.append(
                dict(
                    source=sid,
                    person=p['id'],
                    cx=p['center'][0],
                    cy=p['center'][1],
                    confidence=p['confidence'],
                    inside_zones=';'.join(p['inside_zones']),
                    between_pairs=json.dumps(p['between_zone_pairs']),
                    cone_alert=p['cone_alert'],
                    occupant_excluded=p['occupant_excluded'],
                    near_vehicle=p['near_vehicle_alert'],
                    near_machinery=p['near_machinery_alert'],
                ),
            )
        for co in record['cones']:
            conerows.append(
                dict(
                    source=sid,
                    cone=co['id'],
                    cx=co['center'][0],
                    cy=co['center'][1],
                    confidence=co['confidence'],
                    cluster=co['cluster_label'],
                    zone=co['zone_id'],
                ),
            )
        pairrows.extend(
            dict(source=sid, **r) for r in record['proximity_pairs']
        )
        detrows.extend(
            dict(
                source=sid,
                detection=i + 1,
                x1=r[0],
                y1=r[1],
                x2=r[2],
                y2=r[3],
                confidence=r[4],
                class_id=int(r[5]),
            )
            for i, r in enumerate(record['detections'])
        )
    for filename, rows in [
        ('people.csv', personrows),
        ('cones.csv', conerows),
        ('proximity_pairs.csv', pairrows),
        ('detections.csv', detrows),
    ]:
        save_csv(out / 'data' / filename, rows)
    pdf_names = [
        '00_processing_overview',
        '10_position_comparison',
        '14_backup_comparison',
    ] + [
        n
        for n in images
        if n
        not in {
            '00_processing_overview',
            '10_position_comparison',
            '14_backup_comparison',
        }
    ]
    # Uniform A4 landscape pages; source PNGs retain their full resolution.
    pdf_pages = []
    for name in pdf_names:
        art = ImageOps.contain(
            images[name],
            (2440, 1660),
            Image.Resampling.LANCZOS,
        )
        sheet = Image.new('RGB', (2480, 1754), 'white')
        sheet.paste(art, ((2480 - art.width) // 2, (1754 - art.height) // 2))
        text(
            ImageDraw.Draw(sheet),
            (35, 1705),
            f'{name}  |  {pdf_names.index(name) + 1}/{len(pdf_names)}',
            20,
            MUTED,
        )
        pdf_pages.append(sheet)
    pdf_pages[0].save(
        out / 'patent_supplement_figures.pdf',
        save_all=True,
        append_images=pdf_pages[1:],
        resolution=212.1,
        quality=95,
    )
    readme = (
        '補充技術圖片（2026-10-21）\n\n入口：patent_supplement_figures.pdf（15 '
        '頁）；figures/ 為 15 張完整解析度 PNG。\n'
        '00_processing_overview.png：原圖、錐筒偵測、框中心、HDBSCAN '
        '分群、各群凸包及人員警示。\n07–10：實際影像之禁區內、禁區外、兩禁區之間與三情境比較。\n'
        '11–14：同框操作室／車外人員、接近車輛警示、操作室放大及備案比較。\n\n來源與可重現性\n'
        '三張照片取自既有本機訓練資料夾，未移動人物、錐筒或改造照片；來源原檔位於 sources/。\n'
        '這些是從既有資料選出的實作展示，並非獨立測試集或模型效能證明。\n以本機 best_yolo26m.pt '
        '重新推論，CPU、imgsz=1280、conf=0.25、iou=0.7。\n'
        '所有框直接來自本次模型輸出；錐筒與人員均用框中心；座標原點為原圖左上，x 向右、y 向下。\n'
        'HDBSCAN(min_samples=3,min_cluster_size=2,'
        'copy=True)；雜訊略過，每群至少三點分別取凸包。\n禁區頂點、逐人警示與人車接近計數均已與原專案 '
        'DangerDetector 的實際結果核對。\n圖上座標有顯示用四捨五入；運算使用 JSON 內完整精度。\n\n判讀界線\n'
        'contains 僅包含多邊形內部，恰在邊界的點不算進入；人員框與禁區重疊本身不是判斷條件。\n'
        '「兩禁區之間」用於圖片說明：中心在兩區共同凸包內、但不在任一區內；程式並無獨立 between 警示。\n'
        '未觸發錐筒禁區規則僅反映該項規則；不能據此認定現場安全。\n'
        '乘員排除是現有框幾何啟發式，並非模型辨識出的駕駛／乘員身分，也不能據此主張準確率。\n來源 C 的 P2 '
        '位於操作室且符合排除；P1 在車外不排除，但因人／機具框面積比超限而未發接近警示。\n來源 A 的 P3 '
        '不符合乘員排除，對 M1 車輛符合面積及水平、垂直門檻，實際發出接近車輛警示。\n'
        '乘員排除只作用於人車／人機接近警示，不會排除錐筒禁區警示。\n影像距離指標為像素框邊差，並非地面距離或已標定的公尺距離。\n'
        '靜態照片無追蹤及移動狀態；使用六欄原始偵測資料，未假造移動或追蹤結果。\n\n資料\n'
        'data/results.json：模型及來源 '
        'SHA256、軟體版本、完整框、錐筒中心、群別、禁區頂點、逐人結果及原始警示。\n'
        'data/people.csv、cones.csv、proximity_pairs.csv、detections.csv'
        '：可用試算表核對，UTF-8 BOM 編碼。\nreproduce/：本次產圖程式、來源清單及實際使用的 '
        'geometry.py、danger_detector.py 供版本比對。\n在原專案根目錄執行：\n  '
        '.venv/bin/python scripts/generate_patent_figures.py\n'
        '模型原檔依本機既有路徑載入；模型權重不重複打包。results.json 記錄本次權重雜湊。\n\n'
        '這批為技術補充圖片，不是原始美國申請圖 1–3 的正式替換圖頁。\n'
    )
    (out / 'README.txt').write_text(readme)
    for path in [
        Path(__file__),
        args.sources,
        ROOT / 'src/geometry.py',
        ROOT / 'src/danger_detector.py',
    ]:
        shutil.copy2(path, out / 'reproduce' / path.name)
    archive = out.parent / f'{out.name}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
        # Include only final deliverables, excluding exploratory candidate
        # scans.
        paths = [out / 'README.txt', out / 'patent_supplement_figures.pdf']
        for folder in ['sources', 'figures', 'data', 'reproduce']:
            paths.extend(sorted((out / folder).glob('*')))
        for path in paths:
            zipped.write(path, path.relative_to(out.parent))
    print(
        json.dumps(
            dict(
                output=str(out),
                archive=str(archive),
                png_count=len(images),
                pdf_pages=len(pdf_pages),
                scenarios=manifest['selected_cases'],
                validation=manifest['validation'],
            ),
            ensure_ascii=False,
            indent=2,
        ),
    )


class FigureExport:
    def __init__(self, out, images):
        self.out = out
        self.images = images

    def export(self, name, im):
        self.images[name] = im
        im.save(self.out / 'figures' / f'{name}.png', dpi=(200, 200))


def _position_card(record, p, title, extra):
    cx, cy = p['center']
    hits = '、'.join(p['inside_zones']) or '無'
    return page(
        overlay(record, 'position', focus={p['id']}),
        title,
        (
            '來源 '
            f"{record['source_id']}"
            '｜'
            f"{p['id']}"
            ' 中心=('
            f'{cx:.1f}'
            ', '
            f'{cy:.1f}'
            ') px｜各區分別判斷'
        ),
        [
            f"中心位於禁區：{hits}；此人錐筒禁區警示={'是' if p['cone_alert'] else '否'}。",
            extra,
        ],
    )


def _coordinate_gap(pair):
    return abs(pair[0] - pair[1])


if __name__ == '__main__':
    main()
