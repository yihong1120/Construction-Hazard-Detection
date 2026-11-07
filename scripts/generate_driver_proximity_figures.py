from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.generate_patent_figures import FigureExport  # noqa: E402
from scripts.generate_aerial_cone_figures import wrap  # noqa: E402
from scripts.generate_patent_figures import (  # noqa: E402
    GOLD,
    INK,
    MUTED,
    RED,
    TEAL,
    board,
    save_csv,
    sha,
    tag,
    text,
)
from src.danger_detector import DangerDetector  # noqa: E402


def p_row(p):
    return [*p['bbox'], p['confidence'], 5]


def v_row(v):
    return [*v['bbox'], v['confidence'], v['class_id']]


def occupant_conditions(p, v):
    pl, pt, pr, pb = p['bbox']
    vl, vt, vr, vb = v['bbox']
    pw, ph, vh = pr - pl, pb - pt, vb - vt
    specs = [
        (
            'D1',
            '人員框底部高於機具／車輛框底部',
            'p_bottom < v_bottom',
            pb,
            '<',
            vb,
            pb < vb,
        ),
        (
            'D2',
            '人員底部與機具／車輛底部差 ≥ 半個人高',
            'v_bottom - p_bottom >= p_height / 2',
            vb - pb,
            '>=',
            ph / 2,
            vb - pb >= ph / 2,
        ),
        (
            'D3',
            '人員框左緣不超出左側容許範圍',
            'p_left >= v_left - p_width / 2',
            pl,
            '>=',
            vl - pw / 2,
            pl >= vl - pw / 2,
        ),
        (
            'D4',
            '人員框右緣不超出右側容許範圍',
            'p_right <= v_right + p_width / 2',
            pr,
            '<=',
            vr + pw / 2,
            pr <= vr + pw / 2,
        ),
        (
            'D5',
            '人員框頂部低於機具／車輛框頂部',
            'p_top > v_top',
            pt,
            '>',
            vt,
            pt > vt,
        ),
        (
            'D6',
            '人員框高度 ≤ 機具／車輛框高度的一半',
            'p_height <= v_height / 2',
            ph,
            '<=',
            vh / 2,
            ph <= vh / 2,
        ),
    ]
    return [
        dict(
            id=i,
            description=desc,
            formula=formula,
            lhs=lhs,
            operator=op,
            rhs=rhs,
            passed=passed,
        )
        for i, desc, formula, lhs, op, rhs, passed in specs
    ]


def distance_checks(p, v):
    pl, pt, pr, pb = p['bbox']
    vl, vt, vr, vb = v['bbox']
    pw, ph = pr - pl, pb - pt
    area = (vr - vl) * (vb - vt)
    ratio = pw * ph / area
    limit = 0.05 if v['class_id'] == 8 else 0.1
    dx = min(abs(pr - vl), abs(pl - vr))
    dy = min(abs(pb - vt), abs(pt - vb))
    return dict(
        person_width=pw,
        person_height=ph,
        person_area=pw * ph,
        object_area=area,
        object_area_valid=area > 0,
        area_ratio=ratio,
        area_ratio_limit=limit,
        ratio_passed=ratio <= limit,
        dx=dx,
        dx_limit=5 * pw,
        dx_passed=dx <= 5 * pw,
        dy=dy,
        dy_limit=1.5 * ph,
        dy_passed=dy <= 1.5 * ph,
        ratio_type='機具' if v['class_id'] == 8 else '車輛',
    )


def replay(case_id, r, pid, vid, detector):
    p = next(p for p in r['people'] if p['id'] == pid)
    selected = next(v for v in r['machinery_and_vehicles'] if v['id'] == vid)
    vs = r['machinery_and_vehicles']
    rows = [v_row(v) for v in vs]
    idx = detector._build_spatial_index(rows)
    candidates = list(detector._nearby_machinery_vehicles(p_row(p), idx))
    trace = []
    for row in candidates:
        v = next(v for v in vs if v_row(v) == row)
        tests = occupant_conditions(p, v)
        occupant = all(t['passed'] for t in tests)
        assert occupant == detector._is_driver_detection(p_row(p), row)
        metrics = distance_checks(p, v)
        if occupant:
            reached = 'occupant_exclusion'
        elif not metrics['ratio_passed']:
            reached = 'area_ratio_rejected'
        else:
            reached = 'distance_comparison'
        trace.append(
            dict(
                object_id=v['id'],
                class_id=v['class_id'],
                occupant_conditions=tests,
                occupant=occupant,
                distance_metrics=metrics,
                execution_branch=reached,
                metrics_note=(
                    'Metrics are computed for this report; '
                    'production skips distance checks after occupant '
                    'exclusion or ratio rejection.'
                ),
            ),
        )
        if occupant:
            break
    flags = detector._person_proximity_flags(p_row(p), idx)
    assert flags == (
        p['occupant_excluded'],
        p['near_machinery_alert'],
        p['near_vehicle_alert'],
    )
    warnings = {}
    detector.check_proximity_violations([p_row(p)], rows, warnings)
    assert warnings.get('warning_close_to_vehicle', {}).get('count', 0) == int(
        flags[2],
    )
    assert warnings.get('warning_close_to_machinery', {}).get(
        'count',
        0,
    ) == int(flags[1])
    tests = occupant_conditions(p, selected)
    return dict(
        id=case_id,
        source_id=r['source_id'],
        person_id=pid,
        object_id=vid,
        person=p,
        object=selected,
        occupant_conditions=tests,
        distance_metrics=distance_checks(p, selected),
        candidate_evaluation_order=trace,
        production_flags=dict(
            occupant_excluded=flags[0],
            close_to_machinery=flags[1],
            close_to_vehicle=flags[2],
        ),
        single_person_proximity_warnings=warnings,
    )


def photograph(r, person_ids, object_id='M1', labels=None):
    im = Image.open(ROOT / r['source']['path']).convert('RGB')
    d = ImageDraw.Draw(im)
    obj = next(v for v in r['machinery_and_vehicles'] if v['id'] == object_id)
    d.rectangle(obj['bbox'], outline='#0769e0', width=5)
    tag(
        d,
        (obj['bbox'][0] + 9, obj['bbox'][1] + 9),
        object_id + (' 機具' if obj['class_id'] == 8 else ' 車輛'),
        '#0769e0',
        bounds=im.size,
    )
    for p in r['people']:
        if p['id'] not in person_ids:
            continue
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
        tag(
            d,
            (p['bbox'][0], p['bbox'][1] - 38),
            (labels or {}).get(p['id'], p['id']),
            color,
            bounds=im.size,
        )
    return im


def table(rows):
    width, rowheight = 1800, 91
    im = Image.new('RGB', (width, 78 + rowheight * len(rows)), 'white')
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, width, 67), fill=INK)
    for x, title in [
        (20, '條件／執行階段'),
        (790, '本例計算值（原圖座標）'),
        (1595, '判定'),
    ]:
        text(d, (x, 15), title, 27, 'white', True)
    for i, row in enumerate(rows):
        label, formula, numbers, status, color = row
        y = 72 + i * rowheight
        d.rectangle(
            (0, y, width, y + rowheight),
            fill='#f2f6f9' if i % 2 == 0 else 'white',
        )
        text(d, (20, y + 7), label, 25)
        text(d, (20, y + 47), formula, 21, MUTED)
        text(d, (790, y + 24), numbers, 26)
        tag(d, (1590, y + 22), status, color, size=24)
    return im


def document(title, subtitle, panels, notes):
    width = max(1800, *(p.width for p in panels))
    d = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    lines = [line for note in notes for line in wrap(d, note, width - 80, 26)]
    height = (
        155
        + sum(p.height for p in panels)
        + 20 * len(panels)
        + 43 * len(lines)
        + 30
    )
    out = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, width, 9), fill=INK)
    text(draw, (35, 25), title, 41, bold=True)
    text(draw, (35, 88), subtitle, 25, MUTED)
    y = 155
    for panel in panels:
        out.paste(panel, ((width - panel.width) // 2, y))
        y += panel.height + 20
    for line in lines:
        text(draw, (35, y), line, 26)
        y += 43
    return out


def flowchart():
    im = Image.new('RGB', (1800, 1690), 'white')
    d = ImageDraw.Draw(im)
    text(d, (45, 25), '人員／機具／車輛接近警示：實際判定順序', 43, bold=True)
    diagram = _FlowDiagram(d)
    node = diagram.node
    arrow = diagram.arrow
    node(
        (360, 110, 1320, 235),
        '① 取得偵測框',
        '人員 class=5；機具 class=8；車輛 class=10。',
    )
    node(
        (360, 285, 1320, 410),
        '② 每人：旗標設為 False，取得候選',
        '使用空間索引；無候選時，該人不產生接近警示。',
    )
    node(
        (360, 460, 1320, 585),
        '③ 取下一個候選：乘員六條件全部成立？',
        '任一候選符合乘員規則，就排除這位人員的全部接近警示。',
    )
    node(
        (20, 650, 315, 890),
        '符合：乘員排除',
        '忽略先前累積的接近旗標，直接處理下一位人員。',
        GOLD,
    )
    node(
        (360, 650, 1320, 775),
        '④ 不符合乘員：檢查人／物件框面積比',
        '車輛須 ≤ 0.10；機具須 ≤ 0.05；物件面積須大於零。',
    )
    node(
        (360, 850, 1320, 975),
        '⑤ 面積比通過：檢查兩項框邊指標',
        'dx ≤ 5×人寬，且 dy ≤ 1.5×人高，兩者都須通過。',
    )
    node(
        (360, 1050, 1320, 1175),
        '⑥ 兩項通過：記錄對應接近旗標',
        '機具記錄人機接近；車輛記錄人車接近；繼續處理剩餘候選。',
        RED,
    )
    node(
        (1390, 650, 1770, 1135),
        '繼續／彙總',
        '門檻未通過，或該候選檢查完成：有剩餘候選就回③；候選全部用盡才彙總。',
    )
    node(
        (360, 1290, 1320, 1425),
        '⑦ 候選用盡且未被乘員規則排除',
        '依旗標產生該人的接近機具／車輛警示，再處理下一位人員。',
    )
    for top, bottom in [
        (235, 285),
        (410, 460),
        (585, 650),
        (775, 850),
        (975, 1050),
    ]:
        arrow([(840, top), (840, bottom)])
    arrow([(360, 525), (168, 525), (168, 650)], '全部成立', (175, 555))
    arrow([(1320, 715), (1390, 715)], '未通過')
    arrow([(1320, 915), (1390, 915)], '未通過')
    arrow([(1320, 1110), (1390, 1110)])
    arrow([(1580, 650), (1580, 520), (1320, 520)], '有剩餘候選', (1415, 555))
    arrow([(1580, 1135), (1580, 1355), (1320, 1355)], '候選用盡')
    text(
        d,
        (45, 1495),
        '乘員排除僅作用於接近機具／車輛規則；不會排除錐筒管制區或其他規則。',
        27,
    )
    text(
        d,
        (45, 1550),
        '框幾何條件不等同駕駛身分辨識；dx、dy 是像素指標，未標定為地面距離。',
        27,
    )
    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        type=Path,
        default=ROOT / 'outputs/driver_proximity_20261021',
    )
    parser.add_argument(
        '--input-results',
        type=Path,
        default=ROOT / 'outputs/patent_figures_20261021/data/results.json',
    )
    args = parser.parse_args()
    out = args.output.resolve()
    for folder in ['sources', 'figures', 'data', 'reproduce']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    original = json.loads(args.input_results.read_text())
    assert (
        sha(ROOT / 'src/danger_detector.py')
        == original['code_sha256']['src/danger_detector.py']
    )
    sources = {s: original['sources'][s] for s in ['A', 'C']}
    for r in sources.values():
        source = ROOT / r['source']['path']
        assert sha(source) == r['source_sha256']
        shutil.copy2(source, out / r['original_copy'])
    detector = DangerDetector()
    cases = [
        replay('DRIVER', sources['C'], 'P2', 'M1', detector),
        replay('NEAR_VEHICLE', sources['A'], 'P3', 'M1', detector),
        replay('EXTERIOR_RATIO', sources['C'], 'P1', 'M1', detector),
    ]
    driver, near, exterior = cases
    assert driver['production_flags']['occupant_excluded']
    assert near['production_flags']['close_to_vehicle']
    assert not any(exterior['production_flags'].values())
    images = {}
    export = FigureExport(out, images).export
    export('01_decision_flow', flowchart())
    shot = photograph(sources['C'], {'P2'})
    crop = shot.crop((1250, 90, 1660, 420))
    top = board([shot, crop], 2, cell=(870, 560))
    rows = [
        (
            t['id'] + ' ' + t['description'],
            t['formula'],
            f"{t['lhs']:.1f} {t['operator']} {t['rhs']:.1f}",
            '成立' if t['passed'] else '不成立',
            TEAL if t['passed'] else RED,
        )
        for t in driver['occupant_conditions']
    ]
    export(
        '02_occupant_six_conditions',
        document(
            '操作室內人員｜六項乘員排除條件',
            '來源 C／P2 與 M1（吊車機具）；左為全景，右為操作室裁切',
            [top, table(rows)],
            [
                '六項全部成立 → 符合乘員排除；這位人員不進入接近機具／車輛警示分支。',
                '這是框幾何啟發式，並非模型辨識到「駕駛」身分類別。',
                '本例即使只看距離，dy 也未通過；因此本圖只展示排除條件成立，不據此主張降低誤警率。',
                '操作室裁切原圖座標：(1250,90)–(1660,420) px；僅裁切與等比例顯示。',
            ],
        ),
    )
    metrics = near['distance_metrics']
    top = ImageOps.contain(photograph(sources['A'], {'P3'}), (1800, 1015))
    rows = [
        (
            '排除檢查：未符合任一候選的乘員條件',
            'occupant_excluded = False',
            '乘員排除：否',
            '通過',
            TEAL,
        ),
        (
            '車輛框面積為正',
            'v_area > 0',
            f"{metrics['object_area']:.1f} > 0",
            '通過',
            TEAL,
        ),
        (
            '人員框／車輛框面積比',
            'p_area / v_area <= 0.10',
            f"{metrics['area_ratio']:.4f} <= 0.10",
            '通過',
            TEAL,
        ),
        (
            '水平框邊指標',
            'dx <= 5 * p_width',
            f"{metrics['dx']:.1f} <= {metrics['dx_limit']:.1f} px",
            '通過',
            TEAL,
        ),
        (
            '垂直框邊指標',
            'dy <= 1.5 * p_height',
            f"{metrics['dy']:.1f} <= {metrics['dy_limit']:.1f} px",
            '通過',
            TEAL,
        ),
    ]
    export(
        '03_near_vehicle_checks',
        document(
            '車外人員｜逐項通過接近車輛警示',
            '來源 A／P3 與 M1；本例只追蹤此人的判定，原畫面完整警示另存 JSON',
            [top, table(rows)],
            [
                '未被乘員排除，面積比與水平、垂直兩項指標均通過 → P3 觸發接近車輛警示。',
                '以 P3 單獨回放時，warning_close_to_vehicle.count=1；整張原圖該警示人數為 3。',
            ],
        ),
    )
    em = exterior['distance_metrics']
    rows = [
        (
            '乘員排除條件未全部成立',
            'occupant_excluded = False',
            '車外 P1：不符合乘員排除',
            '繼續',
            TEAL,
        ),
        (
            '人員框／機具框面積比',
            'p_area / v_area <= 0.05',
            f"{em['area_ratio']:.4f} > 0.05",
            '未通過',
            RED,
        ),
        (
            '水平指標：正式流程已在面積比處返回',
            'dx <= 5 * p_width',
            f"{em['dx']:.1f} <= {em['dx_limit']:.1f} px",
            '未執行',
            MUTED,
        ),
        (
            '垂直指標：正式流程已在面積比處返回',
            'dy <= 1.5 * p_height',
            f"{em['dy']:.1f} <= {em['dy_limit']:.1f} px",
            '未執行',
            MUTED,
        ),
    ]
    export(
        '04_exterior_ratio_rejection',
        document(
            '車外人員｜面積比未通過的實際案例',
            '來源 C／P1 與 M1；距離數值旁列供核對，不表示正式流程有執行到該步驟',
            [
                ImageOps.contain(
                    photograph(sources['C'], {'P1'}),
                    (1800, 1015),
                ),
                table(rows),
            ],
            [
                'P1 未被乘員排除；對 M1 的面積比 0.1728 大於機具門檻 0.05，因此此候選不發接近警示。',
                '已回放 P1 的全部候選：另兩個車輛候選也未通過面積比，本例最終不發接近警示。',
            ],
        ),
    )
    export(
        '05_same_frame_occupant_exterior',
        document(
            '同一畫面｜操作室內與車外人員的不同判定分支',
            '來源 C：黃色 P2 為操作室內人員；綠色 P1 為車外人員；藍色 M1 為吊車機具框',
            [
                ImageOps.contain(
                    photograph(sources['C'], {'P1', 'P2'}),
                    (1800, 1015),
                ),
            ],
            [
                'P2：六項乘員條件全部成立 → 排除本人接近機具／車輛警示。',
                'P1：乘員條件未全部成立 → 繼續檢查，但面積比超過機具門檻 → 本例不發接近警示。',
            ],
        ),
    )
    im = photograph(sources['A'], {'P3'})
    d = ImageDraw.Draw(im)
    p, v = near['person']['bbox'], near['object']['bbox']
    hx = min([(p[2], v[0]), (p[0], v[2])], key=_coordinate_gap)
    vy = min([(p[3], v[1]), (p[1], v[3])], key=_coordinate_gap)
    yy, xx = 685, p[2] + 30
    d.line((hx[0], yy, hx[1], yy), fill=RED, width=5)
    d.line((xx, vy[0], xx, vy[1]), fill=RED, width=5)
    tag(
        d,
        (v[2], yy + 5),
        f"dx={metrics['dx']:.1f} px",
        RED,
        size=19,
        bounds=im.size,
    )
    tag(
        d,
        (xx + 5, min(vy)),
        f"dy={metrics['dy']:.1f} px",
        RED,
        size=19,
        bounds=im.size,
    )
    detail = im.crop((0, 370, 700, 730)).resize(
        (1750, 900),
        Image.Resampling.LANCZOS,
    )
    export(
        '06_distance_geometry',
        document(
            '水平／垂直框邊指標｜原圖上的計算位置',
            '來源 A／P3 與 M1；裁切 (0,370)–(700,730) px，等比例放大 2.5 倍',
            [detail],
            [
                (
                    'dx=min(|p_right-v_left|, |p_left-v_right|)=96.7 '
                    'px；門檻=5×47.25=236.2 px。'
                ),
                (
                    'dy=min(|p_bottom-v_top|, '
                    '|p_top-v_bottom|)=154.6 px；門檻=1.5×121.49=182.2 '
                    'px。'
                ),
                '兩項取的是相對框邊差的最小值；包含投影重疊時仍可能非零，並非地面最近距離。',
            ],
        ),
    )
    simple_cards = []
    for case, sid, pid, title, note in [
        (
            driver,
            'C',
            'P2',
            '操作室內：符合乘員排除',
            '六項全部成立；排除本人接近警示。',
        ),
        (
            near,
            'A',
            'P3',
            '車外：觸發接近車輛警示',
            '非乘員；面積比、dx、dy 全部通過。',
        ),
        (
            exterior,
            'C',
            'P1',
            '車外：面積比未通過',
            '非乘員；0.1728 > 機具門檻 0.05。',
        ),
    ]:
        simple_cards.append(
            document(
                title,
                f'來源 {sid}／{pid} 與 M1',
                [
                    ImageOps.contain(
                        photograph(sources[sid], {pid}),
                        (1800, 1015),
                    ),
                ],
                [note],
            ),
        )
    export(
        '00_three_case_comparison',
        board(
            simple_cards,
            3,
            cell=(950, 850),
            heading='操作室內人員與靠近車輛的人員｜三種實際判定結果',
        ),
    )
    condrows = [
        dict(
            case=case['id'],
            source=case['source_id'],
            person=case['person_id'],
            object=case['object_id'],
            **row,
        )
        for case in cases
        for row in case['occupant_conditions']
    ]
    save_csv(out / 'data/occupant_conditions.csv', condrows)
    save_csv(
        out / 'data/proximity_metrics.csv',
        [
            dict(
                case=case['id'],
                source=case['source_id'],
                person=case['person_id'],
                object=case['object_id'],
                **case['distance_metrics'],
                **case['production_flags'],
            )
            for case in cases
        ],
    )
    metadata = dict(
        created_at=datetime.now(ZoneInfo('Asia/Taipei')).isoformat(),
        model=original['model'],
        inference=original['inference'],
        software=original['software'],
        sources=sources,
        cases=cases,
        method=(
            'Replay archived real YOLO26m detections through '
            'unchanged production proximity methods; no new YOLO '
            'inference in this script.'
        ),
        input_results_sha256=sha(args.input_results),
        validation=dict(
            production_code_hash_matches_original=True,
            archived_person_flags_match_replay=True,
            single_person_warning_counts_match=True,
        ),
        code_sha256={
            str(p.relative_to(ROOT)): sha(p)
            for p in [Path(__file__), ROOT / 'src/danger_detector.py']
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
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n',
    )
    names = [
        '00_three_case_comparison',
        '01_decision_flow',
        '02_occupant_six_conditions',
        '03_near_vehicle_checks',
        '04_exterior_ratio_rejection',
        '05_same_frame_occupant_exterior',
        '06_distance_geometry',
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
        out / 'driver_proximity.pdf',
        save_all=True,
        append_images=sheets[1:],
        resolution=212.1,
        quality=95,
    )
    (out / 'README.txt').write_text(
        '''操作室內人員與車外接近警示：判定過程補充

00 三案例對照；01 完整流程；02 乘員六條件；03 接近車輛逐項檢查；04 面積比不通過；05 同畫面操作室／車外；06 dx/dy 框邊指標。
所有圖片取自先前第一批的實際照片與 YOLO26m 框，逐人回放原專案 DangerDetector；這支腳本沒有重新執行 YOLO。
來源 C/P2/M1：吊車操作室內人員符合全部六項乘員規則；排除其接近警示。
來源 A/P3/M1：車外人員未被排除，面積比、dx、dy 全通過，單人回放產生一筆接近車輛警示；原圖全體人車接近警示人數為三。
來源 C/P1/M1：車外人員未被排除，但面積比 0.1728 大於機具門檻 0.05；其全部候選均未觸發接近警示。

乘員六條件：pb<vb、vb-pb>=ph/2、pl>=vl-pw/2、pr<=vr+pw/2、pt>vt、ph<=(vb-vt)/2。
其中人員為 p、候選物件為 v；l/t/r/b 為框左右上下邊，pw=pr-pl、ph=pb-pt。
任一空間索引候選符合全部六項，即排除這位人的人機／人車接近警示，包括先前候選累積的旗標。
未被排除才依各候選判斷面積比；車輛門檻 0.10、機具門檻 0.05。
通過面積比才比較 dx<=5pw 且 dy<=1.5ph。
dx=min(|pr-vl|,|pl-vr|)，dy=min(|pb-vt|,|pt-vb|)；均為像素框邊指標，不是地面最近距離或公尺距離。
JSON 為核對也列出未到達步驟的數值；candidate_evaluation_order.execution_branch 區分正式程式實際走到的分支。

乘員規則是幾何啟發式，並非駕駛身分辨識。C/P2 即使不排除，dy 也未通過，故不能用此案例主張排除減少誤警的效果。
乘員排除僅作用於接近機具／車輛規則，不影響錐筒禁區等其他規則。
sources/ 有兩張原始照片；data/ 有完整框、六條件 CSV、距離 CSV、原始場景警示與逐候選回放過程。
程式與原版本 SHA256 一致性已核對，reproduce/ 保留本次腳本及原始危險判斷程式。
重跑：python scripts/generate_driver_proximity_figures.py（先生成第一批資料）。
''',
    )
    for p in [
        Path(__file__),
        ROOT / 'src/danger_detector.py',
        ROOT / 'src/geometry.py',
    ]:
        shutil.copy2(p, out / 'reproduce' / p.name)
    archive = out.parent / f'{out.name}.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(out.parent))
    print(
        json.dumps(
            dict(
                output=str(out),
                png_count=len(images),
                pdf_pages=len(sheets),
                archive=str(archive),
                validation=metadata['validation'],
            ),
            ensure_ascii=False,
            indent=2,
        ),
    )


class _FlowDiagram:
    def __init__(self, draw):
        self.draw = draw

    def node(self, box, title, detail, color=INK):
        self.draw.rounded_rectangle(
            box,
            radius=16,
            outline=color,
            width=4,
            fill='#f3f7fa',
        )
        text(self.draw, (box[0] + 20, box[1] + 14), title, 29, color, True)
        lines = wrap(self.draw, detail, box[2] - box[0] - 40, 24)
        for i, line in enumerate(lines):
            text(
                self.draw,
                (box[0] + 20, box[1] + 58 + i * 34),
                line,
                24,
                MUTED,
            )

    def arrow(self, points, label=None, label_at=None):
        self.draw.line(points, fill=INK, width=4)
        x, y = points[-1]
        px, py = points[-2]
        if y > py:
            self.draw.polygon(
                [(x, y), (x - 8, y - 14), (x + 8, y - 14)],
                fill=INK,
            )
        elif x > px:
            self.draw.polygon(
                [(x, y), (x - 14, y - 8), (x - 14, y + 8)],
                fill=INK,
            )
        elif x < px:
            self.draw.polygon(
                [(x, y), (x + 14, y - 8), (x + 14, y + 8)],
                fill=INK,
            )
        else:
            self.draw.polygon(
                [(x, y), (x - 8, y + 14), (x + 8, y + 14)],
                fill=INK,
            )
        if label:
            text(
                self.draw,
                label_at or (points[0][0] + 8, points[0][1] + 7),
                label,
                23,
            )


def _coordinate_gap(pair):
    return abs(pair[0] - pair[1])


if __name__ == '__main__':
    main()
