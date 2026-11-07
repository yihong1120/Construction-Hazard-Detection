from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from shapely.geometry import Polygon

INK, MUTED, RED, TEAL, GOLD = (
    '#142d42',
    '#466075',
    '#e32e40',
    '#008875',
    '#b26b00',
)
COLORS = ['#0769e0', '#c025a9', '#13966b', '#df6a14', '#764ac4']
PACKAGE = Path('.')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def font(size=24, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    local = PACKAGE / 'reproduce/fonts' / name
    return ImageFont.truetype(
        str(
            (
                local
                if local.exists()
                else Path('/usr/share/fonts/truetype/dejavu') / name
            ),
        ),
        size,
    )


def txt(d, xy, value, size=24, color=INK, bold=False):
    d.text(xy, str(value), font=font(size, bold), fill=color)


def lines(value, width, size=24, bold=False):
    d = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    result, current = [], ''
    for word in str(value).split():
        candidate = f'{current} {word}'.strip()
        if current and d.textlength(candidate, font=font(size, bold)) > width:
            result.append(current)
            current = word
        else:
            current = candidate
    return result + ([current] if current else [])


def tag(d, xy, value, color=INK, size=19, bounds=None):
    f = font(size, True)
    b = d.textbbox((0, 0), str(value), font=f)
    w, h = b[2] + 14, b[3] - b[1] + 12
    x, y = xy
    if bounds:
        x, y = min(max(0, x), max(0, bounds[0] - w)), min(
            max(0, y),
            max(0, bounds[1] - h),
        )
    d.rounded_rectangle((x, y, x + w, y + h), radius=4, fill=color)
    d.text((x + 7, y + 3 - b[1]), str(value), font=f, fill='white')


def cross(d, c, color, r=9):
    x, y = c
    d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline='white')
    d.line((x - r, y, x + r, y), fill=color, width=2)
    d.line((x, y - r, x, y + r), fill=color, width=2)


def photo(record):
    return Image.open(
        PACKAGE / 'workspace' / record['source']['path'],
    ).convert('RGB')


def overlay(r, stage='zones', focus=None, objects=()):
    im = photo(r)
    color_map = {
        z['cluster_label']: COLORS[i % len(COLORS)]
        for i, z in enumerate(r['zones'])
    }
    if stage in {'zones', 'warning', 'position'}:
        layer = Image.new('RGBA', im.size)
        d = ImageDraw.Draw(layer)
        for z in r['zones']:
            c = color_map[z['cluster_label']]
            rgb = tuple(int(c[i: i + 2], 16) for i in (1, 3, 5))
            pts = [tuple(p) for p in z['vertices']]
            d.polygon(pts, fill=(*rgb, 40))
            d.line(pts, fill=(*rgb, 255), width=max(3, im.width // 450))
        im = Image.alpha_composite(im.convert('RGBA'), layer).convert('RGB')
    d = ImageDraw.Draw(im)
    size = max(14, min(23, im.width // 85))
    if stage in {'detections', 'centers', 'clusters', 'zones'}:
        for c in r['cones']:
            color = (
                color_map.get(c['cluster_label'], '#687a86')
                if stage in {'clusters', 'zones'}
                else '#b56100'
            )
            if stage in {'detections', 'centers'}:
                d.rectangle(
                    c['bbox'],
                    outline=color,
                    width=max(2, im.width // 700),
                )
            if stage != 'detections':
                cross(d, c['center'], color)
            if stage == 'zones':
                continue
            value = c['id']
            if stage == 'detections':
                value += f" {c['confidence']:.2f}"
            elif stage == 'centers':
                value += f" ({c['center'][0]:.0f},{c['center'][1]:.0f})"
            elif stage == 'clusters':
                value += f" / {c['zone_id'] or 'noise'}"
            off = -size - 14 if int(c['id'][1:]) % 2 else 9
            tag(
                d,
                (c['bbox'][0], c['bbox'][1] + off),
                value,
                color,
                size,
                im.size,
            )
    if stage in {'zones', 'warning', 'position'}:
        for z in r['zones']:
            pt = Polygon(z['vertices']).representative_point()
            tag(
                d,
                (pt.x, pt.y),
                z['id'],
                color_map[z['cluster_label']],
                size + 1,
                im.size,
            )
    for o in r['machinery_and_vehicles']:
        if o['id'] in objects:
            d.rectangle(o['bbox'], outline='#0769e0', width=5)
            tag(
                d,
                (o['bbox'][0] + 8, o['bbox'][1] + 8),
                o['id'] + (' machinery' if o['class_id'] == 8 else ' vehicle'),
                '#0769e0',
                size,
                im.size,
            )
    if stage in {'zones', 'warning', 'position', 'proximity'}:
        for p in r['people']:
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
            d.rectangle(
                p['bbox'],
                outline=color,
                width=max(3, im.width // 400),
            )
            tag(
                d,
                (p['bbox'][0], p['bbox'][1] - size - 15),
                p['id'] + f" {p['confidence']:.2f}",
                color,
                size,
                im.size,
            )
            if stage != 'proximity':
                cross(d, p['center'], color)
    return im


def page(im, title, r=None, notes=(), subtitle=None):
    width = max(1600, im.width)
    ts, bs = max(32, width // 52), max(23, width // 75)
    headings = lines(title, width - 70, ts, True)
    subtitle = subtitle or (
        (
            'Source '
            f"{r['source_id']}"
            ' | original '
            f"{r['size'][0]}"
            ' x '
            f"{r['size'][1]}"
            ' px | YOLO26m | conf=0.25'
        )
        if r
        else ''
    )
    subs = lines(subtitle, width - 70, bs)
    foot = [line for note in notes for line in lines(note, width - 70, bs)]
    header = 36 + len(headings) * (ts + 12) + len(subs) * (bs + 10) + 20
    out = Image.new(
        'RGB',
        (width, header + im.height + 35 + len(foot) * (bs + 13)),
        'white',
    )
    out.paste(im, ((width - im.width) // 2, header))
    d = ImageDraw.Draw(out)
    d.rectangle((0, 0, width, 8), fill=INK)
    y = 25
    for line in headings:
        txt(d, (35, y), line, ts, bold=True)
        y += ts + 12
    for line in subs:
        txt(d, (35, y), line, bs, MUTED)
        y += bs + 10
    y = header + im.height + 14
    for line in foot:
        txt(d, (35, y), line, bs)
        y += bs + 13
    return out


def board(cards, columns=2, heading='', cell=(1100, 900)):
    gap, top = 24, 120
    nrows = (len(cards) + columns - 1) // columns
    out = Image.new(
        'RGB',
        (
            columns * cell[0] + (columns + 1) * gap,
            nrows * cell[1] + (nrows + 1) * gap + top,
        ),
        '#edf2f6',
    )
    d = ImageDraw.Draw(out)
    for i, line in enumerate(lines(heading, out.width - 48, 34, True)):
        txt(d, (24, 18 + 42 * i), line, 34, bold=True)
    for i, card in enumerate(cards):
        art = ImageOps.contain(card, cell, Image.Resampling.LANCZOS)
        x, y = gap + i % columns * (
            cell[0] + gap
        ), top + gap + i // columns * (cell[1] + gap)
        d.rectangle((x, y, x + cell[0], y + cell[1]), fill='white')
        out.paste(
            art,
            (x + (cell[0] - art.width) // 2, y + (cell[1] - art.height) // 2),
        )
    return out


def table(rows):
    prepared = []
    for label, formula, values, status, color in rows:
        a, b = lines(label, 710, 25), lines(formula, 710, 21)
        c = lines(values, 700, 26)
        prepared.append(
            (
                a,
                b,
                c,
                status,
                color,
                max(92, 20 + 32 * len(a) + 28 * len(b), 30 + 34 * len(c)),
            ),
        )
    im = Image.new('RGB', (1780, 82 + sum(p[-1] for p in prepared)), 'white')
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, im.width, 70), fill=INK)
    for x, value in [
        (20, 'Condition / execution stage'),
        (760, 'Original-coordinate calculation'),
        (1510, 'Result'),
    ]:
        txt(d, (x, 20), value, 25, 'white', True)
    y = 78
    for i, (a, b, c, status, color, height) in enumerate(prepared):
        d.rectangle(
            (0, y, im.width, y + height),
            fill='#f2f6f9' if i % 2 == 0 else 'white',
        )
        yy = y + 10
        for value in a:
            txt(d, (20, yy), value, 25)
            yy += 32
        for value in b:
            txt(d, (20, yy), value, 21, MUTED)
            yy += 28
        for j, value in enumerate(c):
            txt(d, (760, y + 24 + 34 * j), value, 26)
        tag(d, (1510, y + 22), status, color, 23)
        y += height
    return im


def stack(panels):
    w = max(p.width for p in panels)
    im = Image.new(
        'RGB',
        (w, sum(p.height for p in panels) + 18 * (len(panels) - 1)),
        'white',
    )
    y = 0
    for p in panels:
        im.paste(p, ((w - p.width) // 2, y))
        y += p.height + 18
    return im


def pdf(path, images):
    pages = []
    for name, im in images:
        art = ImageOps.contain(im, (2440, 1660), Image.Resampling.LANCZOS)
        p = Image.new('RGB', (2480, 1754), 'white')
        p.paste(art, ((2480 - art.width) // 2, (1754 - art.height) // 2))
        txt(ImageDraw.Draw(p), (30, 1710), name, 20, MUTED)
        pages.append(p)
    pages[0].save(
        path,
        save_all=True,
        append_images=pages[1:],
        resolution=212.1,
        quality=95,
    )


class Batch:
    def __init__(self, folder):
        self.path = PACKAGE / folder
        self.data = json.loads((self.path / 'data/results.json').read_text())
        self.sources = self.data['sources']
        self.images = {}

    def save(self, name, im):
        self.images[name] = im
        im.save(self.path / 'figures' / f'{name}.png', dpi=(200, 200))

    def finish(self, pdfname, names):
        pdf(self.path / pdfname, [(n, self.images[n]) for n in names])
        old = self.data.get('figures', [])
        original = {
            (f.get('path') or f.get('file')): f.get(
                'original_language_sha256',
                f.get('sha256'),
            )
            for f in old
        }
        self.data['figures'] = [
            dict(
                path=f'figures/{n}.png',
                sha256=sha(self.path / 'figures' / f'{n}.png'),
                width=im.width,
                height=im.height,
                original_language_sha256=original.get(f'figures/{n}.png'),
            )
            for n, im in self.images.items()
        ]
        self.data['presentation_language'] = 'en'
        self.data['english_rendering'] = (
            'Rebuilt overlays from unchanged archived '
            'detections; source photos unchanged; no new '
            'inference.'
        )
        (self.path / 'data/results.json').write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + '\n',
        )
        print(self.path.name, len(self.images), 'English figures', flush=True)


def processing(r, stage, title):
    notes = {
        ('original'): (
            'Original image content and viewpoint retained; captions '
            'are outside the photograph.'
        ),
        ('detections'): (
            f"{len(r['cones'])}"
            ' detected cones. C identifiers and detection confidence '
            'are displayed.'
        ),
        ('centers'): (
            'cx=(x1+x2)/2; cy=(y1+y2)/2. Displayed coordinates are '
            'rounded; calculations retain full precision.'
        ),
        ('clusters'): (
            'HDBSCAN: min_samples=3, min_cluster_size=2. Colours '
            'identify groups; grey denotes noise. Each hull needs at '
            'least three points.'
        ),
        ('zones'): (
            f"{len(r['zones'])}"
            ' separate convex hulls of grouped cone centres. Gaps '
            'between different groups are not automatically filled.'
        ),
        ('warning'): (
            'Red: centre inside a zone. Cone-zone warning count='
            f"{sum(p['cone_alert'] for p in r['people'])}"
            '. Green means this cone-zone rule is not triggered; '
            'other rules may still apply.'
        ),
    }
    return page(overlay(r, stage), title, r, [notes[stage]])


def measured(r, pid, vid='M1'):
    im = overlay(r, 'proximity', {pid}, {vid})
    p = next(p['bbox'] for p in r['people'] if p['id'] == pid)
    v = next(v['bbox'] for v in r['machinery_and_vehicles'] if v['id'] == vid)
    m = next(
        m
        for m in r['proximity_pairs']
        if m['person_id'] == pid and m['object_id'] == vid
    )
    hx = min([(p[2], v[0]), (p[0], v[2])], key=_coordinate_gap)
    vy = min([(p[3], v[1]), (p[1], v[3])], key=_coordinate_gap)
    d = ImageDraw.Draw(im)
    yy, xx = 685, p[2] + 30
    d.line((hx[0], yy, hx[1], yy), fill=RED, width=5)
    d.line((xx, vy[0], xx, vy[1]), fill=RED, width=5)
    for x in hx:
        d.line((x, yy - 7, x, yy + 7), fill=RED, width=3)
    for y in vy:
        d.line((xx - 7, y, xx + 7, y), fill=RED, width=3)
    tag(d, (v[2], yy + 5), f"dx={m['dx']:.1f} px", RED, 19, im.size)
    tag(d, (xx + 10, sum(vy) / 2), f"dy={m['dy']:.1f} px", RED, 19, im.size)
    return im, m


def batch1():
    b = Batch('01_original_examples')
    a, between, c = (b.sources[k] for k in ['A', 'B', 'C'])
    stages = [
        ('01_original', 'original', '01 Original construction-site image'),
        ('02_cone_detection', 'detections', '02 Safety-cone detection'),
        ('03_cone_centers', 'centers', '03 Cone-box centre coordinates'),
        ('04_hdbscan_clusters', 'clusters', '04 HDBSCAN grouping'),
        (
            '05_zone_boundaries',
            'zones',
            '05 Per-group restricted-zone boundaries',
        ),
        (
            '06_center_based_warning',
            'warning',
            '06 Person-centre warning decisions',
        ),
    ]
    for name, stage, title in stages:
        b.save(name, processing(a, stage, title))
    b.save(
        '00_processing_overview',
        board(
            [b.images[n] for n, _, _ in stages],
            heading=(
                'Cone detection > centres > groups > boundaries > '
                'person warnings'
            ),
        ),
    )
    for name, r, pid, title, extra in [
        (
            '07_person_inside',
            a,
            'P1',
            '07 Person centre inside a restricted zone',
            (
                'A box may straddle the boundary; the centre remains '
                'the decision point.'
            ),
        ),
        (
            '08_person_outside',
            a,
            'P2',
            '08 Person centre outside the restricted zones',
            (
                'Decisions are per person. Another person in this '
                'same scene can still trigger the rule.'
            ),
        ),
        (
            '09_person_between_zones',
            between,
            'P3',
            '09 Person centre between two restricted zones',
            (
                'Z1 and Z2 are tested separately. The inter-group '
                'gap is not automatically restricted.'
            ),
        ),
    ]:
        p = next(p for p in r['people'] if p['id'] == pid)
        b.save(
            name,
            page(
                overlay(r, 'position', {pid}),
                title,
                r,
                [
                    (
                        f'{pid}'
                        ' centre=('
                        f"{p['center'][0]:.1f}"
                        ','
                        f"{p['center'][1]:.1f}"
                        ') px; containing zones='
                        f"{','.join(p['inside_zones']) or 'none'}"
                        '; cone warning='
                        f"{('yes' if p['cone_alert'] else 'no')}"
                        '.'
                    ),
                    extra,
                ],
            ),
        )
    b.save(
        '10_position_comparison',
        board(
            [
                b.images[n]
                for n in [
                    '07_person_inside',
                    '08_person_outside',
                    '09_person_between_zones',
                ]
            ],
            3,
            (
                'Person-centre locations and cone-zone warnings: '
                'three real examples'
            ),
            (950, 850),
        ),
    )
    b.save(
        '11_occupant_and_exterior',
        page(
            overlay(c, 'proximity', {'P1', 'P2'}, {'M1'}),
            '11 Cabin occupant versus exterior personnel',
            c,
            [
                (
                    'P2 meets the geometric occupant rule and is '
                    'excluded from proximity warnings.'
                ),
                (
                    'P1 is not excluded; its person/machinery area '
                    'ratio 0.173 exceeds 0.05, so this case has no '
                    'proximity warning.'
                ),
                (
                    'The rule is a geometric heuristic, not '
                    'independent driver/passenger identity '
                    'recognition; it applies only to proximity rules.'
                ),
            ],
        ),
    )
    im, m = measured(a, 'P3')
    b.save(
        '12_exterior_proximity_warning',
        page(
            im,
            '12 Exterior person triggering a vehicle-proximity warning',
            a,
            [
                (
                    'P3 is not occupant-excluded. Area ratio='
                    f"{m['area_ratio']:.3f}"
                    ' <=0.10; dx='
                    f"{m['dx']:.1f}"
                    ' <='
                    f"{m['dx_limit']:.1f}"
                    ' px; dy='
                    f"{m['dy']:.1f}"
                    ' <='
                    f"{m['dy_limit']:.1f}"
                    ' px.'
                ),
                (
                    'Both edge indicators pass. They use opposing '
                    'box-edge differences and are not calibrated '
                    'ground distances.'
                ),
            ],
        ),
    )
    b.save(
        '13_cabin_detail',
        page(
            photo(c).crop((1280, 110, 1660, 440)),
            '13 Cabin detail',
            c,
            [
                (
                    'Original-image crop: (1280,110)-(1660,440) px. '
                    'Image content is preserved.'
                ),
            ],
        ),
    )
    b.save(
        '14_backup_comparison',
        board(
            [
                b.images['11_occupant_and_exterior'],
                b.images['12_exterior_proximity_warning'],
            ],
            heading=(
                'Backup comparison: occupant geometry and exterior '
                'vehicle proximity'
            ),
        ),
    )
    names = [
        '00_processing_overview',
        '10_position_comparison',
        '14_backup_comparison',
    ] + [
        n
        for n in b.images
        if n
        not in {
            '00_processing_overview',
            '10_position_comparison',
            '14_backup_comparison',
        }
    ]
    b.finish('patent_supplement_figures.pdf', names)


def batch2():
    b = Batch('02_elevated_views')
    for sid, r in b.sources.items():
        b.save(
            f'{sid}_original',
            page(
                photo(r),
                r['source']['description'] + ' | original',
                r,
                [
                    (
                        'Existing photograph retained; no person, '
                        'cone or image content was fabricated or '
                        'moved.'
                    ),
                ],
            ),
        )
        b.save(
            f'{sid}_cone_zones',
            page(
                overlay(r, 'zones'),
                r['source']['description'] + ' | zones',
                r,
                [
                    (
                        f"{len(r['cones'])}"
                        ' cones; '
                        f"{len(r['zones'])}"
                        ' separately grouped image-space regions.'
                    ),
                    (
                        'Boundaries are hulls of detected cone-box '
                        'centres, not manually certified physical '
                        'worksite boundaries.'
                    ),
                ],
            ),
        )
        cards = []
        for stage, title in [
            ('detections', '01 Cone detections'),
            ('centers', '02 Cone centre coordinates'),
            ('clusters', '03 HDBSCAN groups'),
            ('zones', '04 Per-group convex hulls'),
        ]:
            card = processing(r, stage, title)
            cards.append(card)
            if sid == 'A':
                b.save('A_step_' + stage, card)
        b.save(
            sid + '_processing_steps',
            board(cards, heading=r['source']['description'] + ' | processing'),
        )
    b.save(
        '00_aerial_selection',
        board(
            [b.images[s + '_cone_zones'] for s in b.sources],
            heading=(
                'Elevated construction views | safety-cone '
                'restricted-zone examples'
            ),
        ),
    )
    names = ['00_aerial_selection', 'A_processing_steps'] + [
        f'{s}_{suffix}'
        for s in b.sources
        for suffix in ['original', 'cone_zones']
    ]
    b.finish('aerial_cone_zones.pdf', names)


def batch3():
    b = Batch('03_elevated_people')
    for sid, r in b.sources.items():
        stamp = r['source']['capture_time_visible']
        b.save(
            sid + '_original',
            page(
                photo(r),
                r['source']['description'] + ' | original',
                r,
                [
                    (
                        'Visible capture time: '
                        f'{stamp}'
                        '. Original 640x360 pixels preserved.'
                    ),
                ],
            ),
        )
        im = overlay(r, 'warning')
        notes = [f'Visible capture time: {stamp}.']
        for p in r['people']:
            notes.append(
                (
                    f"{p['id']}"
                    ' centre=('
                    f"{p['center'][0]:.1f}"
                    ','
                    f"{p['center'][1]:.1f}"
                    ') px; zones='
                    f"{','.join(p['inside_zones']) or 'none'}"
                    '; cone-zone warning='
                    f"{('yes' if p['cone_alert'] else 'no')}"
                    '.'
                ),
            )
        notes += [
            (
                'Display enlarged 2x; calculations and reported '
                'coordinates use original pixels. Green excludes '
                'only this cone-zone rule.'
            ),
        ]
        b.save(
            sid + '_center_decision',
            page(
                im.resize((1280, 720), Image.Resampling.LANCZOS),
                r['source']['description'],
                r,
                notes,
            ),
        )
    for sid, roi, name in [
        ('E2', (385, 180, 575, 340), 'E2_inside_detail'),
        ('E4', (145, 110, 355, 240), 'E4_inside_outside_detail'),
    ]:
        im = overlay(b.sources[sid], 'warning').crop(roi)
        im = im.resize((im.width * 5, im.height * 5), Image.Resampling.LANCZOS)
        b.save(
            name,
            page(
                im,
                'Elevated person-centre decision | detail',
                b.sources[sid],
                [
                    (
                        'Original-image crop: '
                        f'{roi}'
                        ' px; 5x enlargement. No person details were '
                        'reconstructed or added.'
                    ),
                ],
            ),
        )
    b.save(
        '00_same_frame_inside_outside',
        board(
            [
                b.images['E4_center_decision'],
                b.images['E4_inside_outside_detail'],
            ],
            heading=(
                'Elevated same-image comparison: person centres '
                'inside / outside'
            ),
        ),
    )
    b.save(
        '01_same_camera_samples',
        board(
            [b.images[s + '_center_decision'] for s in b.sources],
            heading=(
                'Same-camera time samples | independent detections, '
                'not a tracked entry/exit trajectory'
            ),
        ),
    )
    cards = []
    for stage, title in [
        ('detections', '01 Cone detections'),
        ('centers', '02 Cone centres'),
        ('clusters', '03 HDBSCAN groups'),
        ('zones', '04 Zones and person centres'),
    ]:
        card = processing(b.sources['E4'], stage, title)
        b.save('E4_step_' + stage, card)
        cards.append(card)
    b.save(
        '02_overhead_processing',
        board(
            cards,
            heading=(
                'Elevated personnel example | detection, centres, '
                'grouping and zone decisions'
            ),
        ),
    )
    b.finish(
        'aerial_people.pdf',
        [
            '00_same_frame_inside_outside',
            '01_same_camera_samples',
            '02_overhead_processing',
            'E2_inside_detail',
            'E4_inside_outside_detail',
        ],
    )


def flowchart():
    im = Image.new('RGB', (1900, 1850), 'white')
    d = ImageDraw.Draw(im)
    txt(
        d,
        (40, 20),
        'Person / machinery / vehicle proximity: decision order',
        40,
        bold=True,
    )
    diagram = _FlowDiagram(d)
    node = diagram.node
    arrow = diagram.arrow
    node(
        (380, 100, 1370, 240),
        '1. Obtain detection boxes',
        'Person class=5; machinery=8; vehicle=10.',
    )
    node(
        (380, 295, 1370, 435),
        '2. Per person: reset flags and obtain candidates',
        (
            'Use the spatial index. With no candidates, this person '
            'has no proximity warning.'
        ),
    )
    node(
        (380, 490, 1370, 650),
        '3. Next candidate: do all six occupant conditions hold?',
        (
            'A matching candidate excludes this person from all '
            'machinery/vehicle proximity warnings.'
        ),
    )
    node(
        (20, 735, 330, 1035),
        'YES: occupant exclusion',
        (
            'Discard any proximity flags set by earlier candidates '
            'and move to the next person.'
        ),
        GOLD,
    )
    node(
        (380, 735, 1370, 900),
        '4. Not excluded: check person/object area ratio',
        'Vehicle <=0.10; machinery <=0.05. Object area must be positive.',
    )
    node(
        (380, 985, 1370, 1150),
        '5. Ratio accepted: compare both edge indicators',
        'dx <=5 x person width AND dy <=1.5 x person height. Both must pass.',
    )
    node(
        (380, 1230, 1370, 1395),
        '6. Both pass: set the corresponding proximity flag',
        (
            'Record machinery or vehicle proximity. Continue through '
            'remaining candidates.'
        ),
        RED,
    )
    node(
        (1450, 735, 1870, 1360),
        'Continue / aggregate',
        (
            'After a failed threshold or completed candidate: if '
            'more candidates remain, return to step 3; aggregate '
            'only when candidates are exhausted.'
        ),
    )
    node(
        (380, 1510, 1370, 1675),
        '7. Candidates exhausted and no occupant exclusion',
        'Issue warnings from accumulated flags, then process the next person.',
    )
    for top, bottom in [
        (240, 295),
        (435, 490),
        (650, 735),
        (900, 985),
        (1150, 1230),
    ]:
        arrow([(875, top), (875, bottom)])
    arrow([(380, 570), (175, 570), (175, 735)])
    txt(d, (100, 605), 'All six hold', 22)
    for y in [820, 1070, 1310]:
        arrow([(1370, y), (1450, y)])
    arrow([(1660, 735), (1660, 570), (1370, 570)])
    txt(d, (1470, 610), 'More candidates', 22)
    arrow([(1660, 1360), (1660, 1590), (1370, 1590)])
    txt(d, (1470, 1410), 'Exhausted', 22)
    txt(
        d,
        (40, 1740),
        (
            'Occupant exclusion affects proximity only. Geometry '
            'does not establish driver identity.'
        ),
        25,
    )
    txt(
        d,
        (40, 1785),
        (
            'The two edge indicators use image pixels, not '
            'calibrated ground distances.'
        ),
        25,
    )
    return im


def batch4():
    b = Batch('04_proximity_process')
    driver, near, exterior = b.data['cases']
    a, c = b.sources['A'], b.sources['C']
    b.save('01_decision_flow', flowchart())
    shot = overlay(c, 'proximity', {'P2'}, {'M1'})
    top = board(
        [shot, shot.crop((1250, 90, 1660, 420))],
        heading='',
        cell=(870, 550),
    )
    rows = [
        (
            t['id'] + ': ' + t['description'],
            t['formula'],
            f"{t['lhs']:.1f} {t['operator']} {t['rhs']:.1f}",
            'PASS' if t['passed'] else 'FAIL',
            TEAL if t['passed'] else RED,
        )
        for t in driver['occupant_conditions']
    ]
    b.save(
        '02_occupant_six_conditions',
        page(
            stack([top, table(rows)]),
            'Cabin occupant | six exclusion conditions',
            c,
            [
                (
                    'P2 and M1 (crane machinery). All six conditions '
                    'hold: P2 is excluded from proximity warnings.'
                ),
                (
                    'This is a bounding-box heuristic, not a trained '
                    'driver-identity category.'
                ),
                (
                    'In this case dy would also fail without '
                    'exclusion; this figure does not establish a '
                    'reduction in false-alarm rates.'
                ),
                'Cabin crop: (1250,90)-(1660,420) original pixels.',
            ],
        ),
    )
    m = near['distance_metrics']
    rows = [
        (
            'Not excluded as an occupant',
            'occupant_excluded = False',
            'Occupant exclusion: no',
            'PASS',
            TEAL,
        ),
        (
            'Positive vehicle box area',
            'v_area > 0',
            f"{m['object_area']:.1f} > 0",
            'PASS',
            TEAL,
        ),
        (
            'Person / vehicle box-area ratio',
            'p_area / v_area <= 0.10',
            f"{m['area_ratio']:.4f} <= 0.10",
            'PASS',
            TEAL,
        ),
        (
            'Horizontal edge indicator',
            'dx <= 5 * p_width',
            f"{m['dx']:.1f} <= {m['dx_limit']:.1f} px",
            'PASS',
            TEAL,
        ),
        (
            'Vertical edge indicator',
            'dy <= 1.5 * p_height',
            f"{m['dy']:.1f} <= {m['dy_limit']:.1f} px",
            'PASS',
            TEAL,
        ),
    ]
    b.save(
        '03_near_vehicle_checks',
        page(
            stack([overlay(a, 'proximity', {'P3'}, {'M1'}), table(rows)]),
            'Exterior person | passing vehicle-proximity checks',
            a,
            [
                (
                    'P3 is not excluded. Ratio and both edge '
                    'indicators pass: P3 triggers a '
                    'vehicle-proximity warning.'
                ),
                (
                    'Single-person replay count=1. The complete '
                    'original scene has three vehicle-proximity '
                    'warning targets.'
                ),
            ],
        ),
    )
    m = exterior['distance_metrics']
    rows = [
        (
            'Occupant conditions do not all hold',
            'occupant_excluded = False',
            'Exterior P1: not excluded',
            'CONTINUE',
            TEAL,
        ),
        (
            'Person / machinery box-area ratio',
            'p_area / v_area <= 0.05',
            f"{m['area_ratio']:.4f} > 0.05",
            'FAIL',
            RED,
        ),
        (
            'Horizontal metric, shown for audit only',
            'dx <= 5 * p_width',
            f"{m['dx']:.1f} <= {m['dx_limit']:.1f} px",
            'SKIPPED',
            MUTED,
        ),
        (
            'Vertical metric, shown for audit only',
            'dy <= 1.5 * p_height',
            f"{m['dy']:.1f} <= {m['dy_limit']:.1f} px",
            'SKIPPED',
            MUTED,
        ),
    ]
    b.save(
        '04_exterior_ratio_rejection',
        page(
            stack([overlay(c, 'proximity', {'P1'}, {'M1'}), table(rows)]),
            'Exterior person | area-ratio rejection',
            c,
            [
                (
                    'P1 is not occupant-excluded. The 0.1728 ratio '
                    'exceeds the machinery limit 0.05, so M1 '
                    'produces no proximity warning.'
                ),
                (
                    'The other two spatial-index vehicle candidates '
                    'also fail their ratio checks. P1 has no '
                    'proximity warning overall.'
                ),
            ],
        ),
    )
    b.save(
        '05_same_frame_occupant_exterior',
        page(
            overlay(c, 'proximity', {'P1', 'P2'}, {'M1'}),
            'Same image | different branches for cabin and exterior personnel',
            c,
            [
                (
                    'P2 (gold): all six occupant conditions hold; '
                    'exclude this person from proximity warnings.'
                ),
                (
                    'P1 (green): not occupant-excluded, but '
                    'area-ratio checks fail; no proximity warning in '
                    'this example.'
                ),
            ],
        ),
    )
    im, m = measured(a, 'P3')
    im = im.crop((0, 370, 700, 730)).resize(
        (1750, 900),
        Image.Resampling.LANCZOS,
    )
    b.save(
        '06_distance_geometry',
        page(
            im,
            'Horizontal / vertical box-edge measurement locations',
            a,
            [
                'Crop (0,370)-(700,730) original pixels, enlarged 2.5x.',
                (
                    'dx=min(|p_right-v_left|,|p_left-v_right|)=96.7 '
                    'px; limit=5 x 47.25=236.2 px.'
                ),
                (
                    'dy=min(|p_bottom-v_top|,|p_top-v_bottom|)=154.6 '
                    'px; limit=1.5 x 121.49=182.2 px.'
                ),
                (
                    'These are opposing-edge differences. Projection '
                    'overlap can still yield nonzero values; they '
                    'are not nearest ground distances.'
                ),
            ],
        ),
    )
    cards = []
    for r, pid, title, note in [
        (
            c,
            'P2',
            'Cabin: occupant exclusion',
            'All six conditions pass; exclude this person.',
        ),
        (
            a,
            'P3',
            'Exterior: vehicle-proximity warning',
            'Not excluded; ratio, dx and dy all pass.',
        ),
        (
            c,
            'P1',
            'Exterior: area-ratio rejection',
            'Not excluded; 0.1728 exceeds the machinery limit 0.05.',
        ),
    ]:
        cards.append(
            page(overlay(r, 'proximity', {pid}, {'M1'}), title, r, [note]),
        )
    b.save(
        '00_three_case_comparison',
        board(
            cards,
            3,
            (
                'Cabin occupants and exterior personnel | three '
                'actual decision outcomes'
            ),
            (950, 850),
        ),
    )
    b.finish(
        'driver_proximity.pdf',
        [
            '00_three_case_comparison',
            '01_decision_flow',
            '02_occupant_six_conditions',
            '03_near_vehicle_checks',
            '04_exterior_ratio_rejection',
            '05_same_frame_occupant_exterior',
            '06_distance_geometry',
        ],
    )


def guide():
    cards = []
    for title, paragraphs in [
        (
            'Construction safety evidence | English edition',
            [
                (
                    '56 English PNGs, four figure batches, 39 '
                    'combined PDF pages. Original photos and numeric '
                    'detections are preserved.'
                ),
                (
                    'Batch 4: six occupant conditions, per-candidate '
                    'flow and three proximity outcomes.'
                ),
                (
                    'Batch 3: elevated same-image inside/outside '
                    'comparison and independent same-camera time '
                    'samples.'
                ),
                (
                    'Batch 1: cone processing, person centres, '
                    'inter-zone gaps and occupant/proximity cases.'
                ),
                (
                    'Batch 2: four elevated construction-site views '
                    'and separate cone-group hulls.'
                ),
                (
                    'data/ contains complete coordinates, groups, '
                    'boundaries, warnings and conditions. workspace/ '
                    'preserves sampled originals, annotations, model '
                    'weights and production source.'
                ),
                (
                    'English overlays use archived boxes and '
                    'decisions. No new inference or new accuracy '
                    'result is introduced.'
                ),
            ],
        ),
        (
            'Interpretation and evidence limits',
            [
                (
                    'Cone zones: box centres > HDBSCAN > individual '
                    'convex hulls > contains(person centre). Exact '
                    'boundary points are excluded.'
                ),
                (
                    'Combine batch and source IDs when identifying '
                    'A/P1/Z1. IDs are local to the corresponding '
                    'image or chapter.'
                ),
                (
                    'Same-camera samples do not establish a tracked '
                    'identity or continuous entry/exit trajectory. '
                    'The new elevated originals are 640x360; '
                    'enlargement adds no source detail.'
                ),
                (
                    'Occupant exclusion is a box-geometry heuristic, '
                    'not driver-identity recognition. Pixel edge '
                    'indicators are not calibrated metres.'
                ),
                (
                    'The training-directory and selected examples '
                    'are illustrative, not independent field '
                    'accuracy evidence.'
                ),
                (
                    'Original Chinese signs and CCTV text embedded '
                    'in photo pixels remain unchanged. Added titles, '
                    'legends, captions and documents are English.'
                ),
                (
                    'MANIFEST.json, SHA256SUMS.txt and '
                    'translation_validation.json record integrity '
                    'and unchanged numeric evidence.'
                ),
            ],
        ),
    ]:
        p = Image.new('RGB', (2480, 1754), 'white')
        d = ImageDraw.Draw(p)
        txt(d, (80, 70), title, 55, bold=True)
        y = 205
        for paragraph in paragraphs:
            for value in lines(paragraph, 2310, 33):
                txt(d, (85, y), value, 33, MUTED)
                y += 48
            y += 34
        assert y < 1700
        cards.append((title, p))
    pdf(PACKAGE / 'index/reading_guide.pdf', cards)
    subprocess.run(
        [
            'pdfunite',
            str(PACKAGE / 'index/reading_guide.pdf'),
            str(PACKAGE / '04_proximity_process/driver_proximity.pdf'),
            str(PACKAGE / '03_elevated_people/aerial_people.pdf'),
            str(
                PACKAGE / '01_original_examples/patent_supplement_figures.pdf',
            ),
            str(PACKAGE / '02_elevated_views/aerial_cone_zones.pdf'),
            str(PACKAGE / '00_combined_figures.pdf'),
        ],
        check=True,
    )
    rows = []
    for batch in [
        '01_original_examples',
        '02_elevated_views',
        '03_elevated_people',
        '04_proximity_process',
    ]:
        for path in sorted((PACKAGE / batch / 'figures').glob('*.png')):
            rows.append(
                dict(
                    batch=batch,
                    figure=path.stem,
                    path=str(path.relative_to(PACKAGE)),
                    sha256=sha(path),
                ),
            )
    with (PACKAGE / 'index/figure_catalog.csv').open(
        'w',
        encoding='utf-8-sig',
        newline='',
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len(rows) == 56


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--package',
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        '--batch',
        default='all',
        choices=['all', '01', '02', '03', '04'],
    )
    args = parser.parse_args()
    global PACKAGE
    PACKAGE = args.package.resolve()
    for key, function in [
        ('01', batch1),
        ('02', batch2),
        ('03', batch3),
        ('04', batch4),
    ]:
        if args.batch in {'all', key}:
            function()
    if args.batch == 'all':
        guide()


class _FlowDiagram:
    def __init__(self, draw):
        self.draw = draw

    def node(self, box, title, detail, color=INK):
        self.draw.rounded_rectangle(
            box,
            radius=14,
            outline=color,
            width=4,
            fill='#f3f7fa',
        )
        y = box[1] + 12
        for value in lines(title, box[2] - box[0] - 32, 27, True):
            txt(self.draw, (box[0] + 16, y), value, 27, color, True)
            y += 33
        for value in lines(detail, box[2] - box[0] - 32, 22):
            txt(self.draw, (box[0] + 16, y + 5), value, 22, MUTED)
            y += 29

    def arrow(self, pts):
        self.draw.line(pts, fill=INK, width=4)
        x, y = pts[-1]
        px, py = pts[-2]
        if y > py:
            tip = [(x, y), (x - 8, y - 14), (x + 8, y - 14)]
        elif x < px:
            tip = [(x, y), (x + 14, y - 8), (x + 14, y + 8)]
        else:
            tip = [(x, y), (x - 14, y - 8), (x - 14, y + 8)]
        self.draw.polygon(tip, fill=INK)


def _coordinate_gap(pair):
    return abs(pair[0] - pair[1])


if __name__ == '__main__':
    main()
