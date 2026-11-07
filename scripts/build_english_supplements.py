from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from english_supplement_texts import ALGORITHM_EXPLANATION
from english_supplement_texts import ALGORITHM_README
from english_supplement_texts import BATCH_READMES
from english_supplement_texts import FIGURE_README
from english_supplement_texts import GROUND_TRUTH
from english_supplement_texts import TRANSLATIONS

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile('[\u3400-\u9fff]')


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def translate(value):
    if isinstance(value, dict):
        return {k: translate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [translate(v) for v in value]
    if isinstance(value, str) and CJK.search(value):
        if value not in TRANSLATIONS:
            raise ValueError('Missing translation: ' + value)
        return TRANSLATIONS[value]
    return value


def translate_data(root):
    for p in root.rglob('*'):
        if p.suffix == '.json':
            obj = json.loads(p.read_text())
            if CJK.search(p.read_text()):
                write_json(p, translate(obj))
        elif p.suffix == '.csv':
            content = p.read_text(encoding='utf-8-sig')
            if not CJK.search(content):
                continue
            rows = [
                [translate(cell) for cell in row]
                for row in csv.reader(io.StringIO(content))
            ]
            with p.open('w', encoding='utf-8-sig', newline='') as f:
                csv.writer(f).writerows(rows)


INTEGRITY_HELPER = (
    '"""Refresh this English edition\'s inventory after regeneration; '
    'optionally build a ZIP."""\nfrom pathlib import Path\nimport hashlib, '
    'json, sys, zipfile\nroot = Path(__file__).resolve().parents[1]\ndef '
    "sha(p):\n    with p.open('rb') as f:\n        return "
    "hashlib.file_digest(f, 'sha256').hexdigest()\nfiles = [p for p in "
    "sorted(root.rglob('*')) if p.is_file()\n         and p.name not in "
    "{'MANIFEST.json','SHA256SUMS.txt'}\n         and not any(part in "
    "{'__pycache__','.pytest_cache','recomputed'} for part in "
    'p.relative_to(root).parts)]\nmanifest = '
    "json.loads((root/'MANIFEST.json').read_text())\nmanifest['files'] = "
    '[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p))'
    " for p in files]\n(root/'MANIFEST.json').write_text(json.dumps(manifest,i"
    "ndent=2)+'\\n')\nfiles.append(root/'MANIFEST.json')\n(root/'SHA256SUMS.tx"
    "t').write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\\n' for p "
    "in files))\nif '--zip' in sys.argv:\n    target = "
    "root.with_suffix('.zip')\n    with zipfile.ZipFile(target,'w',zipfile.ZIP"
    '_DEFLATED,compresslevel=6) as z:\n        for p in '
    "[*files,root/'SHA256SUMS.txt']:\n            "
    'z.write(p,str(Path(root.name)/p.relative_to(root)))\n    print(target, '
    "target.stat().st_size, 'bytes')\nelse:\n    print('Refreshed "
    "English-edition inventory:',len(files),'files')\n"
)

TEST_RESULTS = (
    'Test conditions, comparisons and limits of the evidence\n\n1. Replay of '
    'actual-image boxes (not an independent accuracy evaluation)\nThe source '
    'set combines the earlier figures and candidate screening, deduplicated '
    'by original-image SHA256. The formal 11 images are included in the '
    'expanded 64; they must not be added as 75 images.\nEvery replay uses '
    'archived YOLO26m detections at conf=0.25 and imgsz=1280. No image '
    'inference is rerun.\nCounts refer to per-image detected person boxes, '
    'not independent subjects or unique cross-frame individuals.\n\nFormal 11 '
    'images / 27 person boxes\n  Current separate cluster hulls: 3 boxes with '
    'cone-zone warnings\n  Single hull of all cones (including noise): 9\n  '
    'Single hull of retained clustered points (controls for noise): 9\n  '
    'Current machinery/vehicle proximity with occupant exclusion: 5\n  Same '
    'proximity rules with exclusion disabled: 5\n\nExpanded 64 images / 121 '
    'person boxes (includes the formal subset)\n  Current separate cluster '
    'hulls: 3 boxes with cone-zone warnings\n  Single hull of all cones '
    '(including noise): 30\n  Single hull of retained clustered points '
    '(controls for noise): 27\n  Current machinery/vehicle proximity with '
    'occupant exclusion: 21\n  Same proximity rules with exclusion disabled: '
    '22\n\nFourteen expanded-set person boxes meet the exclusion conditions, '
    'but only one gains a proximity warning when exclusion is disabled. '
    'Fourteen must not be reported as avoided false alarms.\nThe formal '
    'subset remains 5 versus 5; no warning-count reduction is observed in '
    'that subset.\nReduced warnings could represent valid exclusions or '
    'missed warnings. Without independent truth, they cannot be called '
    'false-positive improvements.\n\n2. Actual-image false positives and '
    'false negatives\nExisting class-box annotations do not independently '
    'label zone intrusion, driver/occupant identity or the correct warning '
    'decision. The material includes training sources and selected '
    'examples.\nReal-world FP, FN, precision and recall are therefore NA, not '
    'zero. Current predictions are not used as truth.\nReplay starts from '
    'detected boxes, so it cannot measure end-to-end misses of entire people '
    'or cones by YOLO.\n\n3. Declared synthetic scenarios (mechanisms and '
    'limitations, not field accuracy)\nsynthetic_cone_25 / current_alert: '
    'n=25, TP=9, FP=0, TN=16, FN=0\nsynthetic_cone_25 / '
    'single_all_cones_hull_alert: n=25, TP=9, FP=7, TN=9, '
    'FN=0\nsynthetic_cone_25 / single_clustered_points_hull_alert: n=25, '
    'TP=9, FP=6, TN=10, FN=0\nsynthetic_proximity_core / current_alert: n=3, '
    'TP=1, FP=0, TN=2, FN=0\nsynthetic_proximity_core / '
    'no_occupant_exclusion_alert: n=3, TP=1, FP=1, TN=1, '
    'FN=0\nsynthetic_projection_ambiguity / current_alert: n=1, TP=0, FP=0, '
    'TN=0, FN=1\nsynthetic_projection_ambiguity / '
    'no_occupant_exclusion_alert: n=1, TP=1, FP=0, TN=0, FN=0\n\nThe 25 cone '
    'decision points comprise three translated/scaled variants of eight '
    'centre-rule scenarios and one outlier case. They are not 25 '
    'photographs.\nThe proximity core contains three predefined examples. One '
    'additional hypothetical exterior person projects inside an object box; '
    'current exclusion produces an FN in this ambiguity case.\nThese '
    'confusion counts are relative to the declared synthetic targets only; '
    'they are not on-site false-positive or false-negative rates.\n\n4. Code '
    'test records\nexisting_tests: 30/30 passed\nsnapshot_tests: 41/41 '
    'passed\nThe snapshot suite includes the same 30 existing tests plus 11 '
    'synthetic-contract/limitation-observation tests. Do not sum these as '
    'independent samples.\nExisting test_driver_filtering does not directly '
    'assert the exclusion outcome; other tests use mocks. test_results.csv '
    'identifies interpretation limits per test.\nCollinear and single-group '
    'tests observe current behaviour. A passing test does not mean these '
    'limitations have been fixed.\nThe English edition additionally reruns '
    'the 41-test snapshot suite and verifies identical numeric replay '
    'outputs.\n'
)


def algorithm(original, target):
    shutil.copytree(original, target, dirs_exist_ok=True)
    translate_data(target)
    (target / 'README.txt').write_text(ALGORITHM_README)
    (target / 'docs/01_algorithm_explanation.txt').write_text(
        ALGORITHM_EXPLANATION,
    )
    p = target / 'docs/02_pseudocode.txt'
    p.write_text(
        (
            'Pseudocode reflecting current behaviour; complete source: '
            'source/src/\n'
        )
        + p.read_text().split('\n', 1)[1],
    )
    (target / 'docs/03_test_results_and_effects.txt').write_text(TEST_RESULTS)
    (target / 'docs/04_ground_truth_and_followup.txt').write_text(GROUND_TRUTH)
    (target / 'reproduce/build_algorithm_evidence.py').write_text(
        INTEGRITY_HELPER,
    )
    audit = dict(
        presentation_language='en',
        original_manifest_sha256=sha(original / 'MANIFEST.json'),
        original_builder_sha256=sha(
            original / 'reproduce/build_algorithm_evidence.py',
        ),
        numeric_results_unchanged=True,
        production_source_unchanged=True,
        scope=(
            'English explanatory text and descriptive metadata; no new '
            'inference or independent field '
            'evaluation.'
        ),
    )
    for p in (original / 'results').iterdir():
        assert p.read_bytes() == (target / 'results' / p.name).read_bytes(), p
    for p in (original / 'source').rglob('*.py'):
        assert (
            p.read_bytes() == (target / p.relative_to(original)).read_bytes()
        ), p
    replay = subprocess.run(
        [
            sys.executable,
            str(
                target / 'reproduce/evaluate_algorithm_evidence.py',
            ),
        ],
        capture_output=True,
        text=True,
    )
    (target / 'tests/english_replay.log').write_text(
        replay.stdout + replay.stderr,
    )
    assert replay.returncode == 0, replay.stderr
    replayed = []
    for p in (target / 'recomputed').iterdir():
        if p.is_file():
            assert (
                p.read_bytes() == (target / 'results' / p.name).read_bytes()
            ), p
            replayed.append(dict(path=p.name, sha256=sha(p)))
    shutil.rmtree(target / 'recomputed')
    audit['english_portable_replay_exact_match'] = replayed
    tests = subprocess.run(
        [
            sys.executable,
            str(
                target / 'reproduce/run_tests.py',
            ),
        ],
        capture_output=True,
        text=True,
    )
    (target / 'tests/english_snapshot_tests.log').write_text(
        tests.stdout + tests.stderr,
    )
    assert tests.returncode == 0, tests.stdout + tests.stderr
    assert '41 passed' in tests.stdout
    audit['english_snapshot_tests_passed'] = 41
    write_json(target / 'translation_validation.json', audit)
    print('Algorithm: exact numeric replay; 41 tests passed.', flush=True)


def figures(original, target):
    shutil.copytree(original, target, dirs_exist_ok=True)
    translate_data(target)
    (target / 'README.txt').write_text(FIGURE_README)
    for batch, text in BATCH_READMES.items():
        (target / batch / 'README.txt').write_text(text)
    (target / 'reproduce').mkdir(exist_ok=True)
    shutil.copy2(
        ROOT / 'scripts/render_english_figures.py',
        target / 'reproduce/render_english_figures.py',
    )
    fonts = target / 'reproduce/fonts'
    fonts.mkdir(exist_ok=True)
    for name in ['DejaVuSans.ttf', 'DejaVuSans-Bold.ttf']:
        shutil.copy2(
            Path('/usr/share/fonts/truetype/dejavu') / name,
            fonts / name,
        )
    shutil.copy2(
        '/usr/share/doc/fonts-dejavu-core/copyright',
        fonts / 'LICENSE.txt',
    )
    (target / 'reproduce/refresh_inventory.py').write_text(INTEGRITY_HELPER)
    wrapper = (
        '"""Regenerate English overlays from archived boxes, without new '
        'inference."""\nfrom pathlib import Path\nimport subprocess, '
        'sys\nroot = Path(__file__).resolve().parents[{depth}]\nraise '
        "SystemExit(subprocess.run([sys.executable,str(root/'reproduce/render_"
        "english_figures.py'),'--package',str(root),'--batch','{batch}']).retu"
        'rncode)\n'
    )
    legacy_hashes = []
    for p in target.rglob('*.py'):
        mapping = {
            'generate_patent_figures.py': '01',
            'generate_aerial_cone_figures.py': '02',
            'generate_aerial_people_figures.py': '03',
            'generate_driver_proximity_figures.py': '04',
            'build_supplement_bundle.py': 'all',
            'run_figures.py': 'all',
        }
        if p.name in mapping:
            legacy_hashes.append(
                dict(path=str(p.relative_to(target)), original_sha256=sha(p)),
            )
            p.write_text(
                wrapper.format(
                    depth=1 if p.name == 'run_figures.py' else 2,
                    batch=mapping[p.name],
                ),
            )
        elif p.name == 'refine_same_camera.py':
            s = p.read_text()
            for cn, en in TRANSLATIONS.items():
                s = s.replace(cn, en)
            assert not CJK.search(s)
            p.write_text(s)
    (target / 'workspace/README_REPRODUCE.txt').write_text(
        (
            'English-edition reproduction\n\nFrom the extracted package, '
            'run:\n  python reproduce/render_english_figures.py --package '
            '.\nThis uses archived detections and unchanged original '
            'photographs, regenerating all 56 English PNGs and the 39-page '
            'combined PDF.\nDependencies: Pillow, Shapely, and pdfunite '
            '(Poppler). Fonts and their licence are bundled in '
            'reproduce/fonts/.\nTo regenerate one batch, add --batch 01, 02, '
            '03 or 04. Run the complete command to refresh the combined PDF '
            'and figure catalog.\nAfter regeneration, refresh the inventory '
            'using python reproduce/refresh_inventory.py; add --zip to '
            'rebuild the archive.\nProduction source, original annotations, '
            'source images and both model weights are preserved. Historical '
            'Chinese rendering wrappers were replaced for English '
            'presentation.\nLegacy inference/screening records remain '
            'historical provenance; this English renderer performs no model '
            'inference and makes no new accuracy '
            'claim.\n'
        ),
    )
    historical = target / 'data/reproduction_validation.json'
    record = json.loads(historical.read_text())
    record['validation_scope'] = (
        'Historical original-language edition only. Its render hashes do '
        'not refer to the English overlays; see '
        'translation_validation.json.'
    )
    write_json(historical, record)
    subprocess.run(
        [
            sys.executable,
            str(
                target / 'reproduce/render_english_figures.py',
            ),
            '--package',
            str(target),
        ],
        check=True,
    )
    audited_sources = 0
    for batch in BATCH_READMES:
        old = json.loads((original / batch / 'data/results.json').read_text())
        new = json.loads((target / batch / 'data/results.json').read_text())
        assert translate(old['sources']) == new['sources'], batch
        audited_sources += len(old['sources'])
    retained = []
    for folder in ['workspace', 'research']:
        for p in (original / folder).rglob('*'):
            if (
                p.is_file()
                and p.suffix.lower()
                in {'.jpg', '.jpeg', '.png', '.pt', '.yaml', '.yml'}
                or (p.is_file() and p.suffix == '.txt' and 'labels' in p.parts)
            ):
                q = target / p.relative_to(original)
                assert sha(p) == sha(q), p
                retained.append(str(p.relative_to(original)))
    for p in (original / 'workspace/src').rglob('*.py'):
        assert (
            p.read_bytes() == (target / p.relative_to(original)).read_bytes()
        ), p
    audit = dict(
        presentation_language='en',
        original_manifest_sha256=sha(original / 'MANIFEST.json'),
        original_rendering_scripts=legacy_hashes,
        source_records_verified=audited_sources,
        numeric_source_records_exact_match=True,
        unchanged_original_media_models_and_labels=len(retained),
        production_source_unchanged=True,
        png_figures=56,
        combined_pdf_pages=39,
        no_new_inference=True,
        photo_pixel_text_preserved=True,
    )
    write_json(target / 'translation_validation.json', audit)
    print(
        'Figures: source records, original media, models and labels verified.',
        flush=True,
    )


def finish(original, target):
    for p in target.rglob('*'):
        if p.is_file() and p.suffix in {
            '.txt',
            '.py',
            '.csv',
            '.json',
            '.yaml',
            '.yml',
            '.log',
            '.xml',
        }:
            assert not CJK.search(p.read_text(errors='replace')), p
    for d in list(target.rglob('__pycache__')) + list(
        target.rglob('.pytest_cache'),
    ):
        if d.exists():
            shutil.rmtree(d)
    manifest = json.loads((target / 'MANIFEST.json').read_text())
    manifest['created_at'] = datetime.now().astimezone().isoformat()
    manifest['language'] = 'en'
    manifest['original_manifest_sha256'] = sha(original / 'MANIFEST.json')
    manifest['files'] = []
    write_json(target / 'MANIFEST.json', manifest)
    script = target / 'reproduce/refresh_inventory.py'
    if not script.exists():
        script = target / 'reproduce/build_algorithm_evidence.py'
    subprocess.run([sys.executable, str(script), '--zip'], check=True)


def main():
    algo_original = ROOT / 'outputs/algorithm_evidence_20261022'
    algo_target = ROOT / 'outputs/algorithm_evidence_en_20261022'
    fig_original = ROOT / 'outputs/supplement_compilation_20261021_v2'
    fig_target = ROOT / 'outputs/figure_supplement_en_20261022'
    algorithm(algo_original, algo_target)
    figures(fig_original, fig_target)
    finish(algo_original, algo_target)
    finish(fig_original, fig_target)


if __name__ == '__main__':
    main()
