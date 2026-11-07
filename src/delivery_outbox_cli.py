from __future__ import annotations

import argparse
import json

from src.settings.delivery import DeliverySettings
from src.violation_delivery import ViolationOutbox


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Inspect or retry failed evidence events',
    )
    parser.add_argument('--retry', metavar='EVENT_ID')
    args = parser.parse_args()
    settings = DeliverySettings()
    found = False
    for path in settings.directory.glob('*.sqlite3'):
        box = ViolationOutbox(path, settings.max_events, settings.max_bytes)
        if args.retry:
            found = box.requeue(args.retry) or found
        else:
            for event in box.failures():
                print(json.dumps({'queue': path.name, **event}))
    if args.retry and not found:
        parser.error('Failed event was not found')


if __name__ == '__main__':
    main()
