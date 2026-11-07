from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.settings.delivery import DeliverySettings
from src.violation_delivery import ViolationOutbox


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Inspect or recover archived evidence events',
    )
    parser.add_argument('database', type=Path)
    parser.add_argument('action', choices=('list', 'show', 'requeue', 'purge'))
    parser.add_argument('--event-id')
    parser.add_argument('--older-than-days', type=float, default=7)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error('Outbox database does not exist')
    if args.action in ('show', 'requeue') and not args.event_id:
        parser.error('--event-id is required')
    if args.older_than_days < 0:
        parser.error('--older-than-days must be nonnegative')
    settings = DeliverySettings()
    box = ViolationOutbox(
        args.database,
        settings.max_events,
        settings.max_bytes,
        settings.archive,
    )
    result: object
    if args.action == 'list':
        result = box.failures()
    elif args.action == 'show':
        result = box.inspect(args.event_id)
    elif args.action == 'requeue':
        result = {'requeued': box.requeue(args.event_id)}
    else:
        result = {
            'purged': box.purge_failures(
                time.time() - args.older_than_days * 86400,
            ),
        }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
