from __future__ import annotations

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from types import TracebackType
from typing import Literal

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.async_tasks import finish_on_cancel

logger = logging.getLogger(__name__)
Compensation = Callable[[], Awaitable[None]]


class AccountTransaction:
    """Commit account changes once and compensate remote effects on failure.

    Register compensation before sending a reversible remote mutation because
    a timeout may occur after the provider accepted the change. Passwords are
    irreversible: validate and flush all local changes before resetting them.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db
        self._committed = False
        self._compensations: list[Compensation] = []
        self._irreversible: str | None = None

    def compensate_with(self, callback: Compensation) -> None:
        self._compensations.append(callback)

    def irreversible_effect(self) -> None:
        self.uncompensated_effect('password reset')

    def uncompensated_effect(self, description: str) -> None:
        self._irreversible = description

    def reconciled_effect(self) -> None:
        self._irreversible = None

    async def __aenter__(self) -> AccountTransaction:
        return self

    async def _abort(self) -> bool:
        restored = self._irreversible is None
        try:
            await self.db.rollback()
        except Exception:
            restored = False
            logger.exception(
                'Local account rollback failed; reconciliation required',
            )
        for callback in reversed(self._compensations):
            try:
                await callback()
            except Exception:
                restored = False
                logger.exception(
                    'Account compensation failed; reconciliation required',
                )
        if self._irreversible:
            logger.error(
                'Account transaction failed after %s; reconciliation '
                'required',
                self._irreversible,
            )
        return restored

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if exc_type is not None:
            restored = await finish_on_cancel(self._abort())
            if not restored and isinstance(exc, Exception):
                raise HTTPException(
                    503,
                    'account_reconciliation_required',
                ) from exc
            return False
        self._committed = False

        try:
            await finish_on_cancel(self._commit())
        except BaseException as error:
            if self._committed:
                raise
            restored = await finish_on_cancel(self._abort())
            if not restored and isinstance(error, Exception):
                raise HTTPException(
                    503,
                    'account_reconciliation_required',
                ) from error
            raise
        return False

    async def _commit(self):
        await self.db.commit()
        self._committed = True
