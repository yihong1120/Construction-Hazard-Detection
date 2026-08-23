from __future__ import annotations

import asyncio

from src.model_fetcher import ModelFetcher


class ModelTools:
    """Expose only real model repository operations to MCP handlers."""

    def __init__(self, fetcher: ModelFetcher | None = None) -> None:
        """Perform init.

        Args:
            fetcher: Value used by this callable.
        """
        self._fetcher = fetcher or ModelFetcher()

    async def sync_model(
        self,
        model_name: str,
        *,
        force_download: bool = False,
    ) -> dict:
        """Fetch a newer model and atomically install it when one exists."""
        last_update_time = self._fetcher.get_last_update_time(model_name)
        updated = await asyncio.to_thread(
            self._fetcher.request_new_model,
            model_name,
            last_update_time,
            force_download=force_download,
        )
        model_path = self._fetcher.local_dir / f"best_{model_name}.pt"
        return {
            'success': updated or model_path.is_file(),
            'updated': updated,
            'model_name': model_name,
            'model_path': str(model_path) if model_path.is_file() else None,
        }

    async def list_available_models(self) -> dict:
        """List the model identifiers configured for the download client."""
        models = list(self._fetcher.models)
        return {
            'success': True,
            'available_models': models,
            'count': len(models),
        }

    async def get_local_models(self) -> dict:
        """List known local model artefacts without a recursive workspace
        scan."""
        models = [
            str(path)
            for name in self._fetcher.models
            if (path := self._fetcher.local_dir / f"best_{name}.pt").is_file()
        ]
        return {
            'success': True,
            'local_models': models,
            'count': len(models),
        }
