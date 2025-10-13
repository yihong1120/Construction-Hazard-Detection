from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.model_fetcher import ModelFetcher


class ModelTools:
    """Tools for managing ML models and model operations."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._model_fetcher = None
        self._current_models: dict[str, dict] = {}

    async def fetch_model(
        self,
        model_name: str,
        model_version: str | None = None,
        force_download: bool = False,
    ) -> dict:
        """Fetch and download an ML model.

        Args:
            model_name: Name of the model to fetch.
            model_version: Specific version (uses latest if not provided).
            force_download: Force re-download even if the model exists.

        Returns:
            dict[str, Any]: Download status and model information.
        """
        try:
            await self._ensure_model_fetcher()

            # Fallback: request update for a single model
            # synchronously in a thread
            # Note: model_version is ignored by current ModelFetcher API
            last_time = self._model_fetcher.get_last_update_time(model_name)
            await asyncio.to_thread(
                self._model_fetcher.request_new_model,
                model_name,
                last_time,
            )

            model_path = Path(
                self._model_fetcher.local_dir,
                f"best_{model_name}.pt",
            )
            success = bool(model_path.exists())
            model_info = {}
            if success:
                stat = model_path.stat()
                model_info = {
                    'version': None,
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                }
                self._current_models[model_name] = {
                    'path': str(model_path),
                    'version': None,
                    'size': stat.st_size,
                    'fetch_time': asyncio.get_event_loop().time(),
                }

            return {
                'success': success,
                'model_name': model_name,
                'model_path': str(model_path) if success else None,
                'model_info': model_info,
                'message': (
                    f"Model {model_name} downloaded to {model_path}"
                    if success
                    else (
                        'Failed to fetch model '
                        '(API not configured or network error)'
                    )
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to fetch model: {e}")
            raise

    async def list_available_models(self) -> dict:
        """List available models from the model repository.

        Returns:
            dict[str, Any]: List of available models and counts.
        """
        try:
            await self._ensure_model_fetcher()

            # Fallback: list models from ModelFetcher configuration
            models = list(self._model_fetcher.models)

            return {
                'success': True,
                'available_models': models,
                'count': len(models),
                'message': f"Listed {len(models)} configured models",
            }

        except Exception as e:
            self.logger.error(f"Failed to list available models: {e}")
            raise

    async def get_model_info(
        self,
        model_name: str,
    ) -> dict:
        """Get detailed information about a specific model.

        Args:
            model_name: Name of the model.

        Returns:
            dict[str, Any]: Model information.
        """
        try:
            await self._ensure_model_fetcher()

            # Fallback: inspect local filesystem for model info
            p = Path(self._model_fetcher.local_dir, f"best_{model_name}.pt")
            if bool(p.exists()):
                stat = p.stat()
                model_info = {
                    'path': str(p),
                    'size': stat.st_size,
                    'modified': stat.st_mtime,
                }
                return {
                    'success': True,
                    'model_name': model_name,
                    'model_info': model_info,
                    'message': f"Local model found at {p}",
                }
            return {
                'success': False,
                'model_name': model_name,
                'model_info': None,
                'message': (
                    'Model not found locally and remote info not '
                    'implemented'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to get model info: {e}")
            raise

    async def update_model(
        self,
        model_name: str,
        target_version: str | None = None,
    ) -> dict:
        """Update a model to the latest or a specific version.

        Args:
            model_name: Name of the model to update.
            target_version: Specific version to update to (latest if ``None``).

        Returns:
            dict[str, Any]: Update status and new version details.
        """
        try:
            await self._ensure_model_fetcher()

            # Fallback: request update for a single model
            last_time = self._model_fetcher.get_last_update_time(model_name)
            await asyncio.to_thread(
                self._model_fetcher.request_new_model,
                model_name,
                last_time,
            )
            p = Path(self._model_fetcher.local_dir, f"best_{model_name}.pt")
            success = bool(p.exists())
            if success:
                stat = p.stat()
                self._current_models[model_name] = {
                    'path': str(p),
                    'version': None,
                    'size': stat.st_size,
                    'update_time': asyncio.get_event_loop().time(),
                }

            return {
                'success': success,
                'model_name': model_name,
                'previous_version': None,
                'new_version': None,
                'model_path': str(p) if success else None,
                'message': (
                    f"Model {model_name} updated"
                    if success
                    else (
                        'Model update failed '
                        '(API not configured or network error)'
                    )
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to update model: {e}")
            raise

    async def validate_model(
        self,
        model_name: str,
        test_input: str | None = None,
    ) -> dict:
        """Validate model integrity and performance.

        Args:
            model_name: Name of the model to validate.
            test_input: Optional test input (uses default if not provided).

        Returns:
            dict[str, Any]: Validation results.
        """
        try:
            await self._ensure_model_fetcher()

            # Fallback: check if local model file exists
            p = Path(self._model_fetcher.local_dir, f"best_{model_name}.pt")
            exists = bool(p.exists())
            return {
                'success': True,
                'model_name': model_name,
                'is_valid': exists,
                'validation_results': {
                    'exists': exists,
                    'path': str(p) if exists else None,
                },
                'message': (
                    'Local model exists'
                    if exists
                    else 'Local model not found; deep validation not '
                         'implemented'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to validate model: {e}")
            raise

    async def get_local_models(self) -> dict:
        """Get a list of locally cached models.

        Returns:
            dict[str, Any]: Local model information and counts.
        """
        try:
            await self._ensure_model_fetcher()

            # Fallback: scan local model directories
            model_files: list[str] = []
            search_dirs = [
                Path(self._model_fetcher.local_dir),
                Path('models/onnx'),
                Path('models/int8_engine'),
            ]
            exts = {'.pt', '.onnx', '.engine', '.plan'}
            for d in search_dirs:
                if d.exists():
                    for p in d.rglob('*'):
                        if p.is_file() and p.suffix.lower() in exts:
                            model_files.append(str(p))

            return {
                'success': True,
                'local_models': model_files,
                'cached_models': self._current_models,
                'count': len(model_files),
                'message': f"Found {len(model_files)} local model files",
            }

        except Exception as e:
            self.logger.error(f"Failed to get local models: {e}")
            raise

    async def cleanup_old_models(
        self,
        keep_versions: int = 2,
    ) -> dict:
        """Clean up old model versions to save disk space.

        Args:
            keep_versions: Number of versions to keep per model.

        Returns:
            dict[str, Any]: Cleanup results and freed space.
        """
        try:
            await self._ensure_model_fetcher()

            # Fallback: nothing to cleanup (single-version files)
            return {
                'success': True,
                'cleaned_models': 0,
                'freed_space_mb': 0,
                'keep_versions': keep_versions,
                'message': (
                    'Cleanup not implemented for single-file models; '
                    'nothing to do'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to cleanup old models: {e}")
            raise

    async def _ensure_model_fetcher(self) -> None:
        """Ensure the model fetcher is initialised."""
        if self._model_fetcher is None:
            self._model_fetcher = ModelFetcher()
            self.logger.info('Initialised model fetcher')
