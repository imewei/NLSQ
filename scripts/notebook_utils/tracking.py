"""Incremental processing tracking with checksum-based change detection."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingTracker:
    """Tracks processed notebooks to enable incremental updates.

    Uses SHA-256 checksums to detect changes. Stores state in
    .notebook_transforms.json in the repository root.
    """

    def __init__(self, state_file: Path = None):
        """Initialize tracker with state file.

        Args:
            state_file: Path to state file (default: .notebook_transforms.json)
        """
        if state_file is None:
            # Default to repo root
            state_file = Path.cwd() / ".notebook_transforms.json"

        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> dict:
        """Load processing state from file.

        Returns:
            Dictionary mapping notebook paths to processing info
        """
        if not self.state_file.exists():
            return {}

        try:
            with open(self.state_file) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load state file: {e}")
            return {}

    def _save_state(self) -> None:
        """Save processing state to file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    def _compute_checksum(self, notebook_path: Path) -> str:
        """Compute SHA-256 checksum of notebook file.

        Args:
            notebook_path: Path to notebook

        Returns:
            Hex digest of SHA-256 hash
        """
        sha256 = hashlib.sha256()

        with open(notebook_path, "rb") as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def needs_processing(self, notebook_path: Path, transformations: list[str]) -> bool:
        """Check if notebook needs processing.

        Args:
            notebook_path: Path to notebook
            transformations: List of transformation names to apply

        Returns:
            True if notebook should be processed
        """
        # Key by resolved absolute path: a cwd-relative key would silently
        # change (and defeat the checksum-based skip) when this tool runs
        # from a different working directory across invocations.
        path_key = str(notebook_path.resolve())

        # Check if we have state for this notebook
        if path_key not in self.state:
            return True

        state_entry = self.state[path_key]

        # Check if transformations changed
        if set(state_entry.get("transformations", [])) != set(transformations):
            return True

        # Check if file changed - return True if checksum differs
        current_checksum = self._compute_checksum(notebook_path)
        return state_entry.get("checksum") != current_checksum

    def mark_processed(
        self,
        notebook_path: Path,
        transformations: list[str],
        stats: dict = None,
    ) -> None:
        """Mark notebook as processed.

        Args:
            notebook_path: Path to notebook
            transformations: List of transformation names applied
            stats: Optional statistics from processing
        """
        # Key by resolved absolute path (see needs_processing for why).
        path_key = str(notebook_path.resolve())

        # Compute checksum
        checksum = self._compute_checksum(notebook_path)

        # Update state
        self.state[path_key] = {
            "checksum": checksum,
            "transformations": sorted(transformations),
            "last_processed": datetime.now().isoformat(),
            "stats": stats or {},
        }

        # Save state
        self._save_state()

    def clear(self) -> None:
        """Clear all processing state."""
        self.state = {}
        if self.state_file.exists():
            self.state_file.unlink()

    def get_stats(self) -> dict:
        """Get statistics about tracked notebooks.

        Returns:
            Dictionary with tracking statistics
        """
        return {
            "total_tracked": len(self.state),
            "state_file": str(self.state_file),
            "state_file_exists": self.state_file.exists(),
        }
