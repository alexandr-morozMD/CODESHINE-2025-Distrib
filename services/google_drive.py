from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
except ImportError:  # pragma: no cover - optional dependency
    service_account = None  # type: ignore[assignment]
    build = None  # type: ignore[assignment]
    MediaFileUpload = None  # type: ignore[assignment]

SCOPES = ["https://www.googleapis.com/auth/drive.file"]


class GoogleDriveError(RuntimeError):
    """Raised when uploading artifacts to Google Drive fails."""


def _get_credentials():
    if service_account is None:
        raise GoogleDriveError(
            "Библиотеки google-api-python-client / google-auth не установлены."
        )

    json_blob = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    file_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")

    if json_blob:
        try:
            info = json.loads(json_blob)
        except json.JSONDecodeError as exc:  # pragma: no cover - bad config
            raise GoogleDriveError("GOOGLE_SERVICE_ACCOUNT_JSON невалиден") from exc
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)

    if file_path:
        credentials_path = Path(file_path).expanduser()
        if not credentials_path.exists():
            raise GoogleDriveError(f"File not found: {credentials_path}")
        return service_account.Credentials.from_service_account_file(
            str(credentials_path), scopes=SCOPES
        )

    raise GoogleDriveError(
        "Укажите GOOGLE_SERVICE_ACCOUNT_JSON или GOOGLE_SERVICE_ACCOUNT_FILE для подключения к Google Drive."
    )


def upload_run_to_drive(
    run_id: str,
    run_dir: Path,
    *,
    parent_id: Optional[str] = None,
) -> str:
    """Upload all artifacts from the run directory to Google Drive."""

    if build is None or MediaFileUpload is None:
        raise GoogleDriveError(
            "google-api-python-client libraries are not available in the environment."
        )

    if not run_dir.exists():
        raise GoogleDriveError(f"Directory not found: {run_dir}")

    credentials = _get_credentials()
    service = build("drive", "v3", credentials=credentials)

    folder_metadata = {
        "name": run_id,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        folder_metadata["parents"] = [parent_id]

    folder = (
        service.files()
        .create(body=folder_metadata, fields="id", supportsAllDrives=True)
        .execute()
    )
    folder_id = folder.get("id")
    if not folder_id:
        raise GoogleDriveError("Failed to create folder in Google Drive")

    for path in sorted(run_dir.iterdir()):
        if path.is_dir():
            continue
        media = MediaFileUpload(str(path), resumable=False)
        file_metadata = {"name": path.name, "parents": [folder_id]}
        service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True,
        ).execute()

    return f"https://drive.google.com/drive/folders/{folder_id}"


__all__ = ["GoogleDriveError", "upload_run_to_drive"]
