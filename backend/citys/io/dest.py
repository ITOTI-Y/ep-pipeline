import asyncio
import tempfile
from itertools import groupby
from pathlib import Path

import httpx
import py7zr
from loguru import logger
from pydantic import BaseModel
from tqdm.asyncio import tqdm as tqdm_async

from backend.citys.io._share import (
    BTYPE_SHORT,
    DEST_CATALOG_URL,
    DEST_LOAD_URL,
)
from backend.citys.models.schemas import DownloadConfigSchema


class DestCatalogEntry(BaseModel):
    city: str
    year: int
    btype: str


async def _download_dest_one(
    client: httpx.AsyncClient,
    entry: DestCatalogEntry,
    cfg: DownloadConfigSchema,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Path | None:
    btype_short = BTYPE_SHORT[entry.btype]
    dest = output_dir / f"{entry.city}_{btype_short}_{entry.year}.accdb"
    if dest.exists():
        return dest
    async with semaphore:
        for attempt in range(cfg.max_retries):
            try:
                await asyncio.sleep(cfg.request_interval)
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = (
                        Path(temp_dir) / f"{entry.city}_{btype_short}_{entry.year}.7z"
                    )
                    resp = await client.post(
                        DEST_LOAD_URL,
                        json={
                            "data": {
                                "building_type": entry.btype,
                                "location": entry.city,
                                "year": entry.year,
                            }
                        },
                    )
                    resp.raise_for_status()
                    temp_path.write_bytes(resp.content)
                    with py7zr.SevenZipFile(temp_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)
                        accdb_file = next(Path(temp_dir).glob("*.accdb"), None)
                        if accdb_file is None:
                            logger.warning(
                                f"No .accdb file in 7z for {entry.city}_{btype_short}_{entry.year}"
                            )
                            return None
                        accdb_file.rename(dest)
                        return dest
            except Exception as e:
                if attempt < cfg.max_retries - 1:
                    wait = cfg.backoff_wait * (2**attempt)
                    logger.warning(
                        f"Retry {attempt + 1} for {entry.city}_{btype_short}_{entry.year}: {e}"
                    )
                    await asyncio.sleep(wait)
                else:
                    raise
    return dest


async def fetch_catalog() -> list[DestCatalogEntry]:
    entries = []
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        resp = await client.get(DEST_CATALOG_URL)
        resp.raise_for_status()
        data = resp.json()
        for key in data.get("names_mapping", {}):
            btype, city, year = key.split("_")
            entries.append(DestCatalogEntry(btype=btype, city=city, year=int(year)))
    entries.sort(key=lambda e: (e.btype, e.city, -e.year))
    deduped = [next(g) for _, g in groupby(entries, key=lambda e: (e.btype, e.city))]
    return deduped


async def download_dest_models(
    catalog: list[DestCatalogEntry],
    cfg: DownloadConfigSchema,
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        semaphore = asyncio.Semaphore(cfg.concurrency)
        tasks = []
        for entry in catalog:
            tasks.append(_download_dest_one(client, entry, cfg, output_dir, semaphore))

        pbar = tqdm_async(
            total=len(tasks), desc="Downloading DeST models", unit="accdb"
        )
        downloaded: list[Path] = []
        failed = 0
        for fut in asyncio.as_completed(tasks):
            try:
                result = await fut
                if isinstance(result, Path):
                    downloaded.append(result)
                    pbar.set_postfix_str(result.name)
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                logger.warning(f"Failed to download DeST model: {e}")
            finally:
                pbar.update(1)
        pbar.close()
    if failed:
        logger.warning(f"Failed to download {failed} DeST models")
    logger.info(f"Downloaded {len(downloaded)}/{len(catalog)} DeST models")
    return downloaded
