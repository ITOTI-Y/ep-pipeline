import asyncio
from pathlib import Path

import httpx
from loguru import logger

from backend.citys.io._share import (
    BTYPE_SHORT,
    BUILDING_TYPES,
    DEST_API_URL,
    DEST_LOAD_URL,
    DEST_YEARS,
)
from backend.citys.models.schemas import DownloadConfigSchema


async def _download_with_fallback(
    client: httpx.AsyncClient,
    city: str,
    btype: str,
    catalog: dict,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    cfg: DownloadConfigSchema,
) -> tuple[str, str]:
    short = BTYPE_SHORT[btype]
    for year in DEST_YEARS:
        key = f"{short}_{city}_{year}"
        out_path = output_dir / city / f"{key}.accdb.7z"
        if out_path.exists():
            return key, "ok"

        if catalog and (city, year, btype) not in catalog:
            continue

        async with semaphore:
            try:
                await asyncio.sleep(cfg.request_interval)
                resp = await client.post(
                    DEST_LOAD_URL,
                    json={
                        "data": {"building_type": btype, "location": city, "year": year}
                    },
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                download_url = data.get("url", "")
                if not download_url:
                    continue
                file_resp = await client.get(download_url)
                file_resp.raise_for_status()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(file_resp.content)
                return key, "ok"
            except httpx.ConnectError:
                await asyncio.sleep(cfg.backoff_wait)
            except (httpx.HTTPStatusError, KeyError, ValueError) as e:
                logger.debug(f"DeST download failed {key}: {e}")

    return f"{short}_{city}_none", "fail"


async def fetch_catalog(cfg: DownloadConfigSchema) -> dict[tuple[str, int, str], bool]:
    catalog: dict[tuple[str, int, str], bool] = {}
    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        for attempt in range(cfg.max_retries):
            try:
                resp = await client.get(DEST_API_URL)
                resp.raise_for_status()
                data = resp.json()
                for entry in data.get("models", data if isinstance(data, list) else []):
                    city = entry.get("location", "")
                    year = entry.get("year", 0)
                    btype = entry.get("building_type", "")
                    if city and year and btype:
                        catalog[(city, int(year), btype)] = True
                logger.info(f"Catalog: {len(catalog)} entries")
                return catalog
            except Exception as e:
                logger.warning(f"Catalog fetch attempt {attempt + 1} failed: {e}")
                if attempt == cfg.max_retries - 1:
                    raise RuntimeError(
                        f"Failed to fetch catalog after {cfg.max_retries} attempts"
                    ) from e
                await asyncio.sleep(cfg.retry_wait)
    return catalog


async def download_dest_models(
    cities: list[str],
    catalog: dict[tuple[str, int, str], bool],
    cfg: DownloadConfigSchema,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(cfg.concurrency)
    report: dict[str, str] = {}

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        tasks = []
        for city in cities:
            for btype in BUILDING_TYPES:
                tasks.append(
                    _download_with_fallback(
                        client, city, btype, catalog, output_dir, semaphore, cfg
                    )
                )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, tuple):
                report[r[0]] = r[1]
            elif isinstance(r, BaseException):
                logger.warning(f"DeST download task failed: {r}")

    ok = sum(1 for v in report.values() if v == "ok")
    logger.info(f"DeST download: {ok}/{len(report)} ok")
    return report
