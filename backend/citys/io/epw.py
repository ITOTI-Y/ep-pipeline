import asyncio
import re
import tempfile
import zipfile
from pathlib import Path

import httpx
from loguru import logger
from pydantic.dataclasses import dataclass
from tqdm.asyncio import tqdm as tqdm_async

from backend.citys.io._share import TMYX_BASE_URL
from backend.citys.models.schemas import DownloadConfigSchema

YEARS = "2011-2025"
ZIP_PATTERN = f'href="(\\S+\\.{YEARS}\\.zip)"'
PARSE_PATTERN = r"(\S+)/(\S+)\.(\d+)_TMYx\.(\d{4}-\d{4})\.zip"


@dataclass
class ZipLink:
    link: str
    city: str
    province: str
    wmo_id: str
    years: tuple[int, int]


async def _parse_city_name(city_match: str) -> str:
    process1 = city_match.split("_")[-1]
    process2 = process1.split(".")[0]
    city_name = process2.split("-")[-1]
    return city_name


async def _parse_zip_links(link: str) -> ZipLink | None:
    match = re.match(PARSE_PATTERN, link)
    if match:
        province = match.group(1).split("_")[-1]
        city = await _parse_city_name(match.group(2))
        wmo_id = match.group(3)
        years = match.group(4).split("-")
        return ZipLink(
            f"{TMYX_BASE_URL}/{link}",
            city,
            province,
            wmo_id,
            (int(years[0]), int(years[1])),
        )
    return None


async def _download_one(
    client: httpx.AsyncClient,
    url: str,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    cfg: DownloadConfigSchema,
) -> Path | None:
    zip_link = await _parse_zip_links(url)
    if zip_link is None:
        logger.warning(f"Failed to parse zip link: {url}")
        return None

    link = zip_link.link
    new_name = f"{zip_link.city}_{zip_link.wmo_id}.epw"
    dest = output_dir / new_name
    if dest.exists():
        return dest

    async with semaphore:
        for attempt in range(cfg.max_retries):
            try:
                await asyncio.sleep(cfg.request_interval)
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = (
                        Path(temp_dir) / f"{zip_link.city}_{zip_link.wmo_id}.zip"
                    )
                    resp = await client.get(link)
                    resp.raise_for_status()
                    temp_path.write_bytes(resp.content)
                    with zipfile.ZipFile(temp_path, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)
                        epw_file = next(Path(temp_dir).glob("*.epw"))
                        epw_file.rename(dest)
                return dest
            except Exception as e:
                if attempt < cfg.max_retries - 1:
                    wait = cfg.backoff_wait * (2**attempt)
                    logger.warning(f"Retry {attempt + 1} for {new_name}: {e}")
                    await asyncio.sleep(wait)
                else:
                    raise
    return dest


async def download_epw_dataset(
    output_dir: Path, cfg: DownloadConfigSchema
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        index_resp = await client.get(TMYX_BASE_URL)
        index_resp.raise_for_status()
        zip_links = re.findall(ZIP_PATTERN, index_resp.text)

        semaphore = asyncio.Semaphore(cfg.concurrency)
        tasks = []
        for link in zip_links:
            tasks.append(_download_one(client, link, output_dir, semaphore, cfg))

        pbar = tqdm_async(total=len(tasks), desc="Downloading EPW files", unit="file")
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
                logger.warning(f"Download failed: {e}")
            finally:
                pbar.update(1)
        pbar.close()
    if failed:
        logger.warning(f"Failed to download {failed} EPW files")
    logger.info(f"Downloaded {len(downloaded)}/{len(zip_links)} EPW files")
    return downloaded
