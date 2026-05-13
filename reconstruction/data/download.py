import os
import shutil
import argparse
import json
import urllib.request
import hashlib
from urllib.error import URLError, HTTPError
from typing import Union

from tqdm import tqdm

FIGSHARE_URL_PREFIX = 'https://figshare.com/ndownloader/files/'
FIGSHARE_FALLBACK_PREFIX = 'https://ndownloader.figshare.com/files/'


def main(cfg):
    with open(cfg.filelist, 'r') as f:
        filelist = json.load(f)

    target = filelist[cfg.target]

    for fl in target['files']:
        output = os.path.join(target['save_in'], fl['name'])

        # Downloading
        if not os.path.exists(output):
            print(f'Downloading {output} from {fl["url"]}')
            download_file(fl['url'], output, progress_bar=True, md5sum=fl['md5sum'])

        # Postprocessing
        if 'postproc' in fl:
            for pp in fl['postproc']:
                if pp['name'] == 'unzip':
                    print(f'Unzipping {output}')
                    if 'destination' in pp:
                        dest = pp['destination']
                    else:
                        dest = './'
                    shutil.unpack_archive(output, extract_dir=dest)


def download_file(url: str, destination: str, progress_bar: bool = True, md5sum: Union[str, None] = None) -> None:
    '''Download a file.'''

    candidate_urls = [url]
    fallback_url = get_figshare_fallback_url(url)
    if fallback_url is not None:
        candidate_urls.append(fallback_url)

    last_error = None
    for attempt, candidate_url in enumerate(candidate_urls, start=1):
        try:
            _download_file_once(candidate_url, destination, progress_bar=progress_bar, md5sum=md5sum)
            return
        except (OSError, ValueError, URLError, HTTPError) as error:
            last_error = error
            if os.path.exists(destination):
                os.remove(destination)
            if attempt == len(candidate_urls):
                break
            print(f'Download failed from {candidate_url}: {error}')
            print(f'Retrying with fallback URL: {candidate_urls[attempt]}')

    raise RuntimeError(f'Failed to download {destination} from all candidate URLs') from last_error


def get_figshare_fallback_url(url: str) -> Union[str, None]:
    if not url.startswith(FIGSHARE_URL_PREFIX):
        return None
    return url.replace(FIGSHARE_URL_PREFIX, FIGSHARE_FALLBACK_PREFIX, 1)


def _download_file_once(url: str, destination: str, progress_bar: bool = True, md5sum: Union[str, None] = None) -> None:
    response = urllib.request.urlopen(url)
    file_size_header = response.info().get("Content-Length")
    file_size = int(file_size_header) if file_size_header is not None else None

    def _show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            progress.update(downloaded - progress.n)

    with tqdm(
        total=file_size,
        unit='B',
        unit_scale=True,
        desc=destination,
        ncols=100,
        disable=not progress_bar,
    ) as progress:
        urllib.request.urlretrieve(url, destination, _show_progress if progress_bar else None)

    if md5sum is not None:
        md5_hash = hashlib.md5()
        with open(destination, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                md5_hash.update(chunk)
        md5sum_test = md5_hash.hexdigest()
        if md5sum != md5sum_test:
            raise ValueError(f'md5sum mismatch. \nExpected: {md5sum}\nActual: {md5sum_test}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filelist', default='files.json')
    parser.add_argument('target')

    cfg = parser.parse_args()

    main(cfg)
