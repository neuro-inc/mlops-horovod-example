#!/usr/bin/python

import os
from typing import List

from neuro_sdk import get, JobStatus, JobDescription
from neuro_cli.asyncio_utils import run as run_async

HOROVOD_INSTANCE_JOBS_TAG = "HOROVOD_TAG"


async def main() -> None:
    workers = await get_worker_jobs()
    for w in workers:
        print(f"{w.internal_hostname}:{w.container.resources.gpu}")


async def get_worker_jobs() -> List[JobDescription]:
    tag = os.environ[HOROVOD_INSTANCE_JOBS_TAG]
    result = []
    async with get() as client:
        async for job in client.jobs.list(statuses={JobStatus.RUNNING}, tags=[tag]):
            result.append(job)
    return result


run_async(main())
