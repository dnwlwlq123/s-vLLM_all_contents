"""
S3/MinIO 클라이언트 — 외부 오브젝트 스토리지 통신 캡슐화.

boto3 S3 호환 API를 사용하여 MinIO와 AWS S3 모두 지원한다.
외부 응답을 내부 Dict로 변환하여 반환한다.
"""

import os
from typing import Dict, Any, List, Optional

from loguru import logger

from vllm_serving.config import settings


class S3Client:
    """S3/MinIO 오브젝트 스토리지 클라이언트."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        """boto3 S3 클라이언트 지연 초기화."""
        if self._client is None:
            import boto3

            client_kwargs = {
                "aws_access_key_id": settings.S3_ACCESS_KEY,
                "aws_secret_access_key": settings.S3_SECRET_KEY,
                "region_name": settings.S3_REGION,
            }
            if settings.S3_ENDPOINT_URL:
                client_kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL

            self._client = boto3.client("s3", **client_kwargs)
        return self._client

    def download_directory(
        self,
        s3_path: str,
        local_path: str,
        bucket: Optional[str] = None,
    ) -> List[str]:
        """
        S3 디렉토리 전체를 로컬에 다운로드.

        Args:
            s3_path: S3 버킷 내 경로 (버킷명 제외)
            local_path: 로컬 저장 절대 경로
            bucket: S3 버킷 (None이면 설정값 사용)

        Returns:
            다운로드된 파일 상대 경로 리스트
        """
        bucket = bucket or settings.S3_BUCKET
        client = self._get_client()
        os.makedirs(local_path, exist_ok=True)

        paginator = client.get_paginator("list_objects_v2")
        s3_prefix = s3_path.rstrip("/") + "/"
        downloaded = []

        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative = s3_key[len(s3_prefix):]
                if not relative:
                    continue

                dest = os.path.join(local_path, relative)
                os.makedirs(os.path.dirname(dest), exist_ok=True)

                logger.debug(f"  S3 다운로드: {s3_key} → {dest} ({obj['Size']} bytes)")
                client.download_file(bucket, s3_key, dest)
                downloaded.append(relative)

        return downloaded

    def list_objects(self, s3_path: str, bucket: Optional[str] = None) -> List[Dict[str, Any]]:
        """S3 경로 내 오브젝트 목록 조회."""
        bucket = bucket or settings.S3_BUCKET
        client = self._get_client()
        s3_prefix = s3_path.rstrip("/") + "/"

        objects = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                objects.append({
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": str(obj.get("LastModified", "")),
                })

        return objects


s3_client = S3Client()
