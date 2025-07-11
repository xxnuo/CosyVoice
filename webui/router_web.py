import logging

from fastapi import APIRouter, HTTPException

from webui.config import Config

logger = logging.getLogger()

router = APIRouter(
    tags=["Web"],
    responses={404: {"description": "Not found"}},
)
