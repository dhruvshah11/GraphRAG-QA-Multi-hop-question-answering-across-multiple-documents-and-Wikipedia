"""FastAPI dependencies."""
from __future__ import annotations

from app.api.controller import APIController
from app.core.container import get_container

_controllers: dict[int, APIController] = {}


def get_controller() -> APIController:
    container = get_container()
    ctl = _controllers.get(id(container))
    if ctl is None or ctl.c is not container:
        ctl = APIController(container)
        _controllers.clear()
        _controllers[id(container)] = ctl
    return ctl
