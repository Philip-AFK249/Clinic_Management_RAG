"""Async formulary lookups against the ClinicalConsultationService.

The ambient copilot may only prescribe drugs that are actually on the shelf, so
every draft is matched against this live catalogue. Sources, in order:

1. ``GET /api/v1/clinical/pharmacy/drugs/available`` on the Spring Boot service
   (``http://localhost:8083``), falling back to ``/api/v1/clinical/pharmacy/drugs``.
2. Direct read of the service's PostgreSQL database
   (``clinic_management_clinicalconsultation_service`` on port 5432), so a stopped
   Spring Boot process still yields a usable formulary.

Only ``active`` drugs with ``stock_quantity > 0`` are returned, and anything
below :data:`LOW_STOCK_THRESHOLD` units carries a ``LOW_STOCK`` warning so the
LLM prefers drugs with sufficient inventory.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import httpx

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Units below this count are flagged so the LLM prefers better-stocked items.
LOW_STOCK_THRESHOLD = 100

#: Tried in order; the first one answering with a usable payload wins.
DRUG_ENDPOINT_PATHS: tuple[str, ...] = (
    "/api/v1/clinical/pharmacy/drugs/available",
    "/api/v1/clinical/pharmacy/drugs",
)

_STOCK_WARNING = "LOW_STOCK"


class ClinicalClientError(RuntimeError):
    """No formulary source (HTTP or PostgreSQL) could be reached."""


@dataclass(frozen=True)
class AvailableDrug:
    """One in-stock item of the ClinicalConsultationService formulary."""

    id: int
    code: str
    name: str
    concentration: str = ""
    dosage_form: str = ""
    stock_quantity: int = 0
    unit_price: float = 0.0
    bhyt_coverage: str = ""
    is_penicillin_class: bool = False
    active: bool = True

    @property
    def stock_warning(self) -> Optional[str]:
        """``LOW_STOCK`` below the inventory threshold, else ``None``."""
        if self.stock_quantity < LOW_STOCK_THRESHOLD:
            return _STOCK_WARNING
        return None

    @property
    def label(self) -> str:
        """Display name used when the LLM quotes a drug back by name."""
        return f"{self.name} {self.concentration}".strip()

    def compact_row(self) -> str:
        """Single-line formulary row handed to the LLM prompt."""
        return (
            f"{self.id} | {self.code} | {self.name} | {self.concentration} | "
            f"{self.dosage_form} | tồn kho {self.stock_quantity} | "
            f"BHYT {self.bhyt_coverage or '-'}"
            + (" | NHÓM PENICILLIN" if self.is_penicillin_class else "")
            + (" | LOW_STOCK" if self.stock_warning else "")
        )


@dataclass
class FormularyResult:
    """Drugs plus where they came from (and why they are empty, if they are)."""

    drugs: list[AvailableDrug] = field(default_factory=list)
    source: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return bool(self.drugs)


def _pick(raw: dict, *keys: str, default: Any = None) -> Any:
    """First non-``None`` value among ``keys`` (camelCase and snake_case)."""
    for key in keys:
        value = raw.get(key)
        if value is not None:
            return value
    return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if value is None:
        return default
    return bool(value)


def normalise_drug(raw: Any) -> Optional[AvailableDrug]:
    """Fold a Spring Boot ``Drug`` JSON object (or a DB row dict) onto the model."""
    if not isinstance(raw, dict):
        return None
    drug_id = _pick(raw, "id", "drugId")
    if drug_id is None:
        return None
    name = str(_pick(raw, "name", "drugName", default="") or "").strip()
    if not name:
        return None
    return AvailableDrug(
        id=_as_int(drug_id),
        code=str(_pick(raw, "code", "drugCode", default="") or "").strip(),
        name=name,
        concentration=str(_pick(raw, "concentration", default="") or "").strip(),
        dosage_form=str(
            _pick(raw, "dosageForm", "dosage_form", default="") or ""
        ).strip(),
        stock_quantity=max(0, _as_int(_pick(raw, "stockQuantity", "stock_quantity", default=0))),
        unit_price=_as_float(_pick(raw, "unitPrice", "unit_price", default=0.0)),
        bhyt_coverage=str(
            _pick(raw, "bhytCoverage", "bhyt_coverage", default="") or ""
        ).strip(),
        is_penicillin_class=_as_bool(
            _pick(raw, "isPenicillinClass", "is_penicillin_class", default=False)
        ),
        active=_as_bool(_pick(raw, "active", default=True), default=True),
    )


def _in_stock(drugs: Iterable[AvailableDrug]) -> list[AvailableDrug]:
    """Keep only active drugs that still have units on the shelf."""
    filtered = [d for d in drugs if d.active and d.stock_quantity > 0]
    filtered.sort(key=lambda d: (d.name.lower(), d.id))
    return filtered


async def _fetch_from_service(
    settings: Settings, timeout: float
) -> tuple[list[AvailableDrug], str]:
    """Try each formulary endpoint in order; raise when none of them answers."""
    base = settings.CLINICAL_SERVICE_BASE_URL.rstrip("/")
    last_error: Optional[Exception] = None
    answered_with_payload = False

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        for path in DRUG_ENDPOINT_PATHS:
            url = f"{base}{path}"
            try:
                response = await client.get(url)
                response.raise_for_status()
                payload = response.json()
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                # The host itself is unreachable - every other path on the same
                # base URL will fail too, so fail over to PostgreSQL right away.
                raise ClinicalClientError(
                    f"Không kết nối được ClinicalConsultationService tại {base}: {exc}"
                ) from exc
            except Exception as exc:  # HTTP status / JSON payload failures
                last_error = exc
                logger.warning("Formulary endpoint %s failed: %s", url, exc)
                continue

            rows = payload if isinstance(payload, list) else _pick(
                payload, "content", "drugs", "data", default=[]
            ) if isinstance(payload, dict) else []
            drugs = _in_stock(
                drug for drug in (normalise_drug(row) for row in (rows or [])) if drug
            )
            answered_with_payload = True
            if drugs:
                logger.info("Formulary from %s: %d in-stock drug(s)", url, len(drugs))
                return drugs, url
            logger.info("Formulary endpoint %s returned no in-stock drug", url)

    if not answered_with_payload and last_error is not None:
        raise ClinicalClientError(
            f"Không gọi được ClinicalConsultationService tại {base}: {last_error}"
        ) from last_error
    return [], ""


def _fetch_from_database(settings: Settings) -> list[AvailableDrug]:
    """Direct SQL fallback against the service's own PostgreSQL instance."""
    import psycopg

    sql = """
        SELECT id, code, name, concentration, dosage_form, stock_quantity,
               unit_price, bhyt_coverage, is_penicillin_class, active
        FROM drugs
        WHERE active = TRUE AND stock_quantity > 0
        ORDER BY name ASC, id ASC
    """
    with psycopg.connect(
        settings.clinical_db_dsn,
        connect_timeout=settings.CLINICAL_DB_CONNECT_TIMEOUT,
        autocommit=True,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()

    drugs = []
    for row in rows:
        drug = normalise_drug(
            {
                "id": row[0],
                "code": row[1],
                "name": row[2],
                "concentration": row[3],
                "dosageForm": row[4],
                "stockQuantity": row[5],
                "unitPrice": row[6],
                "bhytCoverage": row[7],
                "isPenicillinClass": row[8],
                "active": row[9],
            }
        )
        if drug:
            drugs.append(drug)
    return _in_stock(drugs)


async def get_available_drugs(
    settings: Settings | None = None, timeout: float | None = None
) -> FormularyResult:
    """Return the in-stock formulary, degrading to PostgreSQL when HTTP is down.

    Never raises: an empty ``FormularyResult`` carries an ``error`` string the
    caller can surface as a warning, because a missing formulary must degrade
    the prescription draft rather than fail the whole consultation.
    """
    settings = settings or get_settings()
    timeout = timeout if timeout is not None else settings.CLINICAL_SERVICE_TIMEOUT

    try:
        drugs, source = await _fetch_from_service(settings, timeout)
        if drugs:
            return FormularyResult(drugs=drugs, source=source)
    except ClinicalClientError as exc:
        logger.warning("%s - falling back to direct PostgreSQL read", exc)

    try:
        drugs = await asyncio.to_thread(_fetch_from_database, settings)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}".strip().splitlines()[0][:200]
        logger.warning("Formulary PostgreSQL fallback failed: %s", detail)
        return FormularyResult(
            drugs=[],
            source="",
            error=(
                "Không lấy được danh mục thuốc từ ClinicalConsultationService "
                f"(HTTP và PostgreSQL đều thất bại: {detail})."
            ),
        )

    if not drugs:
        return FormularyResult(
            drugs=[],
            source=f"postgresql:{settings.CLINICAL_DB_NAME}",
            error="Danh mục thuốc đang trống (không có thuốc active còn tồn kho).",
        )
    logger.info(
        "Formulary from PostgreSQL %s: %d in-stock drug(s)",
        settings.CLINICAL_DB_NAME,
        len(drugs),
    )
    return FormularyResult(
        drugs=drugs, source=f"postgresql:{settings.CLINICAL_DB_NAME}"
    )
