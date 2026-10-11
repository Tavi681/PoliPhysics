from ml.models.base import MODEL_REGISTRY, SurrogateModel, build_model, register_model
from ml.models import stub_gnn as _stub_gnn  # noqa: F401
from ml.models import baselines as _baselines  # noqa: F401
from ml.models import gns_baseline as _gns_baseline  # noqa: F401

__all__ = ["MODEL_REGISTRY", "SurrogateModel", "build_model", "register_model"]
