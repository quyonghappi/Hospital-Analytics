from __future__ import annotations

from typing import Sequence

from airflow.providers.google.cloud.operators.vertex_ai.custom_job import (
    CreateCustomContainerTrainingJobOperator as _Base,
)


class CreateCustomContainerTrainingJobOperator(_Base):
    template_fields: Sequence[str] = tuple(
        {*_Base.template_fields, "display_name", "args", "command"}
    )