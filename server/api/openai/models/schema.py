from __future__ import annotations

from typing import Annotated, List, Literal

from pydantic import BaseModel, Field


class Model(BaseModel):
    id: Annotated[
        str,
        Field(
            description="The model identifier, which can be referenced in the API endpoints."
        ),
    ]
    created: Annotated[
        int,
        Field(
            description="The Unix timestamp (in seconds) when the model was created."
        ),
    ]
    object: Annotated[
        Literal["model"], Field(description='The object type, which is always "model".')
    ]
    owned_by: Annotated[str, Field(description="The organization that owns the model.")]


class ListModelsResponse(BaseModel):
    object: Literal["list"]
    data: List[Model]
