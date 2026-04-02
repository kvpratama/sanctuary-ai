from __future__ import annotations

import datetime
import uuid
from typing import (
    Annotated,
    Any,
    List,
    Literal,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
)

from pydantic import BaseModel, Field, Json

NetRequestStatus: TypeAlias = Literal["PENDING", "SUCCESS", "ERROR"]

RealtimeEqualityOp: TypeAlias = Literal["eq", "neq", "lt", "lte", "gt", "gte", "in"]

RealtimeAction: TypeAlias = Literal["INSERT", "UPDATE", "DELETE", "TRUNCATE", "ERROR"]

StorageBuckettype: TypeAlias = Literal["STANDARD", "ANALYTICS", "VECTOR"]

AuthFactorType: TypeAlias = Literal["totp", "webauthn", "phone"]

AuthFactorStatus: TypeAlias = Literal["unverified", "verified"]

AuthAalLevel: TypeAlias = Literal["aal1", "aal2", "aal3"]

AuthCodeChallengeMethod: TypeAlias = Literal["s256", "plain"]

AuthOneTimeTokenType: TypeAlias = Literal["confirmation_token", "reauthentication_token", "recovery_token", "email_change_token_new", "email_change_token_current", "phone_change_token"]

AuthOauthRegistrationType: TypeAlias = Literal["dynamic", "manual"]

AuthOauthAuthorizationStatus: TypeAlias = Literal["pending", "approved", "denied", "expired"]

AuthOauthResponseType: TypeAlias = Literal["code"]

AuthOauthClientType: TypeAlias = Literal["public", "confidential"]

class PublicDocuments(BaseModel):
    author: Optional[str] = Field(alias="author")
    blob_url: str = Field(alias="blob_url")
    current_page: int = Field(alias="current_page")
    id: uuid.UUID = Field(alias="id")
    ingested_at: Optional[datetime.datetime] = Field(alias="ingested_at")
    is_ingesting: bool = Field(alias="is_ingesting")
    last_accessed: Optional[datetime.datetime] = Field(alias="last_accessed")
    name: str = Field(alias="name")
    page_count: Optional[int] = Field(alias="page_count")
    size: int = Field(alias="size")
    thumbnail_url: Optional[str] = Field(alias="thumbnail_url")
    upload_date: datetime.datetime = Field(alias="upload_date")
    user_id: uuid.UUID = Field(alias="user_id")

class PublicDocumentsInsert(TypedDict):
    author: NotRequired[Annotated[Optional[str], Field(alias="author")]]
    blob_url: Annotated[str, Field(alias="blob_url")]
    current_page: NotRequired[Annotated[int, Field(alias="current_page")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    ingested_at: NotRequired[Annotated[Optional[datetime.datetime], Field(alias="ingested_at")]]
    is_ingesting: NotRequired[Annotated[bool, Field(alias="is_ingesting")]]
    last_accessed: NotRequired[Annotated[Optional[datetime.datetime], Field(alias="last_accessed")]]
    name: Annotated[str, Field(alias="name")]
    page_count: NotRequired[Annotated[Optional[int], Field(alias="page_count")]]
    size: Annotated[int, Field(alias="size")]
    thumbnail_url: NotRequired[Annotated[Optional[str], Field(alias="thumbnail_url")]]
    upload_date: NotRequired[Annotated[datetime.datetime, Field(alias="upload_date")]]
    user_id: Annotated[uuid.UUID, Field(alias="user_id")]

class PublicDocumentsUpdate(TypedDict):
    author: NotRequired[Annotated[Optional[str], Field(alias="author")]]
    blob_url: NotRequired[Annotated[str, Field(alias="blob_url")]]
    current_page: NotRequired[Annotated[int, Field(alias="current_page")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    ingested_at: NotRequired[Annotated[Optional[datetime.datetime], Field(alias="ingested_at")]]
    is_ingesting: NotRequired[Annotated[bool, Field(alias="is_ingesting")]]
    last_accessed: NotRequired[Annotated[Optional[datetime.datetime], Field(alias="last_accessed")]]
    name: NotRequired[Annotated[str, Field(alias="name")]]
    page_count: NotRequired[Annotated[Optional[int], Field(alias="page_count")]]
    size: NotRequired[Annotated[int, Field(alias="size")]]
    thumbnail_url: NotRequired[Annotated[Optional[str], Field(alias="thumbnail_url")]]
    upload_date: NotRequired[Annotated[datetime.datetime, Field(alias="upload_date")]]
    user_id: NotRequired[Annotated[uuid.UUID, Field(alias="user_id")]]

class PublicDocumentEmbeddings(BaseModel):
    chunk_key: str = Field(alias="chunk_key")
    content: str = Field(alias="content")
    created_at: datetime.datetime = Field(alias="created_at")
    document_id: uuid.UUID = Field(alias="document_id")
    embedding: list[Any] = Field(alias="embedding")
    id: uuid.UUID = Field(alias="id")
    metadata: Json[Any] = Field(alias="metadata")
    user_id: uuid.UUID = Field(alias="user_id")

class PublicDocumentEmbeddingsInsert(TypedDict):
    chunk_key: Annotated[str, Field(alias="chunk_key")]
    content: Annotated[str, Field(alias="content")]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    document_id: Annotated[uuid.UUID, Field(alias="document_id")]
    embedding: Annotated[list[Any], Field(alias="embedding")]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    metadata: NotRequired[Annotated[Json[Any], Field(alias="metadata")]]
    user_id: Annotated[uuid.UUID, Field(alias="user_id")]

class PublicDocumentEmbeddingsUpdate(TypedDict):
    chunk_key: NotRequired[Annotated[str, Field(alias="chunk_key")]]
    content: NotRequired[Annotated[str, Field(alias="content")]]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    document_id: NotRequired[Annotated[uuid.UUID, Field(alias="document_id")]]
    embedding: NotRequired[Annotated[list[Any], Field(alias="embedding")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    metadata: NotRequired[Annotated[Json[Any], Field(alias="metadata")]]
    user_id: NotRequired[Annotated[uuid.UUID, Field(alias="user_id")]]
