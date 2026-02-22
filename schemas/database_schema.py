from bson import ObjectId
from pydantic import BaseModel, Field
import datetime

from schemas.data_models import Experience, Education, Certificate, Projects


def get_utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")


class DummyTable(BaseModel):
    name: str
    company: str


class UserProfileFullModel(BaseModel):
    user_id: str
    summary: str | None = None
    skills: list[str]
    industry_experience: list[Experience] | None = None
    internship_experience: list[Experience] | None = None
    projects: list[Projects] | None = None
    education: list[Education]
    certificates: list[Certificate] | None = None
    version: int = 0
    resume_hash: str
    is_active: bool = True
    created_at: str
    last_updated_at: str = Field(default_factory=get_utc_timestamp)


class UserProfileShortModel(BaseModel):
    user_id: str
    full_profile_id: str
    profile: str
    prev_roles: list[str] | None
    all_skills: list[str]
    embeddings_status: str = 'pending'
    last_updated_at: str = Field(default_factory=get_utc_timestamp)
    created_at: str


class RoleModel(BaseModel):
    name: str = Field(min_length=1)
    match_score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    selected: bool = False
    strengths: str = Field(min_length=1)
    gaps: str = Field(default='')


class JobRoleRecommendationsModel(BaseModel):
    _id: str
    recommended_roles: list[RoleModel]
    profile_version: int
    last_updated_at: str = Field(default_factory=get_utc_timestamp)


class SkillsModel(BaseModel):
    must_have: list[str]
    good_to_have: list[str]


class JobPostingsModel_DB(BaseModel):
    id: str
    company: str
    role: str
    location: str
    contract_type: str
    contract_time: str
    posted_on: str
    responsibilities: list[str] | None
    skills_required: list[str] | None
    qualifications: list[str] | None
    skills_optional: list[str] | None
    additional_requirements: list[str] | None
    redirect_url: str
    keywords: list[str] | None = None
    job_hashed: str
    what: str
    where: str
    last_seen_at: str = Field(default_factory=get_utc_timestamp)
    last_updated_at: str = Field(default_factory=get_utc_timestamp)
    created_at: str = Field(default_factory=get_utc_timestamp)


class BriefJobPostingsModel(BaseModel):
    id: str
    role: str
    profile: str
    skills_required: list[str]
    skills_optional: list[str]
    full_profile_id: str
    created_at: str = Field(default_factory=get_utc_timestamp)
    last_updated_at: str = Field(default_factory=get_utc_timestamp)


class CountryCodeModel(BaseModel):
    country: str
    country_code: str


class LocationsModel(CountryCodeModel):
    location: str


class RoleBasedKeywordsModel(BaseModel):
    job_title: str
    keywords: list[str]
