from pydantic import BaseModel, Field
import datetime

from data_models import Experience, Education, Certificate, Projects


def get_utc_timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")


class UserProfileFullModel(BaseModel):
    _id: str
    summary: str | None = None
    skills: list[str]
    industry_experience: list[Experience] | None = None
    internship_experience: list[Experience] | None = None
    projects: list[Projects] | None = None
    education: list[Education]
    certificates: list[Certificate] | None = None
    profile_version: int
    last_updated_at: str = Field(default_factory=get_utc_timestamp)


class UserProfileShortModel(BaseModel):
    _id: str
    profile: str
    profile_version: int
    last_updated_at: str = Field(default_factory=get_utc_timestamp)


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


class JobPostingsModel(BaseModel):
    _id: str
    company: str
    role: str
    location: str
    contract_type: str
    contract_time: str
    posted_on: str
    responsibilities: list[str]
    company_description: str
    skills: SkillsModel
    additional_requirements: str | None = None
    url: str
    last_updated_at: str = Field(default_factory=get_utc_timestamp)


class CountryCodeModel(BaseModel):
    country: str
    country_code: str


class LocationsModel(CountryCodeModel):
    location: str


class RoleBasedKeywordsModel(BaseModel):
    job_title: str
    keywords: list[str]
