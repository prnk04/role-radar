# defining pydantic model to validate output schema
from pydantic import BaseModel


# models to be used by resume_parser
class Experience(BaseModel):
    company: str
    role: str
    start_date: str
    end_date: str
    duration: str
    is_current: bool
    summary: list[str]


class Education(BaseModel):
    school_college: str | None
    degree: str
    major: str | None
    start_date: str
    end_date: str
    is_current: bool


class Certificate(BaseModel):
    name: str
    issuing_authority: str | None
    expiry: str | None
    skills: list[str] | None
    issued_on: str | None


class Projects(BaseModel):
    name: str
    link: str | None
    summary: list[str]


class StructuredResume(BaseModel):
    summary: str | None
    skills: list[str]
    industry_experience: list[Experience] | None
    internship_experience: list[Experience] | None
    projects: list[Projects] | None
    education: list[Education]
    certificates: list[Certificate] | None
    email_id: str


class Role(BaseModel):
    role: str
    score: float
    is_relevant: bool
    strength: list[str]
    gaps: list[str]
    summary: str


class JobRoles(BaseModel):
    user_targetted_roles: list[Role]
    recommended_roles: list[Role]


class JobPostingsModel1(BaseModel):
    id: str
    company: str
    role: str
    location: str
    contract_type: str
    contract_time: str
    posted_on: str
    responsibilities: list[dict[str, str]]
    skills_required: list[dict[str, str]]
    qualifications: list[dict[str, str]]
    skills_optional: list[dict[str, str]]
    additional_requirements: list[dict[str, str]]
    redirect_url: str
    keywords: list[str] | None = None
