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
    school_college: str
    degree: str
    major: str | None
    start_date: str
    end_date: str
    is_current: bool


class Certificate(BaseModel):
    name: str
    issuing_authority: str
    expiry: str | None
    skills: list[str] | None
    issued_on: str


class Projects(BaseModel):
    name: str
    link: list[str] | None
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
