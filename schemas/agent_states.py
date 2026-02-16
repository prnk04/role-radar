from typing import List, Optional, TypedDict


class UserDetailsState(TypedDict):
    resume_path: str
    resume_raw: Optional[str]
    resume_formatted: Optional[dict]
    user_profile: str
    target_roles: Optional[List[str]]
    final_role_analysis: Optional[dict]


class JobSearchState(TypedDict):
    titles: List
    locations: List
    country: Optional[str]
    country_code: Optional[str]
    jobs: List[dict]
    page_number: int
    max_pages: int
    threshold: int
    empty_page_count: int
    keywords: List