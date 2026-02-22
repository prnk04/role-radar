import asyncio
import functools
import time
import aiohttp
import bs4
import json
import ollama
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Optional
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
# from pymongo.database import Database
from pymongo.asynchronous.database import AsyncDatabase


import logging

from pydantic import BaseModel
from src.utils.commons import get_hashed
from schemas.database_schema import BriefJobPostingsModel, JobPostingsModel_DB
from src.database.store_job_postings import bulk_store_job_profiles, bulk_store_jobs, store_job_posting, store_job_profiles
from src.utils.error_handler import log_error
from src.vectors.store_job_profiles import store_job_profile_vector


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_BASE_URL = os.getenv("ADZUNA_BASE_URL")
MODEL_NAME = os.getenv("MODEL_JOB_FORMAT", "llama3:latest")
MODEL_NAME_QWEN_SMALL = os.getenv("MODEL_NAME_QWEN_SMALL", "qwen3:1.7b")
# MODEL_NAME = 'llama3:latest'

llm = ChatOllama(model=MODEL_NAME)


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
    job_hashed_list: List


class SkillsModel(BaseModel):
    must_have: list[str]
    good_to_have: list[str]


class JobPostingsModel(BaseModel):
    id: str
    company: str
    role: str
    location: str
    contract_type: str
    contract_time: str
    posted_on: str
    responsibilities: list[str]
    company_description: str
    qualifications: list[str]
    skills: SkillsModel
    additional_requirements: str | None = None
    redirect_url: str
    keywords: list[str]


class JobTitleKeywords(BaseModel):
    job_title: str
    keywords: str


class JobTitleKeywordsList(BaseModel):
    result: list[JobTitleKeywords]


class JobPostingsModel1(BaseModel):
    id: str
    company: str
    role: str
    location: str
    contract_type: str
    contract_time: str
    posted_on: str
    responsibilities: list[str]
    skills_required: list[str]
    qualifications: list[str]
    skills_optional: list[str]
    additional_requirements: list[str]
    redirect_url: str
    keywords: list[str] | None = None
    job_hashed: str


def logging_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Inside function {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Finished function {func.__name__} in {end_time - start_time}")
        return result
    return wrapper


@logging_decorator
def checkLLMResponse(llm_res):
    this_res = llm_res
    llm_res_content = llm_res.content
    # logging.info(f"Inside check llm res")
    if llm_res_content is not None and type(llm_res_content) == str:
        logging.info(f"llm_res is not none, also it is of type str")
        if str(llm_res_content).__contains__("```"):
            this_res = str(llm_res_content).split("```")[1]
        elif str(llm_res_content).__contains__("\n\n"):
            this_res = str(llm_res_content).split("\n\n")[1]
        elif str(llm_res_content).startswith("{"):
            this_res = llm_res.content
    # logging.info(f"Returning from checkLLMResponse")
    return this_res


@logging_decorator
async def get_structured_data_async(data):
    """
    Async version: Fetches company name, job title, roles and responsibilities, skills from the HTML input
    :param data: section part of the HTML that contains details about the job posting
    """
    try:

        parser = JsonOutputParser(pydantic_object=JobPostingsModel1)

        # from_llm(parser=parser, llm=model)

        prompt_cl = ChatPromptTemplate.from_messages([
            ("system", """
                You are an ATS expert specializing in extracting comprehensive, structured data from job postings. 
                
                CRITICAL RULES:
                1. Extract EVERY responsibility, skill, and requirement mentioned - do not summarize or skip items
                2. When you see a list (bullets, numbered items), extract each item separately
                3. Break down composite skills into individual components
                4. Maintain the original wording - do not paraphrase or merge items
                5. DO NOT CHANGE THE id FIELD
                6. STRICTLY follow the provided schema for output {format_instructions}
            """),
            ("human", """
                Extract complete information from the HTML snippet and JSON object below.
                
                EXTRACTION GUIDELINES:
                
                **Responsibilities:**
                - Extract EACH bullet point and sub-item as a separate responsibility
                - Include both high-level categories AND specific tasks under them
                - Do not merge or summarize - preserve all details
                
                **Skills:**
                Required (must-have):
                - Extract from sections like "Required Skills", "Core (must-have)", "Basic Qualifications"
                - Break composite skills: "Python (Pandas, NumPy)" → ["Python", "Pandas", "NumPy"]
                - Include proficiency levels if mentioned: "Python (strong)" → "Python (strong)"
                
                Optional (good-to-have):
                - Extract from "Good-to-have", "Preferred", "Nice-to-have" sections
                - Same breaking rules apply
                
                **Educational Qualifications:**
                Required: From "Required Qualifications", "Basic Qualifications"
                Optional: From "Preferred Qualifications"
                
                **Experience:**
                Required: Minimum years and type from required sections
                Good-to-have: From preferred/optional sections
                
                **Keywords:**
                - Include all explicitly mentioned technologies, frameworks, tools
                - Add related/implied technologies:
                Example: "Deep Learning" → add ["TensorFlow", "PyTorch", "Keras"]
                Example: "Cloud" → add specific providers mentioned or implied
                - Include domain-specific terms (e.g., "RAG", "LLM", "feature engineering")
                
                **Additional Requirements:**
                - Certifications, publications, domain experience from preferred sections
                
                VALIDATION CHECKLIST (mentally verify before outputting):
                - Did I extract every bullet point from the HTML?
                - Did I break down all composite skills?
                - Did I separate required vs optional appropriately?
                - Did I maintain original wording without summarizing?
                - Did I include all technology keywords?
                - Is the JSON valid?
                - Does the output JSON contain data that cannot be parsed? If yes- fix it
                
             
             Always provide ONLY VALID JSON OUTPUT. 
            
                HTML_SNIPPET: {html_snippet}
                JSON_OBJECT: {job_posting_partial}
            """)
        ])

        prompt_cl = prompt_cl.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model=MODEL_NAME, temperature=0.0, format="json")

        chain = prompt_cl | model | RunnableLambda(
            checkLLMResponse) | parser
        max_attempts = 3
        for attempt in range(max_attempts):
            llm_res = None
            try:
                logging.info(f"Attempt: {attempt}")

                # Run LLM call in executor to avoid blocking
                loop = asyncio.get_event_loop()
                llm_res = await loop.run_in_executor(
                    None,
                    lambda: chain.invoke({
                        "html_snippet": data["html_snippet"],
                        "job_posting_partial": data["job_posting_partial"]
                    })
                )
                if llm_res and llm_res is not None and type(llm_res) == dict:
                    # logging.info(f"llm res is: {llm_res}")
                    # logging.info(
                    #     f"And the snippet was: {data['job_posting_partial']}")
                    llm_res['id'] = data["job_posting_partial"].get("id")
                    llm_res['redirect_url'] = llm_res.get("redirect_url") or data["job_posting_partial"].get(
                        "redirect_url")
                    llm_res["job_hashed"] = data["job_posting_partial"].get(
                        "job_hashed")
                return llm_res

            except Exception as e:
                logging.error(
                    f"Error in invoking get structured data: {e.__cause__}")
                logging.error(f"Error in invoking get structured data: {e}")
                if attempt == max_attempts - 1:
                    return None

        return None

    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in getting structured data: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in getting structured data: {str(e)}"
        )
        return None


@logging_decorator
async def fetch_redirect_url_async(session: aiohttp.ClientSession, url: str, timeout: int = 30):
    """
    Async function to fetch content from redirect URL
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                logging.error(
                    f"Redirect URL returned status {response.status}: {url}")
                if response.status == 403:
                    # logging.info("Returning 403")
                    return {"status": 403, "url": url}
                return None

            text = await response.text()
            return {"status": 200, "text": text, "url": url}
    except asyncio.TimeoutError:
        logging.error(f"Timeout fetching redirect URL: {url}")
        log_error(
            f"Timeout fetching redirect URL: {url}"
        )
        return None
    except Exception as e:
        logging.error(f"Error fetching redirect URL {url}: {e}")
        log_error(
            f"Error fetching redirect URL {url}: {e}"
        )
        return None


@logging_decorator
async def extract_job_requirements_async(job_data, session: aiohttp.ClientSession):
    """
    Async version: Extract details of job posting from the result of calling Adzuna API

    :param job_data: Adzuna API response
    :param session: aiohttp session for making requests
    """
    jobs_to_send = dict()
    job_details_hashed = ""
    try:

        company_name = job_data.get("display_name", "NA")
        job_title = job_data.get("title", "NA")
        description = job_data.get("description", "NA")
        contract_time = job_data.get("contract_time", "NA")
        job_location = job_data.get("location", {}).get("display_name", "NA")
        posted_at = job_data.get("created", "NA")
        contract_type = job_data.get("contract_type", "NA")
        more_details_url = job_data.get("redirect_url", "NA")
        adRef = job_data.get("adref", "NA")

        job_details = f"""{job_data.get("id")}|{company_name}|{job_title}|{description}|{contract_time}|{job_location}|{contract_type}"""
        job_details_hashed = get_hashed(job_details)

        job_details_to_show = {
            "id": job_data.get("id", "NA"),
            "company_from_api": company_name,
            "job_title_from_api": job_title,
            "job_location_from_api": job_location,
            "posted_at_from_api": posted_at,
            "contract_time_from_api": contract_time,
            "contract_type_from_api": contract_type,
            "company_description_from_api": description,
            "redirect_url": more_details_url,
            "job_hashed": job_details_hashed
        }

        jobs_to_send = {
            "id": job_data.get("id", "NA"),
            "company": company_name,
            "role": job_title,
            "location": job_location,
            "posted_on": posted_at,
            "contract_type": contract_type,
            "contract_time": contract_time,
            "company_description": description,
            "redirect_url": more_details_url,
            "job_hashed": job_details_hashed
        }

        # Async fetch of redirect URL
        redirect_result = await fetch_redirect_url_async(session, more_details_url)
        # logging.info(f"Response from fetch redirect url: {redirect_result}")

        if redirect_result is None:
            return None

        if redirect_result.get("status") == 403:

            job_details_to_show["redirect_url"] = more_details_url
            mod_details = await extract_data_from_description(description)

            # logging.info(f"Mod details: {mod_details}")
            if mod_details is not None:
                jobs_to_send = {
                    "id": job_data.get("id", "NA"),
                    "company": company_name or mod_details.get('company', ''),
                    "role": job_title or mod_details.get('job_title', ''),
                    "location": job_location or mod_details.get('job_location', ''),
                    "posted_on": posted_at or mod_details.get('posted_at', ''),
                    "contract_type": contract_type or mod_details.get('contract_type', ''),
                    "contract_time": contract_time or mod_details.get('contract_time', ''),
                    "company_description": description or mod_details.get('description', ''),
                    "responsibilities": mod_details.get("responsibilities", []),
                    "skills": mod_details.get("responsibilities", []),
                    "additional_requirements": mod_details.get("additional_requirements", ''),
                    "redirect_url": more_details_url or mod_details.get('more_details_url', ''),
                    "job_hashed": job_details_hashed
                }
            return jobs_to_send

        job_html = bs4.BeautifulSoup(
            redirect_result["text"], features="html.parser")
        job_details = job_html.find_all("section", class_="adp-body")

        company_name_placeholder = job_html.find("div", class_="ui-company")
        company_name_option = (
            company_name_placeholder.text.strip() if company_name_placeholder else ""
        )

        locations_placeholder = job_html.find("div", class_="ui-location")
        locations_option = (
            [
                x
                for x in locations_placeholder.getText(":").strip().split(":")
                if len(x) > 0
            ]
            if locations_placeholder is not None
            else []
        )

        contract_type_placeholder = job_html.find(
            "div", class_="ui-contract-type")
        contract_type_option = (
            contract_type_placeholder.getText(":").strip()
            if contract_type_placeholder is not None
            else "NA"
        )

        contract_time_placeholder = job_html.find(
            "div", class_="ui-contract-time")
        contract_time_option = (
            contract_time_placeholder.getText(":").strip()
            if contract_time_placeholder is not None
            else "NA"
        )

        posted_at_placeholder = job_html.find("div", class_="ui-posted")
        posted_at_option = (
            posted_at_placeholder.getText(":").strip()
            if posted_at_placeholder is not None
            else "NA"
        )

        if len(job_details) == 0:
            job_details_to_show["redirect_url"] = more_details_url
            return jobs_to_send

        job_details_to_show["redirect_url"] = more_details_url

        html_snippet_to_send = job_details[0]

        data_to_send = {
            "html_snippet": str(html_snippet_to_send),
            "job_posting_partial": job_details_to_show
        }

        # Async LLM call
        llm_response = await get_structured_data_async(data_to_send)

        if llm_response is None:
            return jobs_to_send

        return llm_response

    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in extracting job requirements: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in extracting job requirements: {str(e)}"
        )
        return jobs_to_send


@logging_decorator
async def process_jobs_concurrently(jobs_list: List[dict], max_concurrent: int = 5):
    """
    Process multiple job postings concurrently with a concurrency limit

    :param jobs_list: List of job data from Adzuna API
    :param max_concurrent: Maximum number of concurrent requests
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(job_data, session):
        async with semaphore:
            return await extract_job_requirements_async(job_data, session)

    async with aiohttp.ClientSession() as session:
        tasks = [process_with_semaphore(job, session) for job in jobs_list]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        # logging.info(f"results: {results}")

    # Filter out None results and exceptions
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Exception during job processing: {result}")
        elif result is not None:
            valid_results.append(result)

    return valid_results


@logging_decorator
async def extract_data_from_description(data):
    """
    Infers data from the description field of the API result
    :param data: description field of the API result
    """
    try:
        parser = JsonOutputParser(pydantic_object=JobPostingsModel1)
        prompt = ChatPromptTemplate.from_messages([
            ("system",  """
                            You are an expert in extracting meaningful and structured data from unstructured data.
                            You have worked on countless HTML snippets, and JSON data that has multiple fields conveying the same meaning.
                            You STRICTLY FOLLOW THE FORMAT GIVEN TO YOU TO convert the unstructured data, into meaningful JSON data.
                            STRICTLY FOLLOW THE PROVIDED SCHEMA TO CREATE OUTPUT. {format_instructions}
                        """),
            ("human", """
                        You are given a text exerpt from Job Posting, that may contain:
                                - company description, 
                                - or some other information like role, responsibilities, etc.
             
                        Extract all the details, and use them to populate the JSON.
                        Also, given the data, extract keywords that can be used to match a users' skills to the job requirements
             
                        
                        DO NOT include: "$ref", "$defs", or any schema references.
                        DO NOT put must_have/good_to_have at root level - they MUST be nested inside "skills".
                        - Strictly follow the given schema to create a VALID JSON object.
             

                        text: {text}
                            
                        """)
        ])

        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model=MODEL_NAME, temperature=0.0, format="json")

        chain = prompt | model | RunnableLambda(
            checkLLMResponse) | parser
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # logging.info(f"Attempt: {attempt}")

               # Run LLM call in executor to avoid blocking
                loop = asyncio.get_event_loop()
                llm_res = await loop.run_in_executor(
                    None,
                    lambda: chain.invoke({
                        "text": data
                    })
                )
                # logging.info(f"Intermediate response: {llm_res}")

                return llm_res
            except Exception as e:
                logging.error(
                    f"Error in invoking extract_data_from_description: {e}")
                if attempt == max_attempts - 1:
                    return None

        return None

    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in extract_data_from_description: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in extract_data_from_description: {str(e)}"
        )
        return None


@logging_decorator
def extract_job_data(data, extra=None):
    try:
        formatted_data = ""
        extra_return = None
        if extra == "skills":
            pass

        if data is not None:
            if type(data) == dict:
                formatted_data += "|".join(["|".join(x.split(","))
                                           for x in data.values() if isinstance(x, str)])

            elif type(data) == list:
                this_res = ""
                for res in data:
                    if type(res) == dict:
                        this_res += "|".join(["|".join(x.split(","))
                                             for x in res.values() if isinstance(x, str)])
                        this_res += "|"
                    elif type(res) == list:
                        this_res += "|".join(["|".join(x.split(","))
                                             for x in res if isinstance(x, str)])
                        this_res += "|"

                    elif type(res) == str:
                        this_res += "".join(["|".join(x.split(","))
                                            for x in res if isinstance(x, str)])
                        this_res += "|"
                formatted_data += this_res

            elif type(data) == str:
                formatted_data += "|".join(data.split(","))
                formatted_data += "|"
        return formatted_data, formatted_data.split("|")
    except Exception as e:
        logging.error(f"Error occurred in extracting job data: {e}")
        log_error(f"Error occurred in extracting job data: {e}")
        return str(data), None


@logging_decorator
def clean_text_list(thisList):
    data_to_send = list()
    for thisData in thisList:
        modified_data = ",".join([x.strip() for x in thisData.split("(")])
        modified_data = ",".join([x.strip() for x in modified_data.split(")")])
        modified_data = ",".join([x.strip() for x in modified_data.split("&")])
        data_to_send.extend(
            [x.strip() for x in modified_data.split(",") if len(x) > 0])

    data_to_send = list([x.strip()
                        for x in data_to_send if len(x.strip()) > 0])

    return data_to_send


@logging_decorator
def create_job_profile(job_data):
    try:
        # job_data = state['job_postings']
        if type(job_data) == dict:
            job_profile = ""
            job_role = ""
            job_skills_required = list()
            job_skills_optional = list()
            role = job_data.get('role')
            responsibilities = job_data.get("responsibilities")
            qualifications = job_data.get("qualifications")
            skills_required = job_data.get("skills_required")
            skills_optional = job_data.get("skills_optional")
            additional_requirements = job_data.get("additional_requirements")
            keywords = job_data.get("keywords")

            job_profile += f"{role}|" if role is not None else "|"
            job_profile += extract_job_data(responsibilities)[0]
            job_profile += extract_job_data(qualifications)[0]
            data, skills = extract_job_data(skills_optional, "skills")
            job_profile += data
            job_skills_required.extend(clean_text_list(skills))

            data, skills = extract_job_data(skills_required, "skills")
            job_profile += data
            job_skills_optional.extend(clean_text_list(skills))

            job_profile += extract_job_data(additional_requirements)[0]
            data, skills = extract_job_data(keywords, "skills")
            job_profile += data
            job_skills_optional.extend(clean_text_list(skills))

            return job_profile, list(set(job_skills_required)), list(set(job_skills_optional)), role

        else:
            return str(job_data), list(), list(), None
    except Exception as e:
        logging.error(f"Error in creating job profile: {e}")
        log_error(f"Error in creating job profile: {e}")
        return str(job_data), list(), list(), None


@logging_decorator
async def store_jobs_in_db(jobs, db: AsyncDatabase, what: str, where: str):
    try:
        # logging.info(f"Jobs: {jobs}")
        job_profiles_to_store = list()
        job_details_to_send = list()
        for job in jobs:
            this_job_profile, this_job_skills_reqd, this_job_skills_opt, this_job_role = create_job_profile(
                job)
            job_profiles_to_store.append({
                'id': job.get("id"),
                'profile': this_job_profile,
                'skills_required': list() if len(",".join(list(this_job_skills_reqd))) == 0 else list(this_job_skills_reqd),
                'skills_optional': this_job_skills_opt,
                'role': this_job_role
            })

            job_details_to_send.append(JobPostingsModel_DB(
                id=job.get("id"),
                company=job.get("company"),
                role="".join(job.get("role")),
                location=job.get("location"),
                posted_on=job.get("posted_on"),
                contract_type=job.get("contract_type"),
                contract_time=job.get("contract_time"),
                responsibilities=[x for x in job.get(
                    "responsibilities")] if job.get('responsibilities') is not None else [],
                qualifications=[x for x in job.get("qualifications")] if job.get(
                    'qualifications') is not None else [],
                skills_required=[x for x in job.get(
                    "skills_required")] if job.get('skills_required') is not None else [],
                skills_optional=[x for x in job.get(
                    "skills_optional")] if job.get('skills_optional') is not None else [],
                additional_requirements=[x for x in job.get(
                    "additional_requirements")] if job.get('additional_requirements') is not None else [],
                keywords=job.get("keywords", []),
                redirect_url=job.get("redirect_url"),
                job_hashed=job.get("job_hashed"),
                what=what,
                where=where
            ))
        # store_job_posting(job_details_to_send, db)
        # logging.info(f"I should send: {job_details_to_send}")
        # res = store_job_profiles(job_profiles_to_store, db)
        job_id_list = await bulk_store_jobs(job_details_to_send, db)
        logging.info(f"Here, we have the job ids: {job_id_list}")
        if job_id_list is not None:
            for job_profile in job_profiles_to_store:
                job_profile['full_profile_id'] = str(
                    job_id_list.get(job_profile.get("id")))
            job_profiles_to_send = [BriefJobPostingsModel(
                **x) for x in job_profiles_to_store]
            res = await bulk_store_job_profiles(
                job_profiles=job_profiles_to_send, db=db)

            logging.info(f"Result of storing job profile in db: {res}")

            if res is not None:
                for job_profile in job_profiles_to_store:
                    store_job_profile_vector(job_id=job_profile.get("id"),
                                             job_profile=job_profile.get(
                                                 'profile'),
                                             job_role=job_profile.get('role'),
                                             skills_optional=job_profile.get(
                                                 'skills_optional'),
                                             skills_reqd=job_profile.get(
                                                 'skills_optional'),
                                             what=what,
                                             where=where,
                                             db_id=str(res.get(job_profile.get("id"))))

    except Exception as e:
        logging.error(f"Error in storing jobs in db: {e}\ninput was :{jobs}")
        log_error(f"Error in storing jobs in db: {e}\ninput was :{jobs}")


@logging_decorator
async def search_jobs_adzuna_async(state: JobSearchState, db: AsyncDatabase):
    """
    Async version: Search for jobs on Adzuna API
    """
    collected_jobs = state.get("jobs", [])
    try:
        base_url = ADZUNA_BASE_URL
        # country_code = state["country_code"]
        country_code = 'in'
        app_id = ADZUNA_APP_ID
        api_key = ADZUNA_API_KEY

        page_number = state.get("page_number", 1)

        jobs_for_api = []
        for keyword in set(state.get("keywords", [])):
            for location in set(state.get("locations", [])):
                jobs_for_api.append({"what": keyword, "where": location})

        logging.info(f"jobs_for_api: {jobs_for_api}")

        jobs_from_api = []
        timeout = aiohttp.ClientTimeout(total=30)

        # logging.info(f"cc: {country_code}")

        # logging.info(f"GHoing to serach for jobs")

        async with aiohttp.ClientSession() as session:
            for i, job in enumerate(jobs_for_api):
                logging.info(f"Jobs from api: {i}, {job}")
                what = job["what"]
                where = job["where"]
                filters_to_apply = {
                    "app_id": ADZUNA_APP_ID,
                    "app_key": ADZUNA_API_KEY,
                    "what_phrase": what,
                    "where": where,
                    "results_per_page": 10,
                    "full_time": 1,
                    "permanent": 1,
                    "max_days_old": 30,
                }

                # url = f"{base_url}{str(country_code).lower()}/search/{page_number}?app_id={app_id}&app_key={api_key}&results_per_page=10&what={what}&where={where}&content-type=application/json"
                # logging.info(f"url is: {url}")

                try:
                    async with session.get(f"{base_url}{str(country_code).lower()}/search/{page_number}", params=filters_to_apply, timeout=timeout) as response:
                        if response.status == 200:
                            data = await response.json()
                            results = data.get("results", [])
                            # jobs_from_api.extend(results)

                            # logging.info(f"{what} and {where}:\n{results}")
                            # logging.info(
                            #     "------------------------------------------")

                            # with open('adzuna_res.json', 'a') as f:
                            #     json.dump(data, f)
                            logging.info(
                                f"Found {len(results)} jobs for {what} in {where}")
                            processed_jobs = await process_jobs_concurrently(results, max_concurrent=10)

                            logging.info(
                                "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                            logging.info(
                                f"processed jobs: {processed_jobs}")
                            jobs_from_api.extend(processed_jobs)
                            if processed_jobs:
                                x = store_jobs_in_db(
                                    processed_jobs, db, what, where)
                        else:
                            logging.error(
                                f"API returned status {response.status} for {what} in {where}: {response.url}")
                            log_error(
                                f"API returned status {response.status} for {what} in {where}: {response.url}")
                except Exception as e:
                    logging.error(
                        f"Error fetching jobs for {what} in {where}: {e}")
                    log_error(
                        f"Error fetching jobs for {what} in {where}: {e}")

        if len(jobs_from_api) == 0:
            logging.info("No jobs found from API")
            return {
                "jobs": collected_jobs,
                "empty_page_count": state["empty_page_count"] + 1,
            }

        # Process all jobs concurrently
        # logging.info(f"Processing {len(jobs_from_api)} jobs concurrently...")
        # processed_jobs = await process_jobs_concurrently(jobs_from_api, max_concurrent=10)

        added = 0

        for job in jobs_from_api:
            if job is not None and job not in collected_jobs:
                collected_jobs.append(job)
                added += 1

        logging.info(f"Added {added} new jobs. Total: {len(collected_jobs)}")

        return {
            "jobs": collected_jobs,
            "empty_page_count": (state["empty_page_count"] + 1 if added == 0 else 0),
        }

    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in searching jobs on Adzuna: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in searching jobs on Adzuna: {str(e)}"
        )
        return {"jobs": collected_jobs}


@logging_decorator
def increment_page(state: JobSearchState):
    return {"page_number": state["page_number"] + 1}


@logging_decorator
def should_continue(state: JobSearchState):
    if len(state["jobs"]) >= state["threshold"]:
        return "done"

    if state["page_number"] >= state["max_pages"]:
        return "done"

    if state["empty_page_count"] >= 2:
        return "done"

    return "continue"


@logging_decorator
def infer_country(state: JobSearchState):
    """
    Extract country based on locations; later, find a way to store this mapping, so that we do not keep asking the LLM
    """
    try:
        # logging.info(
        #     f"Extracting country name and code from the given location")
        agent = create_agent(
            model=ChatOllama(model=MODEL_NAME_QWEN_SMALL, temperature=0.0),
            system_prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are an assistant that has the knowledge of the entire world",
                    }
                ]
            ),
        )

        agent_res = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        f"""Given a list of locations, infer the following, based on the locations:
                                    - country
                                    - country_code

                                    Provide one country name and country code only.

                                    return VALID JSON ONLY
                                    text: {state['locations']}"""
                    )
                ]
            }
        )

        res_json = json.loads(agent_res.get("messages", {})[-1].content)
        required_fields = ["country", "country_code"]
        for field in required_fields:
            if field not in res_json:
                raise ValueError(f"Missing required field: {field}")

        return {
            "country": res_json.get("country"),
            "country_code": res_json.get("country_code"),
        }
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in extracting entities from user input: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in extracting entities from user input: {str(e)}"
        )
        return None


@logging_decorator
def normalize_job_titles(state: JobSearchState):
    try:
        parser = JsonOutputParser(pydantic_object=JobTitleKeywordsList)
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert in job search optimization and recruitment. 
                
                    Your task is to generate the topmost effective search keyword for each job title provided. This keyword should be:
                    - Commonly used in job postings and descriptions
                    - Industry-standard terms
                    - Specific enough to find relevant positions
                    - Include variations, synonyms, or related terms
                    
                    {format_instructions}"""),
            ("human",
             "Generate the top search keyword for each of these job titles: {job_titles}")
        ])

        # logging.info(f"Fetching keywords corresponding to job titles")

        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())
        model = ChatOllama(model=MODEL_NAME, temperature=0.0)

        chain = prompt | model | parser

        agent_res = chain.invoke(
            {"job_titles": state["titles"]}
        )

        # logging.info(
        #     f"LLM returned jobs name as: {agent_res}"
        # )

        final_list = set()
        if agent_res is None:
            return {"keywords": state["titles"]}
        logging.info(f"Here,: {type(agent_res)}")

        if type(agent_res) == list:
            # logging.info("It is of type list")
            for res in agent_res:
                # logging.info(f"res: {res}")
                for keywords in res.values():
                    if type(keywords) == list:
                        for job in keywords:
                            final_list.add(job)
                    else:
                        final_list.add(keywords)

        elif agent_res.get("result") is not None:
            # logging.info("Result exist")
            for res in agent_res['result']:
                # logging.info(f"res: {res}")
                for keywords in res.values():
                    if type(keywords) == list:
                        for job in keywords:
                            final_list.add(job)
                    else:
                        final_list.add(keywords)

        return {"keywords": list(final_list)}
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in getting alternative job titles: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in getting alternative job titles: {str(e)}"
        )
        return {"keywords": state["titles"]}


@logging_decorator
def job_search_agent(db):
    """
    Create graph for looking for jobs with async support
    """
    graph = StateGraph(JobSearchState)

    graph.add_node("infer_country", infer_country)
    graph.add_node("normalize_job_titles", normalize_job_titles)

    # Wrap async function for LangGraph
    def search_jobs_wrapper(state: JobSearchState):
        return asyncio.run(search_jobs_adzuna_async(state, db=db))

    graph.add_node("search_jobs", search_jobs_wrapper)
    graph.add_node("increment_page", increment_page)

    graph.add_edge(START, "infer_country")
    graph.add_edge(START, "normalize_job_titles")

    def merge_node(state: JobSearchState) -> JobSearchState:
        return state

    graph.add_node("merge", merge_node)

    graph.add_edge("infer_country", "merge")
    graph.add_edge("normalize_job_titles", "merge")

    graph.add_edge("merge", "search_jobs")

    graph.add_conditional_edges(
        "search_jobs",
        should_continue,
        {
            "continue": "increment_page",
            "done": END,
        },
    )

    graph.add_edge("increment_page", "search_jobs")
    graph.add_edge("search_jobs", END)

    return graph


def start():
    """
    Start the process of looking for job
    """
    start_time = time.time()
    graph = job_search_agent(None)
    job_agent = graph.compile()

    entities = job_agent.invoke(
        {
            "jobs": [],
            "page_number": 1,
            "max_pages": 1,
            "threshold": 250,
            "empty_page_count": 0,
            "locations": ["Pune"],
            "titles": ["ML Engineer", "Data Scientist"],
            "country": "",
            "country_code": "",
            "keywords": [],
            "job_hashed_list": []
        }
    )

    print("Entities were: ", entities)

    jobs = entities.get("jobs")
    if jobs is None:
        print(
            f"Unfortunately, we couldn't find any jobs that matches your search criteria"
        )
        return None
    for job in jobs:
        print(job)
        print("-" * 100)
    end_time = time.time()

    with open("job_list_llama3_latest_v8.json", "w") as f:
        json.dump({"time": end_time - start_time, "jobs": jobs}, f)

    return jobs


if __name__ == "__main__":
    start_time = time.time()
    start()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
