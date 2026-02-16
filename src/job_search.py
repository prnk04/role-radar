import asyncio
import time
import aiohttp
import requests
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


import logging

from pydantic import BaseModel
from src.utils.error_handler import log_error

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
ADZUNA_API_KEY = os.getenv("ADZUNA_API_KEY")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_BASE_URL = os.getenv("ADZUNA_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:8b")

llm = ChatOllama(model=MODEL_NAME)


class JobSearchState(TypedDict):
    titles: List
    locations: List
    country: Optional[str]
    country_code: Optional[str]
    # user_query: str
    jobs: List[dict]
    page_number: int
    max_pages: int
    threshold: int
    empty_page_count: int
    keywords: List


class SkillsModel(BaseModel):
    must_have: list[str]
    good_to_have: list[str]


class JobPostingsModel(BaseModel):
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
    redirect_url: str


class JobTitleKeywords(BaseModel):
    job_title: str
    keywords: str


class JobTitleKeywordsList(BaseModel):
    result: list[JobTitleKeywords]


async def get_structured_data(data):
    """
    Fetches company name, job title, roles and responsibilities, skills from the HTML input received from the redirect URL
    :param data: section part of the HTML that contains details about the job posting
    """
    try:
        parser = JsonOutputParser(pydantic_object=JobPostingsModel)
        prompt = ChatPromptTemplate.from_messages([
            ("system",  """
                            You are an expert in extracting meaningful and structured data from unstructured data.
                            You have worked on countless HTML snippets, and JSON data that has multiple fields conveying the same meaning.
                            You STRICTLY FOLLOW THE FORMAT GIVEN TO YOU TO convert the unstructured data, into meaningful JSON data.
                            STRICTLY FOLLOW THE PROVIDED SCHEMA TO CREATE OUTPUT. {format_instructions}
                        """),
            ("human", """
                        You are given 2 inputs:
                            1. an HTML code snippet, extracted from an internet JOB POSTING, that contains the following information:
                                - company name
                                - company description
                                - responsibilities
                                - required skills
                                - optional skills
                                - day in the life of an employee
                                - additional information about benefits, or job role requirements
                                - salary(at times)
                            2. A JSON object that:
                                - also has information regarding the job posting, but partial,
                                - and it contains fields having different names, but conveying the same value/meaning.
                                - the field company_description_from_api:
                                    - may contain company description, or some other information like role, responsibilities, etc.
                                    - Extract all the details, and use them to populate the JSON.
             
                        Your task is:
                            - From the HTML snippet for job posting:
                                - extract the mentioned information from the HTML snippet
                                - if any infromation is not present, infer it from other fields. But do not make anything up
                            - Combine the information that you extracted from the HTML code snippet and from the given JSON object:
                                - Remove duplicates
                            - Strictly follow the given schema to create a VALID JSON object.

                        HTML_SNIPPET: {html_snippet}
                        json_object: {job_posting_partial}
                            
                        """)
        ])

        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model=MODEL_NAME, temperature=0.0)

        chain = prompt | model | parser
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logging.info(f"Attempt: {attempt}")

                # llm_res = chain.invoke(
                #     {"html_snippet": data["html_snippet"], "job_posting_partial": data["job_posting_partial"]})
                # Run LLM call in executor to avoid blocking
                loop = asyncio.get_event_loop()
                llm_res = await loop.run_in_executor(
                    None,
                    lambda: chain.invoke({
                        "html_snippet": data["html_snippet"],
                        "job_posting_partial": data["job_posting_partial"]
                    })
                )
                logging.info(f"Intermediate response: {llm_res}")
                return llm_res
            except Exception as e:
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


async def extract_data_from_description(data):
    """
    Infers data from the description field of the API result
    :param data: description field of the API result
    """
    try:
        parser = JsonOutputParser(pydantic_object=JobPostingsModel)
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
             
                        
                        - Strictly follow the given schema to create a VALID JSON object.

                        text: {text}
                            
                        """)
        ])

        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model=MODEL_NAME, temperature=0.0)

        chain = prompt | model | parser
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logging.info(f"Attempt: {attempt}")

               # Run LLM call in executor to avoid blocking
                loop = asyncio.get_event_loop()
                llm_res = await loop.run_in_executor(
                    None,
                    lambda: chain.invoke({
                        "text": data
                    })
                )
                logging.info(f"Intermediate response: {llm_res}")
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


async def fetch_redirect_url_async(session: aiohttp.ClientSession, url: str, timeout: int = 10):
    """
    Async function to fetch content from redirect URL
    """
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
            if response.status != 200:
                logging.error(
                    f"Redirect URL returned status {response.status}: {url}")
                if response.status == 403:
                    return {"status": 403, "url": url}
                return None

            text = await response.text()
            return {"status": 200, "text": text, "url": url}
    except asyncio.TimeoutError:
        logging.error(f"Timeout fetching redirect URL: {url}")
        log_error(
            f"Asyncio timeout error: Error in getting fetching redirect URL async: {url}"
        )
        return None
    except Exception as e:
        logging.error(f"Error fetching redirect URL {url}: {e}")
        log_error(
            f"Type of error: {type(e)}\nError in fetching redirect URL: {str(e)}"
        )
        return None


async def extract_job_requirements(job_data, session: aiohttp.ClientSession):
    """
    Extract details of job posting from the result of calling Adzuna API

    :param job_data: Adzuna API response
    """
    jobs_to_send = dict()
    try:
        company_name = job_data.get("display_name", "NA")
        job_title = job_data.get("title", "NA")
        description = job_data.get("description", "NA")
        contract_time = job_data.get("contract_time", "NA")
        job_location = job_data.get("location", {}).get("display_name", "NA")
        posted_at = job_data.get("createdAt", "NA")
        contract_type = job_data.get("contract_type", "NA")
        more_details_url = job_data.get("redirect_url", "NA")
        adRef = job_data.get("adref", "NA")

        job_details_to_show = {
            "company_from_api": company_name,
            "job_title_from_api": job_title,
            "job_location_from_api": job_location,
            "posted_at_from_api": posted_at,
            "contract_time_from_api": contract_time,
            "contract_type_from_api": contract_type,
            "company_description_from_api": description,
            "redirect_url": more_details_url
        }

        jobs_to_send = {
            "company": company_name,
            "role": job_title,
            "location": job_location,
            "posted_on": posted_at,
            "contract_type": contract_type,
            "contract_time": contract_time,
            "company_description": description,
            "redirect_url": more_details_url
        }

        # Async fetch of redirect URL
        redirect_result = await fetch_redirect_url_async(session, more_details_url)

        if redirect_result is None:
            return None

        if redirect_result.get("status") == 403:
            job_details_to_show["redirect_url"] = more_details_url
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
            else ""
        )

        contract_time_placeholder = job_html.find(
            "div", class_="ui-contract-time")
        contract_time_option = (
            contract_time_placeholder.getText(":").strip()
            if contract_time_placeholder is not None
            else ""
        )

        job_details_to_show["company_name_from_posting"] = company_name_option
        job_details_to_show["locations_from_posting"] = locations_option
        job_details_to_show["contract_type_from_posting"] = contract_type_option
        job_details_to_show["contract_time_from_posting"] = contract_time_option

        if len(job_details) == 0:
            return jobs_to_send

        job_requirements = await get_structured_data(
            data={"html_snippet": job_details[0], "job_posting_partial": job_details_to_show})

        logging.info(f"Fetched job details from the redirect URL")
        return job_requirements

    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in extracting job requirements: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in extracting job requirements: {str(e)}"
        )
        return jobs_to_send


async def process_jobs_concurrently(jobs_list: List[dict], max_concurrent: int = 5):
    """
    Process multiple job postings concurrently with a concurrency limit

    :param jobs_list: List of job data from Adzuna API
    :param max_concurrent: Maximum number of concurrent requests
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(job_data, session):
        async with semaphore:
            return await extract_job_requirements(job_data, session)

    async with aiohttp.ClientSession() as session:
        tasks = [process_with_semaphore(job, session) for job in jobs_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None results and exceptions
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            logging.error(f"Exception during job processing: {result}")
        elif result is not None:
            valid_results.append(result)

    return valid_results


async def search_jobs_adzuna(state: JobSearchState):
    """
    Given the search criteria, look for jobs from Adzuna
    """

    logging.info(f"Satte:  {state['keywords']}")
    collected_jobs = state.get("jobs", [])
    page_num = state["page_number"]
    initial_count = len(collected_jobs)

    job_whats = state.get("keywords")
    logging.info(f"We have keywords for job search as: {job_whats}")
    job_whats = [] if job_whats is None else job_whats

    job_wheres = state.get("locations", [""])[0]
    job_wheres = "" if job_wheres is None else job_wheres

    try:
        filters_to_apply = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_API_KEY,
            "what_or": " ".join(job_whats),
            "where": job_wheres,
            "results_per_page": 10,
            "full_time": 1,
            "permanent": 1,
            "max_days_old": 90,
        }

        search_url = f"{ADZUNA_BASE_URL}{str(state.get('country_code', '')).lower()}/search/{page_num}"

        logging.info(f"Searching for jobs in page number: {page_num}")
        logging.info(f"Searching for jobs with url: {search_url}")

        jobs_from_api = []
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(search_url, params=filters_to_apply) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get("results", [])
                        jobs_from_api.extend(results)
                        logging.info(
                            f"Found {len(results)} jobs {response.url}")
                    else:
                        logging.error(
                            f"API returned status {response.status} for {search_url}")
            except Exception as e:
                logging.error(f"Error fetching jobs for {search_url}: {e}")

        if len(jobs_from_api) == 0:
            logging.info("No jobs found from API")
            return {
                "jobs": collected_jobs,
                "empty_page_count": state["empty_page_count"] + 1,
            }

        # Process all jobs concurrently
        logging.info(f"Processing {len(jobs_from_api)} jobs concurrently...")
        processed_jobs = await process_jobs_concurrently(jobs_from_api, max_concurrent=10)

        added = 0
        for job in processed_jobs:
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


def increment_page(state: JobSearchState):
    return {"page_number": state["page_number"] + 1}


def should_continue(state: JobSearchState):
    if len(state["jobs"]) >= state["threshold"]:
        return "done"

    if state["page_number"] >= state["max_pages"]:
        return "done"

    if state["empty_page_count"] >= 2:
        return "done"

    return "continue"


def infer_country(state: JobSearchState):
    """
    Extract country based on locations; later, find a way to store this mapping, so that we do not keep asking the LLM
    """
    try:
        logging.info(
            f"Extracting country name and code from the given location")
        agent = create_agent(
            model=llm,
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

        logging.info(f"Fetching keywords corresponding to job titles")

        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())
        model = ChatOllama(model=MODEL_NAME, temperature=0.0)

        chain = prompt | model | parser

        agent_res = chain.invoke(
            {"job_titles": state["titles"]}
        )

        logging.info(
            f"LLM returned jobs name as: {agent_res}"
        )

        final_list = set()
        if agent_res is None:
            return {"keywords": state["titles"]}
        logging.info("going to loop")
        for res in agent_res['result']:
            logging.info(f"res: {res}")
            for keywords in res.values():
                # final_list.add(job)
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


def job_search_agent_old():
    """
    Create graph for looking for jobs
    """
    graph = StateGraph(JobSearchState)

    # graph.add_node("extract_entities", extract_entities)
    graph.add_node("infer_country", infer_country)
    graph.add_node("normalize_job_titles", normalize_job_titles)
    graph.add_node("search_jobs", search_jobs_adzuna)
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


def job_search_agent():
    """
    Create graph for looking for jobs with async support
    """
    graph = StateGraph(JobSearchState)

    graph.add_node("infer_country", infer_country)
    graph.add_node("normalize_job_titles", normalize_job_titles)

    # Wrap async function for LangGraph
    def search_jobs_wrapper(state: JobSearchState):
        return asyncio.run(search_jobs_adzuna(state))

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
    graph = job_search_agent()
    job_agent = graph.compile()

    entities = job_agent.invoke(
        {
            "jobs": [],
            "page_number": 1,
            "max_pages": 1,
            "threshold": 20,
            "empty_page_count": 0,
            "locations": ["Pune"],
            "titles": ["ML Engineer", "Data Scientist"],
            "country": "",
            "country_code": "",
            "keywords": [],
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

    logging.info(f"Time taken: {end_time - start_time}")

    with open("job_list.txt", "w") as f:
        json.dump(jobs, f)
    return jobs


if __name__ == "__main__":
    # user_in = input("Enter your search criteria:\n")
    # job_titles = ["ML Engg", "DS"]
    # get_alternative_titles(job_titles)
    start()
