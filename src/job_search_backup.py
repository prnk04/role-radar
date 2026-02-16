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
import logging

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


def get_structured_data(data):
    """
    Fetches company name, job title, roles and responsibilities, skills from the HTML input received from the redirect URL
    :param data: section part of the HTML that contains details about the job posting
    """
    try:
        user_prompt = f"""I have an HTML snippet taken from a job posting. 
            Your task is to convert prase that string, extract HTML content from it, and provide me important information from it. The text in question is {data}
            The information that I want:
            - Company name: name of the company that posted the job listing
            - Company description: about the company
            - roles & responsibilities
            - skills required
            - skills good to have
            - what will the candidate do
            - any additional notes or benefits or perks, and categorise them
            - salary, if provided

            Provide the response in the form of valid json only
        """

        input_list = [
            {
                "role": "system",
                "content": "You are an expert at extracting meaningful information from unstructured text that was written in html, but was stringified.",
            },
            {"role": "user", "content": user_prompt},
        ]

        ollama_model = "qwen3:8b"

        response_message = ollama.chat(
            model=MODEL_NAME,
            messages=input_list,
            options={"temperature": 0.0},
        )
        response_contents = str(response_message.message.content)

        final_res = json.loads(response_contents)
        logging.info(f"Received structured job posting details from the LLM")
        return final_res

    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in getting structured data: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in getting structured data: {str(e)}"
        )
        return None


def extract_job_requirements(job_data):
    """
    Extract details of job posting from the result of calling Adzuna API

    :param job_data: Adzuna API response
    """
    job_details_to_show = dict()
    try:
        company_name = job_data.get("display_name", "NA")
        job_title = job_data.get("title", "NA")
        description = job_data.get("description", "NA")
        employment_type = job_data.get("contract_time", "NA")
        job_location = job_data.get("location", {}).get("display_name", "NA")
        posted_at = job_data.get("createdAt", "NA")
        contract_type = job_data.get("contract_type", "NA")
        more_details_url = job_data.get("redirect_url", "NA")
        adRef = job_data.get("adref", "NA")

        job_details_to_show = {
            "company_1": company_name,
            "job_title": job_title,
            "job_location": job_location,
            "posted_at": posted_at,
            "employment_type": employment_type,
            "contract_type": contract_type,
        }

        job_from_link = requests.get(url=more_details_url)
        if job_from_link.status_code != 200:
            log_error(
                f"url: {more_details_url}, response: {job_from_link}; adRef: {adRef}"
            )
            if job_from_link.status_code == 403:
                job_details_to_show["redirect_url"] = more_details_url
                return job_details_to_show
            return None

        job_html = bs4.BeautifulSoup(job_from_link.text, features="html.parser")
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

        contract_type_placeholder = job_html.find("div", class_="ui-contract-type")
        contract_type_option = (
            contract_type_placeholder.getText(":").strip()
            if contract_type_placeholder is not None
            else ""
        )

        contract_time_placeholder = job_html.find("div", class_="ui-contract-time")
        contract_time_option = (
            contract_time_placeholder.getText(":").strip()
            if contract_time_placeholder is not None
            else ""
        )

        job_details_to_show["company_name_option"] = company_name_option
        job_details_to_show["locations_option"] = locations_option
        job_details_to_show["contract_type_option"] = contract_type_option
        job_details_to_show["contract_time_option"] = contract_time_option

        if len(job_details) == 0:
            return job_details_to_show

        job_requirements = get_structured_data(job_details[0])
        # print("job req: ", job_requirements)

        if job_requirements is not None:
            for k, v in job_requirements.items():
                job_details_to_show[k] = v
        logging.info(f"Fetched job details from the redirect URL")
        return job_details_to_show

    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in extracting job requirements: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in extracting job requirements: {str(e)}"
        )
        return job_details_to_show


# def extract_entities(state: JobSearchState):
#     """
#     Extract entities like job_title, location, country, country code from prompt written in natural language
#     """
#     try:
#         agent = create_agent(
#             model=llm,
#             # ChatOllama(model=MODEL_NAME),
#             system_prompt=SystemMessage(
#                 content=[
#                     {
#                         "type": "text",
#                         "text": "You are an assistant that has the knowledge of the entire world",
#                     }
#                 ]
#             ),
#         )

#         agent_res = agent.invoke(
#             {
#                 "messages": [
#                     HumanMessage(
#                         f"""Given an unstructured text, extract the following entites:
#                                     - job_title,
#                                     - location,
#                                     - country,
#                                     -country_code
#                                     From the location, find out country and country code as well
#                                     Return the answer in the form of a valid JSON response only
#                                     text: {state['user_query']}"""
#                     )
#                 ]
#             }
#         )

#         res_json = json.loads(agent_res.get("messages", {})[-1].content)
#         required_fields = ["job_title", "location", "country", "country_code"]
#         for field in required_fields:
#             if field not in res_json:
#                 raise ValueError(f"Missing required field: {field}")
#         # print("and json res: ", res_json)
#         logging.info(
#             f"Extracted entities from user message: {agent_res.get("messages", {})[-1].content}"
#         )

#         return {
#             "title": res_json.get("title"),
#             "location": res_json.get("location"),
#             "country": res_json.get("country"),
#             "country_code": res_json.get("country_code"),
#         }
#     except Exception as e:
#         logging.error(
#             f"Type of error: {type(e)}\nError in extracting entities from user input: {str(e)}"
#         )
#         log_error(
#             f"Type of error: {type(e)}\nError in extracting entities from user input: {str(e)}"
#         )
#         return None


def search_jobs_adzuna(state: JobSearchState):
    """
    Given the search criteria, look for jobs from Adzuna
    """

    logging.info(f"Satte:  {state['keywords']}")
    collected_jobs = state.get("jobs", [])
    page_num = state["page_number"]
    initial_count = len(collected_jobs)

    job_whats = state.get("titles")
    print("length of job what: ", len(job_whats))
    print("type of job: ", type(job_whats))
    print("jon whats is: ", job_whats)
    job_whats = [] if job_whats is None else job_whats

    print("jon whats is: ", job_whats)

    print("and: ", " ".join(job_whats))

    job_wheres = state.get("locations", [""])[0]
    job_wheres = "" if job_wheres is None else job_wheres

    try:
        filters_to_apply = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_API_KEY,
            "what": " ".join(job_whats),
            "where": job_wheres,
            "results_per_page": 10,
            "full_time": 1,
            "permanent": 1,
            "max_days_old": 60,
        }

        search_url = f"{ADZUNA_BASE_URL}{str(state.get('country_code', '')).lower()}/search/{page_num}"

        logging.info(f"Searching for jobs in page number: {page_num}")
        logging.info(f"Searching for jobs with url: {search_url}")

        job_res = requests.get(url=search_url, params=filters_to_apply)
        logging.info(f"We called URL: {job_res.url}, and response was: {job_res}")
        logging.info(f"Text: {job_res.text}")
        logging.info(f"josn: {job_res.json()}")
        job_res_json = job_res.json()

        for res in job_res_json.get("results", []):
            job = extract_job_requirements(res)
            if job:
                collected_jobs.append(job)

        added = len(collected_jobs) - initial_count

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


def enrich_data(state: JobSearchState):
    """
    Format JSON for final output
    """
    enriched = []
    try:
        logging.info(f"Started enriching the job details")
        logging.info(f"Jobs are: {state.get("jobs")}")
        if state.get("jobs") is None:
            return state.get("jobs")
        for job in state.get("jobs", []):

            enrichment_agent = create_agent(
                model=llm,
                # ChatOllama(model=MODEL_NAME),
                system_prompt=SystemMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "You are an expert in JSON data structure",
                        }
                    ]
                ),
            )

            result = enrichment_agent.invoke(
                {
                    "messages": [
                        HumanMessage(
                            f"""
                            You are given a JSON data, with multiple fields. The data might contain multiple fields with the same name, or options for the same field.
                            Clean, and restructure the json, and return the result in valid JSON format only.
                            The JSON must include:
                            - company
                            - job_title
                            - job_location
                            - posted_at
                            - employment_type
                            - contract_type
                            - company_description
                            - roles_and_responsibilities
                            - skills_required
                            - skills_good_to_have
                            - additional_notes
                            - salary

                            Return VAILD JSON ONLY
                            input: {job}
                            """
                        )
                    ]
                }
            )

            valid_json = json.loads(result.get("messages", {})[-1].content)
            enriched.append(valid_json)
        return {"jobs": enriched}
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in enriching job data: {str(e)}"
        )
        log_error(f"Type of error: {type(e)}\nError in enriching job data: {str(e)}")
        return {"jobs": enriched}


def infer_country(state: JobSearchState):
    """
    Extract country based on locations; later, find a way to store this mapping, so that we do not keep asking the LLM
    """
    try:
        agent = create_agent(
            model=llm,
            # ChatOllama(model=MODEL_NAME),
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
        logging.info(f"Locations were: {state['locations']}")

        logging.info(
            f"LLM returned country name as: {agent_res.get("messages", {})[-1].content}"
        )

        res_json = json.loads(agent_res.get("messages", {})[-1].content)
        required_fields = ["country", "country_code"]
        for field in required_fields:
            if field not in res_json:
                raise ValueError(f"Missing required field: {field}")
        # print("and json res: ", res_json)
        logging.info(
            f"Extracted entities from user message: {agent_res.get("messages", {})[-1].content}"
        )

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


def job_search_agent():
    """
    Create graph for looking for jobs
    """
    graph = StateGraph(JobSearchState)

    # graph.add_node("extract_entities", extract_entities)
    graph.add_node("infer_country", infer_country)
    graph.add_node("normalize_job_titles", normalize_job_titles)
    graph.add_node("search_jobs", search_jobs_adzuna)
    graph.add_node("increment_page", increment_page)
    graph.add_node("enrich_jobs", enrich_data)

    # graph.add_edge(START, "infer_country")
    # graph.add_edge(START, "normalize_job_titles")

    # def merge_node(state: JobSearchState) -> JobSearchState:
    #     return state

    # graph.add_node("merge", merge_node)

    # graph.add_edge("infer_country", "merge")
    # graph.add_edge("normalize_job_titles", "merge")

    # graph.set_entry_point("extract_entities")
    # graph.set_entry_point("extract_country")

    # graph.set_entry_point("normalize_job_titles")

    # graph.add_edge("extract_entities", "search_jobs")
    # graph.add_edge("merge", "search_jobs")

    graph.add_edge(START, "infer_country")
    graph.add_edge("infer_country", "search_jobs")

    graph.add_conditional_edges(
        "search_jobs",
        should_continue,
        {
            "continue": "increment_page",
            "done": "enrich_jobs",
        },
    )

    graph.add_edge("increment_page", "search_jobs")
    graph.add_edge("enrich_jobs", END)

    return graph


def normalize_job_titles(state: JobSearchState):
    try:
        agent = create_agent(
            model=llm,
            # ChatOllama(model=MODEL_NAME),
            system_prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are a recruiter and a career advisor. You complete the tasks given to you without any error",
                    }
                ]
            ),
        )

        agent_res = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        f"""You are given a list of job titles. For EACH job title, provide a list of 3 VALID keywords to search for.
                                    return VALID JSON CONTAINING 3 VALID keywords corresponding to job titles ONLY
                                    text: {state["titles"]}"""
                    )
                ]
            }
        )
        logging.info(f"jobs were: {state["jobs"]}")

        logging.info(
            f"LLM returned jobs name as: {agent_res.get("messages", {})[-1].content}"
        )

        final_list = set()

        # final_list.extend(x for x in agent_res.get("messages", {})[-1].content)
        res_json = json.loads(agent_res.get("messages", {})[-1].content)
        logging.info(f"First, we have: {res_json}")

        for job, keywords in res_json.items():
            final_list.add(job)
            if type(keywords) == list:
                for job in keywords:
                    final_list.add(job)
            else:
                final_list.add(keywords)

        print("res_json: ", res_json)
        return {"keywords": list(final_list)}
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in getting alternative job titles: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in getting alternative job titles: {str(e)}"
        )
        return {"keywords": state["titles"]}


def start():
    """
    Start the process of looking for job
    """
    graph = job_search_agent()
    job_agent = graph.compile()

    # entities = job_agent.invoke(
    #     {
    #         "user_query": "I am looking for role of data scientist in Pune",
    # "jobs": [],
    # "page_number": 1,
    # "max_pages": 1,
    # "threshold": 20,
    # "empty_page_count": 0,
    # "location": "",
    # "title": "",
    # "country": "",
    # "country_code": "",
    #     }
    # )

    entities = job_agent.invoke(
        {
            "jobs": [],
            "page_number": 1,
            "max_pages": 1,
            "threshold": 20,
            "empty_page_count": 0,
            # "locations": ["Bangalore", "Pune"],
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

    with open("job_list.txt", "w") as f:
        json.dump(jobs, f)
    return jobs


if __name__ == "__main__":
    # user_in = input("Enter your search criteria:\n")
    # job_titles = ["ML Engg", "DS"]
    # get_alternative_titles(job_titles)
    start()
