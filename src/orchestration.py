from typing import List, Optional, TypedDict, Any

from langgraph.graph import StateGraph, END, START

import logging
from src.rol_match import role_resume_matching_graph
from src.utils.error_handler import log_error

from src.resume_parser import resume_parser_agent
from src.job_search_async import job_search_agent
from src.role_resume_matcher import role_resume_matching_agent

from pymongo.database import Database

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OrchestrationState(TypedDict):
    # user_input
    resume_file_path: str
    targetted_roles: List | None
    location: List
    user_query: str

    # resume_parser
    resume_contents: dict | None
    suggested_roles: List | None

    # user input
    selected_roles: List
    starting_point: str

    # searched_jobs
    searched_jobs: List

    # recommended_postings
    recommended_postings: Any


def parse_resume(db: Database, state: OrchestrationState):
    try:
        resume_parser = resume_parser_agent(db=db)

        resume_parser_res = resume_parser.invoke(
            {
                "resume_path": state["resume_file_path"],
                "resume_raw": "",
                "resume_formatted": {},
                "target_roles": state["targetted_roles"],
                "final_role_analysis": {},
                "user_profile": ""
            }
        )

        return {
            "resume_contents": resume_parser_res.get("resume_formatted"),
            "suggested_roles": resume_parser_res.get("final_role_analysis"),
        }

    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in starting resume parser agent: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in starting resume parser agent: {str(e)}"
        )


def job_fetch_step(state: OrchestrationState, db):
    js_agent = job_search_agent(db).compile()
    result = js_agent.invoke(
        {

            "jobs": [],
            "page_number": 1,
            "max_pages": 1,
            "threshold": 20,
            "empty_page_count": 0,
            # "locations": ["Bangalore", "Pune"],
            "locations": state["location"],
            "titles": state["selected_roles"],
            "country": "",
            "country_code": "",
            "keywords": [],
            "job_hashed_list": []
        }
    )
    return {"searched_jobs": result["jobs"]}


def role_resume_match(state: OrchestrationState):
    role_resume_match_agent = role_resume_matching_graph()
    result = role_resume_match_agent.invoke({
        "matching_jobs_chroma": [],
        "recommended_jobs": [],
        "user_email_id": state['resume_contents'].get("email_id", '') if state['resume_contents'] is not None else "",
        "user_profile": {},
        "user_skills": {},
        "where": state['location'],
        "what": state['selected_roles']
    })

    logging.info(f"Hey!\n{result["recommended_jobs"]}")
    return {
        "recommended_postings": result["recommended_jobs"]
    }


def orchestrateAgent(db: Database):
    try:

        graph = StateGraph(OrchestrationState)

        def get_resume_parser_agent(state):
            return parse_resume(db, state=state)

        def get_job_search_agent(state):
            return job_fetch_step(state=state, db=db)

        graph.add_node("resume_step", get_resume_parser_agent)
        graph.add_node("job_fetch_step", get_job_search_agent)
        graph.add_node("job_profile_matching", role_resume_match)

        # graph.set_entry_point("parse_resume")
        graph.add_edge("resume_step", END)
        graph.add_edge("job_fetch_step", "job_profile_matching")
        graph.add_edge("job_profile_matching", END)
        # graph.add_edge("job_fetch_step", END)

        def get_stage(state: OrchestrationState):
            logging.info(f"Starting: {state}")
            return state["starting_point"]

        # graph.add_edge(START, "resume_step")
        graph.set_conditional_entry_point(get_stage)

        return graph.compile()
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in orchestrating agents: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError orchestrating agents: {str(e)}")
        return None


def start_agent(db: Database, details):
    file_path = details.get("resume_path")
    roles = details.get("roles")
    locations = details.get("locations")
    session_id = details.get("session_id")

    orchestrator = orchestrateAgent(db)

    if orchestrator is None:
        print("Orchestrator is none. not sure why")
        return None
    res = orchestrator.invoke(
        {
            "resume_file_path": file_path,
            "resume_contents": dict(),
            "suggested_roles": [],
            "user_query": "",
            "location": locations,
            "targetted_roles": roles,
            "selected_roles": [],
            "starting_point": "",
            "searched_jobs": [],
            "recommended_postings": None
        }
    )

    logging.info("Returned response form orchestration agent")

    return res
