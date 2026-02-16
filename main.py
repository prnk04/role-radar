from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request

from pydantic import BaseModel
import uuid
import logging
from src.utils.error_handler import log_error
from src.orchestration import start_agent, orchestrateAgent

from typing import Dict, Any
from copy import deepcopy
import httpx

SESSION_STORE: Dict[str, Dict[str, Any]] = {}


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Server starting")
    yield
    print("🛑 Server shutting down")

app = FastAPI(lifespan=lifespan)


class ResumeReviewRequest(BaseModel):
    resume_path: str
    roles: list | None
    locations: list


class JobSearch(BaseModel):
    job_list: list
    session_id: str


def create_session(state: dict) -> str:
    session_id = str(uuid.uuid4())
    SESSION_STORE[session_id] = deepcopy(state)
    return session_id


def load_session(session_id: str) -> dict:
    if session_id not in SESSION_STORE:
        raise ValueError("Invalid or expired session_id")

    return deepcopy(SESSION_STORE[session_id])


def save_session(session_id: str, state: dict) -> None:
    if session_id not in SESSION_STORE:
        raise ValueError("Invalid or expired session_id")

    SESSION_STORE[session_id] = deepcopy(state)


def delete_session(session_id: str) -> None:
    SESSION_STORE.pop(session_id, None)


@app.get("/healthCheck")
def healthCheck() -> dict[str, str]:
    return {"healthCheck": "ok"}


@app.post("/review")
def reviewResume(resume_review_request: ResumeReviewRequest):
    try:
        resume_path = resume_review_request.resume_path
        roles = resume_review_request.roles
        locations = resume_review_request.locations

        state = {
            "resume_file_path": resume_path,
            "location": locations,
            "targetted_roles": roles,
            "starting_point": "resume_step",
        }

        if resume_path != "" and len(locations) > 0:
            # agent_res = start_agent(
            #     {
            #         "resume_path": resume_path,
            #         "roles": roles,
            #         "locations": locations,
            #     }
            # )
            orchestrator = orchestrateAgent(state)

            if orchestrator is None:
                raise HTTPException(
                    status_code=500,
                    detail="Something went wrong. Please try again later.",
                )

            agent_res = orchestrator.invoke(
                state, config={"entry_point": "resume_step"}
            )

            logging.info(f"Agent res: {agent_res}")

            session_id = create_session(
                agent_res) if agent_res is not None else ""

            logging.info(f"agent_res: {agent_res}")
            res_to_send = (
                agent_res.get("suggested_roles")
                if agent_res is not None
                else "I will show you later"
            )
            return {"message": res_to_send, "session_id": session_id}

        else:
            raise Exception("Missing required data")
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in starting resume parser agent: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in starting resume parser agent: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail="Something went wrong. Please try again later."
        )


@app.post("/searchJobs")
def searchJobs(job_list: JobSearch):
    try:
        state = load_session(job_list.session_id)
        jobs = job_list.job_list
        state["selected_roles"] = jobs
        state["starting_point"] = "job_fetch_step"

        orchestrator = orchestrateAgent(state)

        agent_res = (
            orchestrator.invoke(
                state, config={"entry_point": "job_fetch_step"})
            if orchestrator is not None
            else None
        )

        if agent_res is not None:
            save_session(job_list.session_id, agent_res)

        logging.info(f"agent_res: {agent_res}")
        res_to_send = (
            agent_res.get("searched_jobs")
            if agent_res is not None
            else "I will show you later"
        )
        recs = agent_res.get(
            "recommended_postings") if agent_res is not None else "Uh oh!"
        return {"recommended": recs}
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in starting resume parser agent: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in starting resume parser agent: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail="Something went wrong. Please try again later."
        )
