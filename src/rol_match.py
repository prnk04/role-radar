'''
Evaluate users resume against job postings, and provide matching score, and reasoning
'''

from collections import defaultdict
import os
import json
import time
from typing import TypedDict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import pandas as pd
from pydantic import BaseModel

from src.utils.commons import get_hashed, logging_decorator
from src.utils.error_handler import log_error

from chroma_store import get_vector_store_jobs, get_vector_store_users

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class JobEvaluation_Model(BaseModel):
    job_id: str
    job_title: str
    match_score: float
    is_eligible: bool
    reason: str
    strengths: list[str]
    gaps: list[str]


class MatchedJobs(BaseModel):
    job_id: str
    profile: str
    score_profile: float
    skills: float
    score_skills: float
    score_total: float


class RoleResumeMatchingState(TypedDict):
    user_email_id: str
    what: list[str]
    where: list[str]
    user_profile: dict
    user_skills: dict
    matching_jobs_chroma: list
    recommended_jobs: list[JobEvaluation_Model]


@logging_decorator
def get_user_profile(state: RoleResumeMatchingState):
    try:
        logging.info(f"Inside job match: {state}")
        user_vector_store = get_vector_store_users()
        logging.info(
            f"Users in vector_store: {len(user_vector_store.get()['ids'])}")

        user_id = get_hashed(state['user_email_id'])

        user_result = user_vector_store.get(where={'user_id': user_id, }, include=[
                                            "embeddings", "metadatas", "documents"])
        grouped = defaultdict(list)
        for doc, meta, emb in zip(
            user_result["documents"],
            user_result["metadatas"],
            user_result["embeddings"]
        ):
            grouped[meta["type"]].append({
                "text": doc,
                "embedding": emb
            })

        logging.info(f"Hashed: {user_id}")

        user_profile = grouped["user_profile"][0] if grouped["user_profile"] else None
        skills = grouped["skills"][0] if grouped["skills"] else None

        logging.info(f"User id: {user_id}")
        logging.info(f"User profile: {user_profile}")
        logging.info(f"User skills: {skills}")

        return {
            "user_profile": user_profile,
            "user_skills": skills
        }

    except Exception as e:
        logging.error(f"Error in getting user profile from Chroma: {e}")
        log_error(f"Error in getting user profile from Chroma: {e}")
        return None


@logging_decorator
def get_jobs_from_chroma(state: RoleResumeMatchingState):
    try:
        jobs_vector_store = get_vector_store_jobs()
        logging.info(
            f"Jobs in vector_store: {len(jobs_vector_store.get()['ids'])}")

        logging.info(
            f"User profile is: {state['user_profile'].get('text', '')}")

        sim_job_profiles = jobs_vector_store.similarity_search_with_relevance_scores(
            query=str(state['user_profile'].get('text', '')),
            filter={"$and": [
                {"what": {"$in": state['what']}},
                {"where": {"$in": state['where']}},
                {"type": "job_profile"}
            ]},
            k=100
        )

        logging.info(f"Profile_sim: {len(sim_job_profiles)}")

        sim_skills = jobs_vector_store.similarity_search_with_relevance_scores(
            query=str(state['user_skills'].get('text', '')),
            filter={"$and": [
                {"what": {"$in": state['what']}},
                {"where": {"$in": state['where']}},
                {"type": "job_profile"}
            ]},
            k=100
        )
        logging.info(f"Profile_sim: {len(sim_skills)}")

        skills_df = pd.DataFrame(
            columns=['job_id', 'doc_id', 'skills', 'score'])

        for doc in sim_skills:
            this_doc = doc[0]
            skills_df.loc[len(skills_df)] = {"job_id": this_doc.metadata['job_id'],
                                             "doc_id": this_doc.metadata['doc_id'], "skills": this_doc.page_content, "score": doc[1]}

        profile_df = pd.DataFrame(
            columns=['job_id', 'doc_id', 'profile', 'score'])

        for doc in sim_job_profiles:
            this_doc = doc[0]
            profile_df.loc[len(profile_df)] = {"job_id": this_doc.metadata['job_id'],
                                               "doc_id": this_doc.metadata['doc_id'], "profile": this_doc.page_content, 'score': doc[1]}

        merged = pd.merge(left=profile_df, right=skills_df,  how="outer",
                          on="job_id", suffixes=("_profile", "_skills"))
        merged['score_total'] = (merged['score_profile'].fillna(
            0)*0.7) + (0.2*merged['score_skills'].fillna(0))
        merged.sort_values(by='score_total', ascending=False, inplace=True)

        matching_jobs = merged[['job_id', 'profile', 'score_profile',
                                'skills', 'score_skills', 'score_total']].to_dict(orient='records')

        logging.info(f"Dataframe: {merged.shape}")

        return {
            "matching_jobs_chroma": matching_jobs
        }

    except Exception as e:
        logging.error(f"Error in getting matching jobs from Chroma: {e}")
        log_error(f"Error in getting matching jobs from Chroma: {e}")
        return None


@logging_decorator
def get_final_response(state: RoleResumeMatchingState):
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a Senior Technical Recruiter with 20+ years of experience. 
                    Your goal is to evaluate candidates with extreme critical rigor. 

                    INTERNAL LOGIC STEPS (Perform these before scoring):
                    1. YOE CALCULATION: Current year is 2026. Calculate total years from the resume. 
                    2. ELIGIBILITY GATE: If (Job Required YOE - Candidate YOE) > 2 years, the candidate is NOT eligible (is_eligible: false).
                    3. SKILL GAP: Identify core "Must-Have" tech. If missing, the score must be below 0.50.
                    4. PROJECT WEIGHTING: Give significant weight to personal projects and certifications to bridge gaps for transitioning roles (e.g., from FE to BE).
                    5. Provide :
                        - reason: why the candidate is or is not eligible for the job. Provide your answer in one sentence
                        - strengths: why the candidate is good match for this job posting. Provide 3 key strengths.
                        - gaps: what is lacking in candidate's profile that makes them unsuitable for the job posting. Provide 3 key gaps

                    ### CRITICAL RULES:
                    1. **Evidence-Only Rule:** You may ONLY list a skill as a 'strength' if it is explicitly stated in the RESUME. Do NOT infer skills from the JOB_POSTING.
                    2. **The "Experience" Penalty:** If a job requires N+ years of experience, and the candidate has N-3 years of relevant, 'is_eligible' MUST be False. No exceptions. 
                        Example: For Principal Full Stcak Engineer, if the job posting is asking for 15 years of experince and user has 12 years of overall experience, but only 5 years in Full Stack domain, mark is_eligible as False
                    3. **The "Hallucination" Check:** Before finishing, ask yourself: "Did the candidate actually work with WhatsApp API, or is that just in the job description?" If it's not in the resume, move it to 'gaps'.
                    4. **Scoring Rigor:** - 0.80+ is reserved for perfect or near-perfect matches.
                    - If a candidate is missing a "Core Requirement", the score CANNOT exceed 0.40.

                    OUTPUT RULES:
                    - Follow {format_instructions} exactly.
                    - Return ONLY the JSON. No preamble, no explanation text.
                    - Match score and eligibility must be logically consistent.
                """,
                ),
                (
                    "human",
                    """
                    Evaluate the following candidate for the specific job role.
                    RESUME:
                    {resume}

                    JOB_POSTING:
                    {job_posting}

                    Strictly follow the formatting instructions provided in the system message.
                """,
                ),
            ]
        )

        parser = JsonOutputParser(pydantic_object=JobEvaluation_Model)
        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model="llama3.1:latest", temperature=0.0)
        chain = prompt | model | parser

        # top_10_jobs =

        # top_10_jobs = [MatchedJobs(**x).profile for x in state['matching_jobs_chroma'][:20]]
        top_10_jobs = [x.get("profile")
                       for x in state['matching_jobs_chroma'][:20]]
        recommended_jobs = list()
        for job_profile in top_10_jobs:
            llm_res = chain.invoke(
                {"resume": state['user_profile']['text'], "job_posting": job_profile})
            recommended_jobs.append(llm_res)

        logging.info(f"Recommended jobs: {recommended_jobs}")

        return {"recommended_jobs": recommended_jobs}

    except Exception as e:
        logging.error(f"Error in getting final response from LLM: {e}")
        log_error(f"Error in getting final response from LLM: {e}")
        return None


def role_resume_matching_graph():
    """
        Create an agent that will control the flow of finding suitable jobs for user
    """

    graph = StateGraph(RoleResumeMatchingState)
    graph.add_node("get_user_profile", get_user_profile)
    graph.add_node("get_matching_jobs_chroma", get_jobs_from_chroma)
    graph.add_node("get_final_response", get_final_response)

    graph.add_edge(START, "get_user_profile")
    graph.add_edge("get_user_profile", "get_matching_jobs_chroma")
    graph.add_edge("get_matching_jobs_chroma", "get_final_response")
    graph.add_edge("get_final_response", END)

    return graph.compile()
