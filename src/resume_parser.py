from typing import Optional, TypedDict, List, DefaultDict
from src.utils.document_loader import DocumentLoader
from pathlib import Path
import os
import json
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import ollama

import logging

from src.utils.error_handler import log_error

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:8b")
llm = ChatOllama(model=MODEL_NAME)


class UserDetailsState(TypedDict):
    resume_path: str
    resume_raw: Optional[str]
    resume_formatted: Optional[dict]
    target_roles: Optional[List[str]]
    final_role_analysis: Optional[dict]


def parse_resume(state: UserDetailsState):
    try:
        resume_parsing_agent = create_agent(
            model=llm,
            system_prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": "You are an assistant at an organisation that actively reviews and evaluates candidates' resume. You perform the tasks assign to you flawlessly",
                    }
                ]
            ),
        )

        agent_res = resume_parsing_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"""
                                You are an expert in reviewing, and analysing resume. You are given contents of a resume in text format.Your task includes:
                                    - reviewing the resume contents
                                    - notifying about the presence of any PII(Personally Identifiable Information), and categorise them
                                    - extract following information from the resume:
                                        - summary section from the resume. If user has provided summary, or objective, extract that
                                        - skills
                                        - experience: Extract list of experience of the format: 
                                            {{
                                                "company:"",
                                                "position": "",
                                                "start_date":"",
                                                "end_date":"",
                                                "duration": <calculate this from end date and start date>,
                                                "is_this_current_role":"True/False",
                                                "role_and_responsibilities":<list_of_roles_and_responsibilities"
                                            }}
                                        - education: Extract list of education provided
                                        - projects: Extract list of personal projects, along with their names, links(if provided), summary.
                                        - Certifications
                                The final JSON data that you should return, should be follow the following schema:
                                {{
                                    'summary':str,
                                    'skills': list(),
                                    'experience': list(),
                                    'education: list(),
                                    'projects': list(),
                                    'certifications': list()
                                }}
                                Base your result on the provided information only. Always return valud JSON.

                                resume_text: {state['resume_raw']}
                                
                            """
                    )
                ]
            }
        )

        ai_message = str(agent_res.get("messages", [])[-1].content)
        formatted_data = json.loads(ai_message)
        return {"resume_formatted": formatted_data}
    except Exception as e:
        logging.error(f"Type of error: {type(e)}\nError in parsing resume: {str(e)}")
        log_error(f"Type of error: {type(e)}\nError in parsing resume: {str(e)}")


def role_analysis(state: UserDetailsState):
    content = ""
    try:
        agent = create_agent(
            model=llm,
            system_prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": """
                                    You are a senior technical recruiter with 15 years of experience 
                                    placing ML/DS candidates. You've seen countless resumes and only recommend roles where 
                                    you'd personally vouch for the candidate's readiness.

                                    SENIORITY DEFINITIONS:
                                    - Entry: 0-2 years in role
                                    - Mid: 2-5 years in role
                                    - Senior: 5-8 years in role  
                                    - Staff/Principal: 8+ years in role + proven technical leadership
                                    Always consider total relevant experience, and not just total experience

                                    SCORING PENALTIES:
                                    - No direct job title match: -0.20
                                    - Seniority inflation (targeting Staff with <5 years): -0.25
                                    - Career transition without projects/certs: -0.15
                                    - Missing 3+ critical skills: -0.15
                                    - Total experience <2 years: cap at 0.70

                                    SCORING CALIBRATION:
                                    - 0.85-1.00: Direct experience + meets seniority expectations
                                    - 0.70-0.84: Good fit, minor gaps or 1 level below seniority
                                    - 0.60-0.69: Transitioning/upskilling candidate, realistic stretch
                                    - <0.60: Not ready, exclude

                                    Never exceed 0.85 unless resume shows direct role experience.
                                """,
                    }
                ]
            ),
        )

        res = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"""
                                    RESUME ANALYSIS:

                                    1. EXTRACT CONTEXT:
                                    - Total experience: [X] years
                                    - Current domain: [domain]
                                    - Target domain: [if stated]
                                    - Transition signals: [certifications, projects, courses]

                                    2. EVALUATE USER-TARGETED ROLES:
                                    For each role:
                                    - Check seniority match (Entry/Mid/Senior/Staff)
                                    - Calculate relevant experience for THIS role
                                    - List 2 strengths + 2 gaps
                                    - Apply scoring penalties
                                    - Verdict: makes_sense (yes/no) + score
                                    

                                    3. RECOMMEND ADDITIONAL ROLES:
                                    - Only roles that:
                                        * Score ≥ 0.65
                                        * Leverage existing strengths
                                        * Are realistic next steps
                                        * Don't duplicate user-targeted roles
                                    
                                    - For transitioning candidates, prioritize:
                                        * Hybrid roles (e.g., ML Engineer for SWE → ML)
                                        * Entry points (e.g., Data Scientist over Staff Data Scientist)

                                    4. EXCLUSIONS:
                                    - Don't recommend roles requiring unrelated expertise (PM, Marketing, Sales)
                                    - Don't suggest roles that contradict stated goals

                                    RESUME: {state['resume_formatted']}
                                    TARGET ROLES: {state.get('target_roles', [])}

                                    If the resume does not provide enough evidence to confidently recommend a role, you MUST exclude it.

                                    OUTPUT FORMAT (STRICT JSON):

                                    {{
                                    "user_targeted_roles_feedback": [
                                        {{
                                        "job_title": str,
                                        "makes_sense": bool,
                                        "score": float,
                                        "reason": str
                                        }}
                                    ],
                                    "additional_recommended_roles": [
                                        {{
                                        "job_title": str,
                                        "score": float,
                                        "reason": str
                                        }}
                                    ],
                                    "summary_narrative": str
                                    }}

                                    RULES:
                                    - Do NOT repeat roles across sections
                                    - Base decisions ONLY on resume evidence
                                    - Be honest, not aspirational
                                    - DO NOT mention the rules that we have used. Just a logical reasoning
                                    - Return only VALID JSON

                                    resume: {state['resume_formatted']}
                                    user_target_roles: {state.get('target_roles', [])}
                            """
                    )
                ]
            }
        )

        content = res["messages"][-1].content
        return {"final_role_analysis": json.loads(content)}

    except Exception as e:
        logging.error(f"Error in role analysis: {e}")
        log_error(f"Error in role analysis: content: {content}; error: {str(e)}")
        return {}


def job_role_mapping(state: UserDetailsState):
    try:
        job_title_agent = create_agent(
            model=llm,
            system_prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": """You are a skeptical hiring manager who has been burned by inflated resumes. 
                                    You only give high scores (>0.85) when evidence is overwhelming and undeniable.
                                    Your goal is to conservatively assess which job titles a candidate is realistically suitable for based only on evidence in the resume. "
                                    You prefer under-recommending over over-recommending.""",
                    }
                ]
            ),
        )

        title_matching_agent_res = job_title_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"""
                                You are an expert in reviewing, and analysing resume. You are given contents of a resume in JSON format.Your task includes:
                                    - reviewing the resume contents
                                    - create user profile based on the resume contents
                                    - finding out job titles that best match user profile
                                    - For each job title:
                                        1. First, decide if the candidate is eligible(yes/no):
                                            - If no, do not include the title
                                        2. If yes, compute a match score
                                    Match score calibration rules:
                                        - 0.90-1.00 → Strong match, candidate meets almost all expectations for this role
                                        - 0.75-0.89 → Good match, some gaps but role is realistic
                                        - 0.60-0.74 → Partial match, role may require upskilling
                                        - <0.60 → Do NOT include
                                    Penalise scores if:
                                        - No direct job title match in experience
                                        - Missing 3+ key skills for role
                                        - Less than 2 years in related work
                                        - Fresher/no experience, unless perfect skill match
                                    Do NOT give scores above 0.9 unless the resume clearly demonstrates direct experience in that role.

                                    - return top-5 such titles, along with:
                                        - the confidence score, on a scale of 0.0-1.0
                                        - reason as to why those titles match users profile. 
                                    Each reason MUST include:
                                        - 1-2 strengths that support the match
                                        - 1 explicit gap or limitation relevant to that role

                                    If the resume does not provide enough evidence to confidently recommend a role, you MUST exclude it.
                                        
                                    
                                The final JSON data that you should return, should be follow the following schema:
                                {[{
                                    'job_title': str,
                                    'match_score': float,
                                    'reason': str
                                }]}
                                Base your result on the provided information only. Always return valud JSON.

                                resume_text: {state['resume_formatted']}
                                
                            """
                    )
                ]
            }
        )

        ai_res = str(title_matching_agent_res.get("messages", [])[-1].content)
        formatted_job_titles = json.loads(ai_res)

        return {"matching_job_titles": formatted_job_titles}

    except Exception as e:
        logging.error(f"Type of error: {type(e)}\nError in mapping job role: {str(e)}")
        log_error(f"Type of error: {type(e)}\nError in mapping job role: {str(e)}")


def evaluate_target_roles(state: UserDetailsState):
    try:
        if not state.get("target_roles"):
            return {}

        agent = create_agent(
            model=llm,
            system_prompt=SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": """You are a brutally honest career advisor.
                                You explain clearly WHY a role makes sense or does not, based strictly on resume evidence.
                                You do not sugarcoat gaps, but you are constructive.
                            """,
                    }
                ]
            ),
        )

        res = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"""
                        You are given:
                        1. Parsed resume (JSON)
                        2. Job titles the user is targeting

                        For EACH target role:
                        - Decide if the role makes sense (yes/no)
                        - Explain WHY
                        - Cite resume evidence
                        - Call out missing skills / experience explicitly
                        - Suggest what would improve eligibility (brief)

                        Return JSON ONLY in this schema:
                        [
                        {{
                            "job_title": str,
                            "makes_sense": bool,
                            "reason": str,
                            "key_gaps": list,
                            "improvement_suggestions": list
                        }}
                        ]

                        resume: {state['resume_formatted']}
                        target_roles: {state['target_roles']}
                        """
                    )
                ]
            }
        )

        content = res["messages"][-1].content
        return {"target_role_feedback": json.loads(content)}

    except Exception as e:
        logging.error(f"Error evaluating target roles: {e}")
        log_error(str(e))
        return {}


def extract_text(state: UserDetailsState):
    try:
        dl = DocumentLoader()
        file_name = Path(state["resume_path"])
        file_contents = dl.load(file_name)
        text_to_parse = file_contents.get("text")

        return {"resume_raw": text_to_parse}
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in extracting text from resume: {str(e)}"
        )
        log_error(
            f"Type of error: {type(e)}\nError in extracting text from resume: {str(e)}"
        )
        return None


def shouldParseResume(state: UserDetailsState):
    job_text_raw = state["resume_raw"]
    if job_text_raw is None or job_text_raw == "":
        return False
    return True


def shouldMapJobs(state: UserDetailsState) -> bool:
    parsed = state.get("resume_formatted")
    if not parsed or not isinstance(parsed, dict):
        return False

    # Minimal signal check (tweak as needed)
    required_keys = ["skills", "experience", "education"]
    return any(parsed.get(k) for k in required_keys)


def resume_parser_agent():
    """
    Create graph defining the flow for resume parser agent
    """
    graph = StateGraph(UserDetailsState)

    graph.add_node("extract_text", action=extract_text)
    graph.add_node("parse_resume", parse_resume)
    graph.add_node("role_analysis", role_analysis)
    # graph.add_node("map_jobs", job_role_mapping)

    # graph.add_node("evaluate_target_roles", evaluate_target_roles)

    graph.set_entry_point("extract_text")
    # # graph.add_edge("extract_text", "parse_resume")
    # graph.add_conditional_edges(
    #     "extract_text",
    #     shouldParseResume,
    #     {
    #         True: "parse_resume",
    #         False: END,
    #     },
    # )

    # # graph.add_edge("parse_resume", "map_jobs")
    # graph.add_conditional_edges(
    #     "parse_resume",
    #     shouldMapJobs,
    #     {
    #         True: "map_jobs",
    #         False: END,
    #     },
    # )

    # graph.add_edge("map_jobs", "evaluate_target_roles")
    # graph.add_edge("evaluate_target_roles", END)

    graph.add_conditional_edges(
        "extract_text", shouldParseResume, {True: "parse_resume", False: END}
    )

    graph.add_conditional_edges(
        "parse_resume", shouldMapJobs, {True: "role_analysis", False: END}
    )

    graph.add_edge("role_analysis", END)

    return graph


def start():
    file_name = "test_resume/Priyanka_Pandey.pdf"

    graph = resume_parser_agent()
    resume_agent = graph.compile()

    data = resume_agent.invoke(
        {
            "resume_path": file_name,
            "resume_raw": "",
            "resume_formatted": {},
            "target_roles": ["Staff Data Scientist", "ML Engineer", "AI Architect"],
            "final_role_analysis": {},
        }
    )

    print(data)
    print("-" * 150)
    print("final analysis: \n", data.get("final_role_analysis"))


"""
"resume_path": file_name,
            "resume_raw": "",
            "resume_formatted": {},
            "matching_job_titles": [],
            "target_roles": ["Staff Data Scientist", "ML Engineer", "AI Architect"],
            "target_role_feedback": [],
"""
if __name__ == "__main__":
    start()
