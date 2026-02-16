import time
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

from schemas.agent_states import UserDetailsState
from schemas.data_models import StructuredResume, JobRoles
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import logging
import hashlib
from src.utils.error_handler import log_error
from copy import deepcopy

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:8b")
MODEL_RESUME_PARSING = os.getenv("MODEL_RESUME_PARSING", "llama3.1:latest")
MODEL_ROLE_ANALYSIS = os.getenv("MODEL_ROLE_ANALYSIS", "qwen3:1.7b")
# llm = ChatOllama(model=MODEL_NAME)


def store_resume_in_file(resume):
    ''' 
        store user's resume in a josn file
    '''
    try:
        logging.info(f"Inside store resume")
        parent_dir = f'data/interim/'
        logging.info(f"Resume: {resume}")
        logging.info(
            "----------------------------------------------------------------------------------------")
        folder_name = hashlib.sha256(resume.get(
            'email_id').encode('utf-8')).hexdigest()
        file_name = ""

        os.makedirs(parent_dir+folder_name, exist_ok=True)
        files = list()
        with os.scandir(parent_dir+folder_name)as entries:
            for entry in entries:
                if entry.is_file():
                    print(entry.name.split(".")[0][1:])
                    files.append(int(entry.name.split(".")[0][1:]))

        if len(files) == 0:
            file_name = f'{parent_dir}{folder_name}/v0.json'
        else:
            file_name = f"{parent_dir}{folder_name}/v{sorted(files)[-1] + 1}.json"

        resume_copy = deepcopy(resume)
        del resume_copy['email_id']

        with open(file_name, "w") as f:
            json.dump(resume_copy, f)

    except Exception as e:
        logging.error(f"Error in storing resume as JSON file: {str(e)}")
        log_error(f"Error in storing resume as JSON file: {str(e)}")


def resume_post_processing(llm_res: dict):
    try:
        logging.info(f"I am here for post processing:\n{llm_res}")
        logging.info(f"type: {type(llm_res)}")
        user_profile = ""
        profile_summary = llm_res.get("summary", "")

        logging.info("joining skills")

        skills = ",".join(llm_res.get("skills", "")) if llm_res.get(
            "skills", "") is not None else ""
        user_profile += (
            f"skills:{skills};" if skills is not None and skills != "" else user_profile
        )
        industry_experience_raw = llm_res.get("industry_experience", [])
        internship_experience = llm_res.get("internship_experience", [])
        projects = llm_res.get("projects", [])
        educations = llm_res.get("education", [])
        certificates = llm_res.get("certificates", [])

        experience_data = "Experience: "
        if industry_experience_raw is not None and len(industry_experience_raw) > 0:
            for exp in industry_experience_raw:
                company = exp.get("company", "")
                role = exp.get("role", "")
                duration = exp.get("duration", "")
                is_current = exp.get("is_current", "")
                logging.info("joining industry exp summary")
                summary = ";".join(exp.get("summary", "")) if exp.get(
                    "summary", "") is not None else ""
                str1 = ""
                str1 += f"{company}|" if company is not None and company != "" else str1
                str1 += f"{role}|" if role is not None and role != "" else str1
                str1 += f"{duration}|" if duration is not None and duration != "" else str1
                str1 += (
                    f"is_current:{is_current}|"
                    if is_current is not None and is_current != ""
                    else str1
                )
                str1 += (
                    f"responsibilities:{summary}|"
                    if summary is not None and summary != ""
                    else str1
                )
                experience_data += (
                    f"{str1};"
                    if str1 is not None and len(str1.strip()) > 0
                    else experience_data
                )
            user_profile += (
                f"{experience_data}"
                if experience_data is not None and len(experience_data.strip()) > 15
                else ""
            )

        internship_data = "Internships: "
        if internship_experience is not None and len(internship_experience) > 0:
            for exp in internship_experience:
                company = exp.get("company", "")
                role = exp.get("role", "")
                duration = exp.get("duration", "")
                is_current = exp.get("is_current", "")
                logging.info("joining industry exp summary")
                summary = ";".join(exp.get("summary", "")) if exp.get(
                    "summary", "") is not None else ""
                str1 = ""
                str1 += f"{company}|" if company is not None and company != "" else str1
                str1 += f"{role}|" if role is not None and role != "" else str1
                str1 += f"{duration}|" if duration is not None and duration != "" else str1
                str1 += (
                    f"is_current:{is_current}|"
                    if is_current is not None and is_current != ""
                    else str1
                )
                str1 += (
                    f"responsibilities:{summary}|"
                    if summary is not None and summary != ""
                    else str1
                )
                internship_data += (
                    f"{str1};"
                    if str1 is not None and len(str1.strip()) > 0
                    else internship_data
                )
            user_profile += (
                f"{internship_data}"
                if internship_data is not None and len(internship_data.strip()) > 15
                else ""
            )

        projects_data = "Projects: "
        if projects is not None and len(projects) > 0:
            for pro in projects:
                name = pro.get("name", "")
                logging.info("joining projects summary")
                about = ";".join(pro.get("summary", "")) if pro.get(
                    "summary", "") is not None else ""
                str1 = ""
                str1 += f"{name}|" if name is not None and name != "" else str1
                str1 += f"{about}|" if about is not None and about != "" else str1
                projects_data += (
                    f"{str1};" if str1 is not None and len(
                        str1.strip()) > 0 else projects_data
                )
            user_profile += (
                f"{projects_data}"
                if projects_data is not None and len(projects_data.strip()) > 15
                else ""
            )

        education_data = "Education: "
        if educations is not None and len(educations) > 0:
            for education in educations:
                degree = education.get("degree", "")
                major = education.get("major", "")
                is_current = education.get("is_current", "")

                str1 = ""
                str1 += f"{degree}|" if degree is not None and degree != "" else str1
                str1 += f"{major}|" if major is not None and major != "" else str1
                str1 += (
                    f"{is_current}|" if is_current is not None and is_current != "" else str1
                )
                education_data += (
                    f"{str1};" if str1 is not None and len(
                        str1.strip()) > 0 else education_data
                )
            user_profile += (
                f"{education_data}"
                if education_data is not None and len(education_data.strip()) > 15
                else ""
            )

        certification_data = "Certificates: "
        if certificates is not None and len(certificates) > 0:
            for certificate in certificates:
                certificate_name = certificate.get("name", "")
                issuing_auth = certificate.get("issuing_authority", "")
                skills = ";".join(certificate.get("skills", [])) if certificate.get(
                    "skills", []) is not None else ""

                str1 = ""
                str1 += (
                    f"{issuing_auth}:"
                    if issuing_auth is not None and issuing_auth != ""
                    else str1
                )
                str1 += (
                    f"{certificate_name}:"
                    if issuing_auth is not None and issuing_auth != ""
                    else str1
                )
                str1 += f"{skills}:" if skills is not None and skills != "" else str1
                certification_data += (
                    f"{str1};"
                    if str1 is not None and len(str1.strip()) > 0
                    else certification_data
                )
            user_profile += (
                f"{certification_data}"
                if certification_data is not None and len(certification_data.strip()) > 15
                else ""
            )
        user_profile += (
            f"Summary: {profile_summary}"
            if profile_summary is not None and profile_summary != ""
            else ""
        )

        # writing resume to the file
        logging.info(f"Going to store the resume")
        store_resume_in_file(llm_res)

        return {"formatted": llm_res, "user_profile": user_profile}
    except Exception as e:
        logging.error(f"{type(e)}: Error in creating user  profile: {str(e)}")
        return {"formatted": llm_res, "user_profile": ""}


def parse_resume(state: UserDetailsState):
    """
        Given users' resume contents in text format,
        this function will convert it into s structured data,
        and create user profile based on the resume
    """
    try:
        parser = JsonOutputParser(pydantic_object=StructuredResume)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an assistant at an organisation that actively reviews and evaluates candidates' resume. 
                        You perform the tasks assign to you flawlessly.
                        You always follow the output schema provided to you.{format_instructions}""",
                ),
                (
                    "human",
                    """
                        You are an expert in reviewing, and analysing resume. You are given contents of a resume in text format.Your task includes:
                            - reviewing the resume contents
                            - extract following information from the resume:
                                - email id
                                - summary section from the resume. If user has provided summary, or objective, extract that
                                - skills
                                - internship experience
                                - work experience
                                - for internship and work experience, calculate the duration using the start date and end date provided
                                - education: Extract list of education provided
                                - projects: Extract list of personal projects, along with their names, links(if provided), summary.
                                - Certifications
                            
                        
                        Base your result on the provided information only. Always return VALID JSON consistent with the given schema.

                        resume_text: {input}
                                
                    """,
                ),
            ]
        )

        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model=MODEL_RESUME_PARSING, temperature=0.0)

        # chain = prompt | model | parser  # ← Added parser here!

        chain = prompt | model | parser | RunnableLambda(
            resume_post_processing)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logging.info(f"Attempt: {attempt}")
                llm_res = chain.invoke({"input": str(state["resume_raw"])})
                logging.info(f"Intermediate response: {llm_res}")
                # parsed_llm_response = parser.parse(llm_res)
                if llm_res is None:
                    continue
                else:
                    user_profile = llm_res.get("user_profile", "")
                    resume_formatted = llm_res.get("formatted", {})
                    user_profile = (
                        str(resume_formatted)
                        if user_profile is None or len(user_profile.strip()) == 0
                        else user_profile
                    )
                    to_ret = {
                        "resume_formatted": resume_formatted,
                        "user_profile": user_profile,
                    }

                    logging.info(f"I am going to return: {to_ret}")

                    return {
                        "resume_formatted": resume_formatted,
                        "user_profile": user_profile,
                    }
            except Exception as e:
                logging.error(f"Error in invoking resume: {e}")
                if attempt == max_attempts - 1:
                    return None

        return None
    except Exception as e:
        logging.error(
            f"Type of error: {type(e)}\nError in parsing resume: {str(e)}")
        log_error(
            f"Type of error: {type(e)}\nError in parsing resume: {str(e)}")


def role_analysis(state: UserDetailsState):
    content = ""
    parser = JsonOutputParser(pydantic_object=JobRoles)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a senior technical recruiter with 15 years of experience placing ML/DS candidates. 
                        You've seen countless resumes and only recommend roles where you'd personally vouch for the candidate's readiness.

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

                        STRICTLY FOLLOW THE PROVIDED SCHEMA TO CREATE OUTPUT. {format_instructions}
                """,
            ),
            (
                "human",
                """
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
                                - Check relevant skills for THIS role
                                - List 2 strengths + 2 gaps
                                - Apply scoring penalties
                                - A user is eligible for the role if:
                                    - they have enough relevant experience in the domain
                                    - at least 80% of their skills matches with the target role requirements
                                    - match_score >= 0.7
                                - Verdict: is_eligible (yes/no) + score
                        

                        3. RECOMMEND ADDITIONAL ROLES:
                            - Only roles that:
                                - Score ≥ 0.65
                                - Leverage existing strengths
                                - Are realistic next or parallel steps
                                - Don't duplicate user-targeted roles
                                - Are distinct from the user-targeted roles
                            
                            - For transitioning candidates, prioritize:
                                - Hybrid roles (e.g., ML Engineer for SWE → ML)
                                - Entry points (e.g., Data Scientist over Staff Data Scientist)

                        4. EXCLUSIONS:
                            - Don't recommend roles requiring unrelated expertise (PM, Marketing, Sales)
                            - Don't suggest roles that contradict stated goals
                            - Do not duplicate user-targeted roles, and recommended roles. 

                        RESUME: {user_profile}
                        TARGET ROLES: {target_roles}

                        If the resume does not provide enough evidence to confidently recommend a role, you MUST exclude it.

                        RULES:
                        - Do NOT repeat roles across sections
                        - Base decisions ONLY on resume evidence
                        - Be honest, not aspirational
                        - DO NOT mention the rules that we have used. Just a logical reasoning
                        - Return only VALID JSON
                        - ALWAYS RETURN STRENGTHS, AND GAPS CORRESPONDING TO A ROLE, BASED ON USER PROFILE
                    """,
            ),
        ]
    )
    try:
        logging.info(f"For role analysis: {state}")
        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model=MODEL_ROLE_ANALYSIS, temperature=0.0)

        chain = prompt | model | parser
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logging.info(f"Attempt: {attempt}")
                logging.info(f"Keys in state: {state.keys()}")
                llm_res = chain.invoke(
                    {"user_profile": state["user_profile"], "target_roles": state["target_roles"]})
                logging.info(f"Intermediate response: {llm_res}")
                # parsed_llm_response = parser.parse(llm_res)
                if llm_res is None:
                    continue
                else:
                    return {
                        "final_role_analysis": llm_res,
                    }
            except Exception as e:
                logging.error(f"Error in invoking role analysis: {e}")
                if attempt == max_attempts - 1:
                    return None

        return None

    except Exception as e:
        logging.error(f"Error in role analysis: {e}")
        # log_error(f"Error in role analysis: content: {content}; error: {str(e)}")
        return {}


def extract_text(state: UserDetailsState):
    try:
        dl = DocumentLoader()
        file_name = Path(state["resume_path"])
        file_contents = dl.load(file_name)
        text_to_parse = file_contents.get("text")
        logging.info(f"Extracted text contents of the resume")

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

    graph.set_entry_point("extract_text")

    graph.add_conditional_edges(
        "extract_text", shouldParseResume, {True: "parse_resume", False: END}
    )

    graph.add_conditional_edges(
        "parse_resume", shouldMapJobs, {True: "role_analysis", False: END}
    )

    graph.add_edge("role_analysis", END)

    return graph.compile()


def start():
    start_time = time.time()
    file_name = "test_resume/Priyanka_Pandey.pdf"

    resume_agent = resume_parser_agent()
    data = resume_agent.invoke(
        {
            "resume_path": file_name,
            "resume_raw": "",
            "resume_formatted": {},
            "target_roles": ["Staff Data Scientist", "ML Engineer", "AI Architect"],
            "final_role_analysis": {},
            "user_profile": "",
        }
    )

    print(data)
    print("-" * 150)
    print("final analysis: \n", data.get("final_role_analysis"))
    end_time = time.time()
    print(f"Totyal: {end_time - start_time}")


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
