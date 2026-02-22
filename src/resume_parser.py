import time
from typing import Optional, TypedDict, List, DefaultDict

from pydantic import ValidationError
from src.vectors.store_user_profile import store_user_profile_vector
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
from schemas.data_models import Certificate, Education, Experience, Projects, StructuredResume, JobRoles
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

import logging
import hashlib
from src.utils.error_handler import log_error
from copy import deepcopy
# from pymongo.database import Database
from pymongo.asynchronous.database import AsyncDatabase

from src.database.store_resume import store_resume, store_user_profile, update_embedding_status
from src.utils.commons import clean_text_list, get_hashed


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

        parent_dir = f'data/interim/'

        # folder_name = hashlib.sha256(resume.get(
        #     'email_id').encode('utf-8')).hexdigest()
        folder_name = get_hashed(resume.get('email_id'))
        file_name = ""

        os.makedirs(parent_dir+folder_name, exist_ok=True)
        files = list()
        with os.scandir(parent_dir+folder_name)as entries:
            for entry in entries:
                if entry.is_file():

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
        logging.info(f"Processing the resume formatted by LLM")
        user_profile = ""
        profile_summary = llm_res.get("summary", "")

        skills = ",".join(llm_res.get("skills", "")) if llm_res.get(
            "skills", "") is not None else ""
        user_profile += (
            f"skills:{skills};" if skills is not None and skills != "" else user_profile
        )

        updated_llm_res = {**llm_res}
        industry_experience_raw = llm_res.get("industry_experience", [])
        work_experience_raw = llm_res.get("work_experience")
        internship_experience = llm_res.get("internship_experience", [])
        projects = llm_res.get("projects", [])
        educations = llm_res.get("education", [])
        certificates = llm_res.get("certificates", [])

        if certificates is None or len(certificates) == 0:
            if llm_res.get('certifications') is not None:
                certificates = llm_res.get('certifications')
                updated_llm_res['certificates'] = llm_res.get('certifications')

        if industry_experience_raw is None or len(industry_experience_raw) == 0:
            if work_experience_raw is not None:
                industry_experience_raw = work_experience_raw
                updated_llm_res['industry_experience'] = work_experience_raw

        experience_data = "Experience: "
        if industry_experience_raw is not None and len(industry_experience_raw) > 0:
            for exp in industry_experience_raw:
                company = exp.get("company", "")
                role = exp.get("role", "")
                duration = exp.get("duration", "")
                is_current = exp.get("is_current", "")
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

        return {"formatted": updated_llm_res, "user_profile": user_profile}
    except Exception as e:
        logging.error(f"{type(e)}: Error in creating user  profile: {str(e)}")
        return {"formatted": llm_res, "user_profile": ""}


async def store_resume_in_db(user_resume: StructuredResume, db: AsyncDatabase, state: UserDetailsState):
    try:
        if isinstance(user_resume, dict):
            user_resume = StructuredResume(**user_resume)
        profile_summary = user_resume.summary
        skills = user_resume.skills
        internship_experience = user_resume.internship_experience
        education = user_resume.education
        projects = user_resume.projects
        industry_experience = user_resume.industry_experience
        certificates = user_resume.certificates

        prev_role = list()
        all_skills = list()

        user_profile = ""
        if profile_summary is not None:
            if type(profile_summary) == list and len(profile_summary) > 0:
                user_profile += '|'.join(profile_summary)
            elif type(profile_summary) == str and len(profile_summary) > 0:
                user_profile += profile_summary

        if skills is not None:
            if type(skills) == list and len(skills) > 0:
                user_profile += "|".join(["|".join(x.split(","))
                                         for x in skills])
                all_skills.extend([x.strip() for x in "|".join(
                    ["|".join(x.split(",")) for x in skills]).split("|")])
            elif type(skills) == str and len(skills) > 0:
                user_profile += "|".join(skills.split(","))
                all_skills.extend([skills.split(',')])
                all_skills = clean_text_list(all_skills)

        if internship_experience is not None:

            if (type(internship_experience) == list or type(internship_experience) == tuple) and len(internship_experience) > 0:
                for exp in internship_experience:
                    if isinstance(exp, Experience):
                        role = exp.role
                        prev_role.append(role)
                        summary = ""
                        if exp.summary is not None:
                            if type(exp.summary) == list:
                                summary = "|".join(exp.summary)
                            elif type(exp.summary) == str:
                                summary = "|".join(
                                    exp.summary.split("."))
                        user_profile += f"|{summary}"
            elif type(internship_experience) == str:
                summary = "|".join(internship_experience.split("."))
                user_profile += f"|{summary}"

        if industry_experience is not None:
            if (type(industry_experience) == list or type(industry_experience) == tuple) and len(industry_experience) > 0:
                for exp in industry_experience:
                    if isinstance(exp, Experience):
                        role = exp.role
                        prev_role.append(role)
                        summary = ""
                        if exp.summary is not None:
                            if type(exp.summary) == list:
                                summary = "|".join(exp.summary)
                            elif type(exp.summary) == str:
                                summary = "|".join(
                                    exp.summary.split("."))
                        user_profile += f"|{summary}"
            elif type(industry_experience) == str:
                summary = "|".join(industry_experience.split("."))
                user_profile += f"|{summary}"

        if education is not None:
            if (type(education) == list or type(education) == tuple) and len(education) > 0:
                for edu in education:
                    if type(edu) == Education:
                        degree = edu.degree
                        user_profile += f"|{degree}"
            elif type(education) == str:
                summary = "|".join(education.split("."))
                user_profile += f"|{summary}"

        if projects is not None:
            if (type(projects) == list or type(projects) == tuple) and len(projects) > 0:
                for pro in projects:
                    if type(pro) == Projects:
                        summary = ""
                        if pro.summary is not None:
                            if type(pro.summary) == list:
                                summary = "|".join(pro.summary)
                            elif type(pro.summary) == str:
                                summary = "|".join(
                                    pro.summary.split("."))
                        user_profile += f"|{summary}"
            elif type(projects) == str:
                summary = "|".join(projects.split("."))
                user_profile += f"|{summary}"

        if certificates is not None:
            if (type(certificates) == list or type(certificates) == tuple) and len(certificates) > 0:
                for cert in certificates:
                    if type(cert) == Certificate:
                        name = cert.name
                        skills = ""
                        if cert.skills is not None:
                            if type(cert.skills) == list:
                                skills = "|".join(cert.skills)
                            elif type(cert.skills) == str:
                                skills = "|".join(
                                    str(cert.skills).split("."))
                        user_profile += f"|{name}|{skills}"
            elif type(certificates) == str:
                summary = "|".join(certificates.split("."))
                user_profile += f"|{summary}"

        user_resume_to_store = user_resume.__deepcopy__()
        user_resume_to_store.skills = clean_text_list(user_resume.skills)

        user_resume_res = await store_resume(
            user_resume_to_store, db, get_hashed(str(state['resume_raw'])))

        if user_resume_res:
            user_profile_res = await store_user_profile(user_resume.email_id, user_profile,
                                                        prev_role, all_skills, db, user_resume_res)
            if user_resume_res is not None:
                vector_db_res = store_user_profile_vector(
                    user_profile, prev_role, all_skills, user_resume.email_id, db_id=user_profile_res)
                x = update_embedding_status(
                    vector_db_res, user_profile_res, db)

    except ValidationError as ve:
        logging.error(f"Error in validation: {ve}")

    except Exception as e:
        logging.error(f"Error in storing resume in db: {e}")


def parse_resume(db: AsyncDatabase, state: UserDetailsState):
    """
        Given users' resume contents in text format,
        this function will convert it into s structured data,
        and create user profile based on the resume
    """
    try:
        logging.info(f"Resume contents raw: {state}")
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
                                - internship experience: include all the experience of candidate's internship
                                - work experience: include all the experience of candidates' professional experience
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
                # logging.info(f"Intermediate response: {llm_res}")
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

                    # logging.info(f"I am going to return: {to_ret}")
                    x = store_resume_in_db(resume_formatted, db, state)

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
                """You are a senior technical hiring evaluator with 15 years of experience.
                        You've seen countless resumes and only recommend roles where you'd personally vouch for the candidate's readiness.
                        You are evaluating structured resume JSON against target roles.

                        You must behave like a deterministic scoring engine - not a resume summarizer.

                        CRITICAL RULES:

                        1. Only use information explicitly present in the provided JSON.
                        2. Do NOT assume hidden experience.
                        3. Do NOT reinterpret total experience as relevant experience.
                        4. Do NOT list generic strengths.
                        5. All strengths must be role-specific and requirement-mapped.
                        6. Calculate relevant experience using industry_experience and projects only.
                        7. Use conservative estimates if duration precision is unclear.
                        8. Relevant Experience includes:
                            - professional roles containing the job role specific responsibilities
                            - projects related to the specific job role
                        9. recommended_roles in output must contain ZERO overlap with target_roles. 
                            Cross-check every recommended role name against target_roles before including it.
                            If any overlap exists, replace it with a different role.



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

                        ROLE-SPECIFIC STRENGTH RULE

                            Each strength must:

                            - Reference a requirement of the evaluated role
                            - Cite matching evidence from structured JSON
                            - Be written as:
                            "Satisfies [requirement] via [resume evidence]"

                            No generic skill listing allowed.


                        STRICTLY FOLLOW THE PROVIDED SCHEMA TO CREATE OUTPUT. {format_instructions}
                """,
            ),
            (
                "human",
                """
                        RESUME ANALYSIS:

                        1. EXTRACT CONTEXT:
                            - Compute Total Professional Experience (from industry_experience): [X] years
                            - Current domain: [domain]
                            - Target domain: [if stated]
                            - Transition signals: [certifications, projects, courses]
                        
                        2. Compute Relevant Experience: [Y] years
                            - List relevant roles
                            - List relevant projects
                            - Assign conservative year equivalents
                            - Sum total relevant years
                            - Determine seniority band

                        3. EVALUATE USER TARGETED ROLES:
                            For each role:
                                STEP 1 - Identify Expected Requirements for this role:
                                    - Core skills relevant for this role
                                    - Expected seniority for this role
                                    - Education requirements for this role
                                    - Domain expectations

                                STEP 2 - Requirement Mapping:
                                For each major requirement:
                                    - Direct Match / Partial Match / No Match
                                    - Evidence from JSON

                                STEP 3 - Seniority Fit:
                                    - Compare required band vs calculated relevant years

                                STEP 4 - Role-Specific Strengths:
                                    Only include DIRECT MATCH requirements.
                                    Give MORE WEIGHTAGE TO SKILLS BASED ON EXPERIENCE AND PROJECTS THAN THE ONES LISTED PURELY AS SKILLS

                                STEP 5 - Gaps:
                                    - Missing skills
                                    - Missing credentials
                                    - Seniority gap
                                    - Domain gap

                                STEP 6 - Apply penalties mathematically

                                STEP 7 - Determine:
                                {{
                                "role": "",
                                "relevant_experience_years": "",
                                "seniority_fit": "",
                                "score": "",
                                "is_relevant": true/false,
                                "strengths": [],
                                "gaps": [],
                                "summary": ""
                                }}

                        4. RECOMMEND ADDITIONAL ROLES:
                            FIRST: List all user-targeted roles explicitly: {target_roles}
                            These are STRICTLY FORBIDDEN from appearing in recommendations.

                            Now recommend 2-3 COMPLETELY NEW roles that:
                                - Are NOT in this list: {target_roles}
                                - Score ≥ 0.65
                                - Leverage existing strengths, FOCUSSING ON THE ONES ACQUIRED THROUGH EXPERIENCE AND PROJECTS
                                - Are realistic next or parallel steps

                            VALIDATION BEFORE OUTPUT:
                                - For each recommended role, confirm: "Is this role in {target_roles}?"
                                - If YES → DISCARD IT and pick a different role
                                - If NO → Include it

                            For transitioning candidates, prioritize:
                                - Hybrid roles (e.g., ML Engineer for SWE → ML)
                                - Entry points (e.g., Data Scientist over Staff Data Scientist)
                            - Exclude any role already in USER TARGET ROLES.

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

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """You are a senior technical recruiter with 15 years of experience across diverse industries.
    #         You've seen countless resumes and only recommend roles where you'd personally vouch for the candidate's readiness.

    #         SENIORITY DEFINITIONS:
    #         - Entry: 0-2 years in role
    #         - Mid: 2-5 years in role
    #         - Senior: 5-8 years in role
    #         - Staff/Principal: 8+ years in role + proven technical leadership
    #         Always consider total relevant experience, not just total experience.

    #         SCORING PENALTIES:
    #         - No direct job title match: -0.20
    #         - Seniority inflation (targeting Staff with <5 years): -0.25
    #         - Career transition without projects/certs: -0.15
    #         - Missing 3+ critical skills: -0.15
    #         - Missing required credentials (degree, license, certification): -0.30
    #         - Wrong domain/industry with no transferable skills: -0.35
    #         - Total experience <2 years: cap at 0.70

    #         SCORING CALIBRATION:
    #         - 0.85-1.00: Direct experience + meets seniority expectations + all qualifications
    #         - 0.70-0.84: Good fit, minor gaps or 1 level below seniority
    #         - 0.60-0.69: Transitioning/upskilling candidate, realistic stretch
    #         - <0.60: Not ready, exclude

    #         Never exceed 0.85 unless resume shows direct role experience.

    #         STRICTLY FOLLOW THE PROVIDED SCHEMA TO CREATE OUTPUT. {format_instructions}
    #         """,
    #         ),
    #         (
    #             "human",
    #             """
    #         RESUME ANALYSIS:

    #         1. EXTRACT CONTEXT:
    #             - Total experience: [X] years
    #             - Educational background: [degrees, certifications, licenses]
    #             - Current domain/industry: [domain]
    #             - Target domain: [if stated]
    #             - Transition signals: [certifications, projects, courses]

    #         2. EVALUATE USER-TARGETED ROLES:
    #             For each role:
    #                 a) Check mandatory requirements:
    #                     - Required education (degrees, certifications, professional licenses)
    #                     - Industry-specific credentials or qualifications
    #                     - Legal/regulatory requirements for the role

    #                 b) Assess domain relevance:
    #                     - Does candidate have experience in this industry/domain?
    #                     - Are their skills transferable to this domain?
    #                     - Is this a realistic career pivot given their background?

    #                 c) Check seniority match (Entry/Mid/Senior/Staff)

    #                 d) Calculate relevant experience for THIS specific role

    #                 e) Evaluate technical/functional skills match

    #                 f) List strengths and gaps:
    #                     - STRENGTHS: Skills, experiences, qualifications that DIRECTLY support THIS role
    #                     - GAPS: Missing qualifications (education, licenses, certifications),
    #                             missing domain experience, insufficient seniority,
    #                             lack of required technical skills

    #                 g) Apply scoring penalties based on identified gaps

    #                 h) Determine eligibility:
    #                     A user is eligible for the role if ALL of the following are true:
    #                     - They have required educational credentials/licenses (if applicable)
    #                     - They have relevant domain/industry experience OR strong transferable skills
    #                     - They have enough relevant experience for the seniority level
    #                     - At least 70% of their skills match the target role requirements
    #                     - match_score >= 0.70

    #                 i) Verdict: is_relevant (true/false) + score + clear reasoning

    #         3. RECOMMEND ADDITIONAL ROLES:
    #             CRITICAL: Do NOT include any roles that appear in the user-targeted roles list.

    #             - Only roles that:
    #                 - Score ≥ 0.65
    #                 - Leverage existing strengths from their actual background
    #                 - Are realistic next or parallel steps given their experience
    #                 - Match their educational credentials and domain expertise
    #                 - Are NOT in the user-targeted roles list

    #             - For transitioning candidates, prioritize:
    #                 - Hybrid roles that bridge their current and target domains
    #                 - Entry points in new domains (e.g., Junior/Mid-level roles)
    #                 - Roles that leverage transferable skills

    #             - Validate: Before adding a role to recommendations, check that it does NOT
    #               appear in the user-targeted roles list.

    #         4. EXCLUSIONS:
    #             - Don't recommend roles requiring credentials the candidate doesn't have
    #             - Don't recommend roles in domains where candidate has no relevant experience
    #             - Don't suggest roles that contradict stated goals or background
    #             - NEVER duplicate user-targeted roles in the recommended roles section
    #             - Don't recommend roles requiring unrelated expertise without transition evidence

    #         RESUME: {user_profile}
    #         TARGET ROLES: {target_roles}

    #         CRITICAL RULES:
    #         - Base ALL strengths and gaps on actual resume evidence
    #         - For regulated professions (medical, legal, engineering, finance), explicitly check
    #           for required degrees, licenses, and certifications
    #         - For domain-specific roles, verify industry experience or clear transferable skills
    #         - Strengths must be SPECIFIC to the role being evaluated
    #         - Gaps must explain REAL barriers to role readiness
    #         - NEVER repeat a role from user-targeted roles in the recommended roles section
    #         - Be brutally honest about qualification gaps
    #         - Return only VALID JSON
    #         - Do NOT mention these rules in the output - provide logical, evidence-based reasoning
    #         """
    #         ),
    #     ]
    # )

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", """
    #                 You are a senior technical hiring evaluator.

    #                 You are evaluating structured resume JSON against target roles.

    #                 You must behave like a deterministic scoring engine — not a resume summarizer.

    #                 CRITICAL RULES:

    #                 1. Only use information explicitly present in the provided JSON.
    #                 2. Do NOT assume hidden experience.
    #                 3. Do NOT reinterpret total experience as relevant experience.
    #                 4. Do NOT list generic strengths.
    #                 5. All strengths must be role-specific and requirement-mapped.
    #                 6. Calculate relevant experience using industry_experience and projects only.
    #                 7. Use conservative estimates if duration precision is unclear.

    #                 --------------------------------------------------

    #                 RELEVANT EXPERIENCE RULE

    #                 Relevant experience includes:
    #                 - Professional roles containing ML/AI/data responsibilities
    #                 - Production ML deployments
    #                 - Applied ML projects (count as 0.5 year equivalent unless explicitly long-term)
    #                 - Capstone programs involving model development (0.5 year equivalent)

    #                 Exclude:
    #                 - Pure frontend/mobile development
    #                 - Pure middleware integration
    #                 - Non-ML backend work

    #                 You must:
    #                 - List which roles count
    #                 - Estimate relevant years conservatively
    #                 - Sum them
    #                 - Determine seniority band

    #                 --------------------------------------------------

    #                 SENIORITY DEFINITIONS (based on relevant experience only):

    #                 Entry: 0-2 years
    #                 Mid: 2-5 years
    #                 Senior: 5-8 years
    #                 Staff: 8+ years + architectural leadership

    #                 --------------------------------------------------

    #                 SCORING SYSTEM

    #                 Start at 1.00

    #                 Apply deductions:

    #                 - No direct role alignment in work history: -0.20
    #                 - Targeting >1 seniority level above relevant exp: -0.25
    #                 - Missing 3+ critical technical skills: -0.15
    #                 - Missing required credentials: -0.30
    #                 - Domain mismatch without transferable skills: -0.35
    #                 - Relevant experience below expected band minimum: -0.20
    #                 - Transition without production/project proof: -0.15

    #                 Final Score = max(0, 1.00 - penalties)

    #                 Never exceed 0.85 unless candidate has held same role title professionally.

    #                 --------------------------------------------------

    #                 ROLE-SPECIFIC STRENGTH RULE

    #                 Each strength must:

    #                 - Reference a requirement of the evaluated role
    #                 - Cite matching evidence from structured JSON
    #                 - Be written as:
    #                 "Satisfies [requirement] via [resume evidence]"

    #                 No generic skill listing allowed.

    #                 --------------------------------------------------

    #                 RECOMMENDED ROLE DISCOVERY

    #                 After evaluating user-targeted roles:

    #                 Generate 3 clusters:

    #                 Cluster A - Direct Variants
    #                 Example: ML Engineer → Applied ML Engineer, Machine Learning Engineer

    #                 Cluster B - Adjacent Technical Roles
    #                 Roles sharing ≥70% overlapping skills
    #                 Example:
    #                 - ML Engineer ↔ AI Engineer
    #                 - ML Engineer ↔ Data Scientist
    #                 - ML Engineer ↔ Applied AI Engineer

    #                 Cluster C - Bridge Roles
    #                 Hybrid roles leveraging both SWE + ML:
    #                 - AI Solutions Engineer
    #                 - AI Product Engineer
    #                 - ML Platform Engineer

    #                 Score each using same system.

    #                 Include only roles:
    #                 - Score ≥ 0.65
    #                 - Not in user-targeted list
    #                 - Realistic given relevant experience

    #                 --------------------------------------------------

    #                 Return ONLY valid JSON.
    #                 No explanations outside JSON.

    #                 """),
    #     ('human', """
    #                 STRUCTURED_RESUME_JSON:
    #                 {user_profile}

    #                 USER_TARGET_ROLES:
    #                 {target_roles}

    #                 --------------------------------------------------

    #                 TASK:

    #                 1) Compute Total Professional Experience (from industry_experience)

    #                 2) Compute Relevant ML/AI/Data Experience:
    #                 - List relevant roles
    #                 - List relevant projects
    #                 - Assign conservative year equivalents
    #                 - Sum total relevant years
    #                 - Determine seniority band

    #                 --------------------------------------------------

    #                 3) Evaluate Each User-Targeted Role

    #                 For each role:

    #                 STEP 1 - Identify Expected Requirements for this role:
    #                     - Core technical skills
    #                     - Expected seniority
    #                     - Education requirements
    #                     - Domain expectations

    #                 STEP 2 - Requirement Mapping:
    #                 For each major requirement:
    #                     - Direct Match / Partial Match / No Match
    #                     - Evidence from JSON

    #                 STEP 3 - Seniority Fit:
    #                     - Compare required band vs calculated relevant years

    #                 STEP 4 - Role-Specific Strengths:
    #                     Only include DIRECT MATCH requirements.

    #                 STEP 5 - Gaps:
    #                     - Missing skills
    #                     - Missing credentials
    #                     - Seniority gap
    #                     - Domain gap

    #                 STEP 6 - Apply penalties mathematically

    #                 STEP 7 - Determine:
    #                 {
    #                 "role": "",
    #                 "relevant_experience_years": "",
    #                 "seniority_fit": "",
    #                 "score": "",
    #                 "is_relevant": true/false,
    #                 "strengths": [],
    #                 "gaps": [],
    #                 "summary": ""
    #                 }

    #                 --------------------------------------------------

    #                 4) Recommend Additional Roles

    #                 Use cluster generation logic.

    #                 Exclude any role already in USER_TARGET_ROLES.

    #                 Return:
    #                 {
    #                 "user_targeted_roles": [...],
    #                 "recommended_roles": [...]
    #                 }

    #                 """)
    # ])

    try:
        # logging.info(f"For role analysis: {state}")
        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model=MODEL_ROLE_ANALYSIS, temperature=0.0)

        chain = prompt | model | parser
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logging.info(f"Attempt: {attempt}")
                # logging.info(f"Keys in state: {state.keys()}")
                llm_res = chain.invoke(
                    {"user_profile": state["user_profile"], "target_roles": state["target_roles"]})
                # logging.info(f"Intermediate response: {llm_res}")
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


def resume_parser_agent(db: AsyncDatabase):
    """
    Create graph defining the flow for resume parser agent
    """
    graph = StateGraph(UserDetailsState)

    def resume_parser(state: UserDetailsState):
        return parse_resume(db, state)

    graph.add_node("extract_text", action=extract_text)
    graph.add_node("parse_resume", resume_parser)
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


# def start():
#     start_time = time.time()
#     file_name = "test_resume/Priyanka_Pandey.pdf"

#     resume_agent = resume_parser_agent()
#     data = resume_agent.invoke(
#         {
#             "resume_path": file_name,
#             "resume_raw": "",
#             "resume_formatted": {},
#             "target_roles": ["Staff Data Scientist", "ML Engineer", "AI Architect"],
#             "final_role_analysis": {},
#             "user_profile": "",
#         }
#     )

#     print(data)
#     print("-" * 150)
#     print("final analysis: \n", data.get("final_role_analysis"))
#     end_time = time.time()
#     print(f"Totyal: {end_time - start_time}")


# """
# "resume_path": file_name,
#             "resume_raw": "",
#             "resume_formatted": {},
#             "matching_job_titles": [],
#             "target_roles": ["Staff Data Scientist", "ML Engineer", "AI Architect"],
#             "target_role_feedback": [],
# """
# if __name__ == "__main__":
#     start()
