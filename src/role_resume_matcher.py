'''
Evaluate users resume against job postings, and provide matching score, and reasoning
'''

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


class JobEvaluation_Model(BaseModel):
    job_id: str
    job_title: str
    match_score: float
    is_eligible: bool
    reason: str
    strengths: list[str]
    gaps: list[str]


model = SentenceTransformer("all-MiniLM-L6-v2")


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RoleResumeMatchingState(TypedDict):
    resume_file_path: str
    job_posting_file_path: str
    job_postings: list[dict[str, str]]
    user_resume: dict | None
    inital_evaluation: Any
    final_evaluation: Any
    user_prev_roles: list[str] | None
    user_skills: list[str] | None
    user_profile: str


def extract_jobs(state: RoleResumeMatchingState):
    all_job_postings = None
    with open(state['job_posting_file_path'], "r") as f:
        all_job_postings = json.load(f)
    return {
        "job_postings": all_job_postings.get("jobs") if all_job_postings is not None else None
    }


def extract_resume(state: RoleResumeMatchingState):
    user_resume = None
    with open(state['resume_file_path'], "r") as f:
        user_resume = json.load(f)
    return {
        "user_resume": user_resume if user_resume is not None else None
    }


def create_user_profile(state: RoleResumeMatchingState):
    try:
        user_resume = state['user_resume']
        profile_summary = user_resume.get("summary", [])
        skills = user_resume.get("skills", [])
        internship_experience = user_resume.get("internship_experience", [])
        work_experience = user_resume.get("work_experience", [])
        education = user_resume.get("education", [])
        projects = user_resume.get("projects", [])
        certifications = user_resume.get("certifications", [])
        industry_experience = user_resume.get("industry_experience", [])
        certificates = user_resume.get("certificates", [])

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

        if work_experience is not None:
            if (type(work_experience) == list or type(work_experience) == tuple) and len(work_experience) > 0:
                for exp in work_experience:
                    if type(exp) == dict:
                        role = exp.get('role', '')
                        prev_role.append(role)
                        summary = ""
                        if exp.get('summary') is not None:
                            if type(exp.get('summary')) == list:
                                summary = "|".join(exp.get('summary', ''))
                            elif type(exp.get('summary')) == str:
                                summary = "|".join(
                                    exp.get('summary', '').split("."))
                        user_profile += f"|{summary}"
            elif type(work_experience) == str:
                summary = "|".join(work_experience.split("."))
                user_profile += f"|{summary}"

        if internship_experience is not None:
            if (type(internship_experience) == list or type(internship_experience) == tuple) and len(internship_experience) > 0:
                for exp in internship_experience:
                    if type(exp) == dict:
                        role = exp.get('role', '')
                        prev_role.append(role)
                        summary = ""
                        if exp.get('summary') is not None:
                            if type(exp.get('summary')) == list:
                                summary = "|".join(exp.get('summary', ''))
                            elif type(exp.get('summary')) == str:
                                summary = "|".join(
                                    exp.get('summary', '').split("."))
                        user_profile += f"|{summary}"
            elif type(internship_experience) == str:
                summary = "|".join(internship_experience.split("."))
                user_profile += f"|{summary}"

        if industry_experience is not None:
            if (type(industry_experience) == list or type(industry_experience) == tuple) and len(industry_experience) > 0:
                for exp in industry_experience:
                    if type(exp) == dict:
                        role = exp.get('role', '')
                        prev_role.append(role)
                        summary = ""
                        if exp.get('summary') is not None:
                            if type(exp.get('summary')) == list:
                                summary = "|".join(exp.get('summary', ''))
                            elif type(exp.get('summary')) == str:
                                summary = "|".join(
                                    exp.get('summary', '').split("."))
                        user_profile += f"|{summary}"
            elif type(industry_experience) == str:
                summary = "|".join(industry_experience.split("."))
                user_profile += f"|{summary}"

        if education is not None:
            if (type(education) == list or type(education) == tuple) and len(education) > 0:
                for edu in education:
                    if type(edu) == dict:
                        degree = edu.get('degree', '')
                        user_profile += f"|{degree}"
            elif type(education) == str:
                summary = "|".join(education.split("."))
                user_profile += f"|{summary}"

        if projects is not None:
            if (type(projects) == list or type(projects) == tuple) and len(projects) > 0:
                for pro in projects:
                    if type(pro) == dict:
                        summary = ""
                        if pro.get('summary') is not None:
                            if type(pro.get('summary')) == list:
                                summary = "|".join(pro.get('summary', ''))
                            elif type(pro.get('summary')) == str:
                                summary = "|".join(
                                    pro.get('summary', '').split("."))
                        user_profile += f"|{summary}"
            elif type(projects) == str:
                summary = "|".join(projects.split("."))
                user_profile += f"|{summary}"

        if certificates is not None:
            if (type(certificates) == list or type(certificates) == tuple) and len(certificates) > 0:
                for cert in certificates:
                    if type(cert) == dict:
                        name = cert.get('name', '')
                        skills = ""
                        if cert.get('skills') is not None:
                            if type(cert.get('skills')) == list:
                                skills = "|".join(cert.get('skills', ''))
                            elif type(cert.get('skills')) == str:
                                skills = "|".join(
                                    str(cert.get('skills')).split("."))
                        user_profile += f"|{name}|{skills}"
            elif type(certificates) == str:
                summary = "|".join(certificates.split("."))
                user_profile += f"|{summary}"

        if certifications is not None:
            if (type(certifications) == list or type(certifications) == tuple) and len(certifications) > 0:
                for cert in certifications:
                    if type(cert) == dict:
                        name = cert.get('name', '')
                        skills = ""
                        if cert.get('skills') is not None:
                            if type(cert.get('skills')) == list:
                                skills = "|".join(cert.get('skills', ''))
                            elif type(cert.get('skills')) == str:
                                skills = "|".join(
                                    str(cert.get('skills')).split("."))
                        user_profile += f"|{name}|{skills}"
            elif type(certifications) == str:
                summary = "|".join(certifications.split("."))
                user_profile += f"|{summary}"

        return {
            "user_profile": user_profile,
            "user_prev_roles": prev_role,
            "user_skills": all_skills}
    except Exception as e:
        logging.error(f"Error occurred in creating user profile: {e}")
        return {
            "user_profile": str(state['user_profile']),
            "user_prev_roles": None,
            "user_skills": None}


def extract_job_data(data, extra=None):
    try:
        formatted_data = ""
        extra_return = None
        if extra == "skills":
            pass

        if data is not None:
            if type(data) == dict:
                formatted_data += "|".join(["|".join(x.split(","))
                                           for x in data.values() if x is not None])

            elif type(data) == list:
                this_res = ""
                for res in data:
                    if type(res) == dict:
                        this_res += "|".join(["|".join(x.split(","))
                                             for x in res.values() if x is not None])
                        this_res += "|"
                    elif type(res) == list:
                        this_res += "|".join(["|".join(x.split(","))
                                             for x in res if x is not None])
                        this_res += "|"

                    elif type(res) == str:
                        this_res += "".join(["|".join(x.split(","))
                                            for x in res if x is not None])
                        this_res += "|"
                formatted_data += this_res

            elif type(data) == str:
                formatted_data += "|".join(data.split(","))
                formatted_data += "|"
        return formatted_data, formatted_data.split("|")
    except Exception as e:
        logging.error(f"Error occurred in extracting job data: {e}")
        return str(data), None


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
            job_skills_required.extend(skills)

            data, skills = extract_job_data(skills_required, "skills")
            job_profile += data
            job_skills_optional.extend(skills)

            job_profile += extract_job_data(additional_requirements)[0]
            data, skills = extract_job_data(keywords, "skills")
            job_profile += data
            job_skills_optional.extend(skills)

            return job_profile, set(job_skills_required), set(job_skills_optional), role

        else:
            return str(job_data), None, None, None
    except Exception as e:
        logging.error(f"Error in creating job profile: {e}")
        return str(job_data), None, None, None


def clean_text_list(thisList):
    data_to_send = list()
    for thisData in thisList:
        modified_data = ",".join([x.strip() for x in thisData.split("(")])
        modified_data = ",".join([x.strip() for x in modified_data.split(")")])
        modified_data = ",".join([x.strip() for x in modified_data.split("&")])
        data_to_send.extend(
            [x for x in modified_data.split(",") if len(x) > 0])

    return data_to_send


def find_candidate_jobs(state: RoleResumeMatchingState):
    user_profile_embedded = model.encode(
        state['user_profile'], normalize_embeddings=True)
    job_matching_df = pd.DataFrame(columns=['job_id', 'score', 'role', 'job'])
    for job in state['job_postings']:
        # print(job)
        this_job_profile, this_job_skills_reqd, this_job_skills_opt, this_job_role = create_job_profile(
            job)
        logging.info(f"Going to clean user skills")
        resume_skills_formatted = clean_text_list(state['user_skills'])
        logging.info(
            f"Going to clean this_job_skills_reqd: {this_job_skills_reqd}")
        job_skills_formatted_reqd = clean_text_list(this_job_skills_reqd)
        logging.info(f"Going to clean this_job_skills_opt")
        job_skills_formatted_opt = clean_text_list(this_job_skills_opt)

        skills_score_reqd = len(set(resume_skills_formatted) & set(job_skills_formatted_reqd))/len(
            set(job_skills_formatted_reqd)) if len(set(job_skills_formatted_reqd)) > 0 else 0
        skills_score_opt = len(set(resume_skills_formatted) & set(job_skills_formatted_opt))/len(
            set(job_skills_formatted_opt)) if len(set(job_skills_formatted_opt)) > 0 else 0

        job_profile_embedded = model.encode(
            this_job_profile, normalize_embeddings=True)

        logging.info(f"Cosine sim for 1")
        profile_score = float(
            user_profile_embedded @ job_profile_embedded)

        # profile_score = cosine_similarity([model.encode(
        #     this_job_profile, normalize_embeddings=True)], [resume_contents_embedding])[0][0]

        role_sim = 0

        if state['user_prev_roles'] is not None:
            this_job_role_embedded = model.encode(
                this_job_role, normalize_embeddings=True, convert_to_numpy=True) if this_job_role is not None else 0
            if type(this_job_role_embedded) == int:
                role_sim = 0
            else:
                if state['user_prev_roles'] is not None:
                    user_prev_roles_embeddings = list()
                    for x in state['user_prev_roles']:
                        user_prev_roles_embeddings.append(model.encode(
                            x, normalize_embeddings=True, convert_to_numpy=True))
                    if len(user_prev_roles_embeddings) == 0:
                        role_sim = 0
                    else:
                        logging.info(
                            f"Job role: {this_job_role_embedded.shape}")
                        for p in user_prev_roles_embeddings:
                            logging.info(f"prev role: {p.shape}")

                        role_sim = max([
                            float(this_job_role_embedded @ x) for x in user_prev_roles_embeddings
                        ])
        else:
            role_sim = 0
        # print(role_sim)

        total_score = (0.6*profile_score) + (0.25*skills_score_reqd) + \
            (0.05*skills_score_opt) + (0.10*role_sim)
        job_matching_df.loc[len(job_matching_df)] = [
            job["id"], round(total_score, 4), this_job_role, job]

    return {
        "inital_evaluation": job_matching_df
    }


def rerank_jobs(state: RoleResumeMatchingState):
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a recruiter, with an experience of over 20 years.
                    You are an expert in figuring out if a candidate will be a good match for a job role, based on the job posting, and candidate's resume
                    You always follow the pattern given to you for the output, {format_instructions}
                """,
                ),
                (
                    "human",
                    """
                    You are 2 inputs:
                        - text contents of a resume
                        - text contents of a job posting
                    As a senior technical recruiter, you are tasked with providing a score indicating how good of a match a candidate would be for the job role.
                    Task:
                        - Analyse the candidate's profile using their resume                       
                        - Review the job posting
                        - Extract candidate's total experience, and relevant experience for the job
                        - Evaluate the following:
                            - Does the user's skill match with the skills mentioned in the job posting?
                            - Does the user have enough relevant experience, as demanded by the job posting?
                            - Does user have any past experiences that can help with the responsibilities?
                            - Does the user hold educational qualifications, as mentioned in the job posting?
                            - In addition to professional experience, does user have any projects that will help them fulfill the job responsibilities?
                            - Does user have any certificates, or projects that can help bridge the gap between the required skills and the skills that user have?
                        - Based on the evaluation, provide a match score depicting how well the candidate is suited for this job posting
                        
                        - Score guidelines:
                            0.90-1.00: Strong direct match, 70%+ skill alignment, clear relevant experience
                            0.70-0.89: Good match, minor gaps
                            0.50-0.69: Moderate match, noticeable gaps
                            0.30-0.49: Weak alignment
                            0-0.29: Poor match

                        - Be critical. Do not inflate scores.Be strict in your evaluation
                        - Return the result in the following format:
                        {{
                            "job_id":"string",
                            "job_title":"string",
                            "is_eligible":"boolean", #True or False
                            "match_score":"float", # range of 0-1
                            "reason":"string",
                            "strengths":["string"],
                            "gaps":["string"]
                        }}

                        - Return VALID JSON response only.

                        RESUME: {resume},
                        JOB_POSTING: {job_posting}
                    """,
                ),
            ]
        )

        parser = JsonOutputParser(pydantic_object=JobEvaluation_Model)
        prompt = prompt.partial(
            format_instructions=parser.get_format_instructions())

        model = ChatOllama(model="llama3.1:latest", temperature=0.2)
        chain = prompt | model | parser

        job_posting_evaluation = pd.DataFrame(
            columns=['job_id', 'job_title', 'is_eligible', 'match_score', 'time_taken', 'reason', 'strengths', 'gaps', ])

        state['inital_evaluation'].drop_duplicates(
            subset=['job_id'], keep='first', inplace=True)

        for job_details in state['inital_evaluation'].sort_values(by='score', ascending=False)['job'].head(30).values:
            start_time = time.time()
            try:
                llm_res = chain.invoke(
                    {
                        "resume": state['user_resume'],
                        "job_posting": job_details
                    }
                )
                end_time = time.time()
                time_taken = end_time - start_time
                print(llm_res)
                print(time_taken)
                print("-"*150)

                if llm_res is not None:
                    job_posting_evaluation.loc[len(job_posting_evaluation)] = {
                        'job_id': llm_res.get('job_id', ''),
                        'job_title': llm_res.get('job_title', ''),
                        'is_eligible': llm_res.get('is_eligible', ''),
                        'match_score': llm_res.get('match_score', 0.0),
                        'time_taken': time_taken,
                        'reason': llm_res.get('reason', ''),
                        'strengths': llm_res.get('strengths', []),
                        'gaps': llm_res.get('gaps', [])
                    }
            except Exception as e:
                print("eror: ", e)

        final_df = pd.merge(left=job_posting_evaluation, right=state['inital_evaluation'][[
                            'job_id', 'score', 'role', ]], on='job_id', how='left')
        final_df.rename(
            columns={'score': 'embeddings_score', 'match_score': 'llm_score'}, inplace=True)
        final_df['weighted_score'] = (
            0.7*final_df['llm_score']) + (0.3*final_df['embeddings_score'])
        final_df.sort_values(by='weighted_score', ascending=False)

        normalized_llm = (final_df['llm_score'] - min(final_df['llm_score'])) / (
            max(final_df['llm_score']) - min(final_df['llm_score']))
        normalized_embed = (final_df['embeddings_score'] - min(final_df['embeddings_score'])) / (
            max(final_df['embeddings_score']) - min(final_df['embeddings_score']))

        final_df['normalised'] = 0.6 * normalized_llm + 0.4 * normalized_embed
        final_df.sort_values(by='normalised', ascending=False, inplace=True)

        final_df.to_csv("jobs_ranks.csv")
        return {
            "final_evaluation": final_df.to_dict(orient='records')
        }

    except Exception as e:
        logging.error(f"Error in reranking jobs: {e}")
        return {""}


def role_resume_matching_agent():
    """
    Create an agent that will evalauate users profile against a job posting
    """

    graph = StateGraph(RoleResumeMatchingState)

    graph.add_node("get_jobs", extract_jobs)
    graph.add_node("get_resume", extract_resume)
    graph.add_node("create_user_profile", create_user_profile)
    graph.add_node("find_candidate_jobs", find_candidate_jobs)
    graph.add_node("rerank_jobs", rerank_jobs)

    # graph.add_edge(START, "get_jobs")
    # graph.add_edge("get_jobs", "get_resume")
    # graph.add_edge("get_resume", "create_user_profile")
    graph.add_edge(START, "create_user_profile")
    graph.add_edge("create_user_profile", "find_candidate_jobs")
    graph.add_edge("find_candidate_jobs", "rerank_jobs")
    graph.add_edge("rerank_jobs", END)

    return graph.compile()


def start():
    start_time = time.time()
    role_resume_agent = role_resume_matching_agent()

    data = role_resume_agent.invoke({
        "final_evaluation": None,
        "inital_evaluation": None,
        "job_posting_file_path": "job_list_llama3_latest_v7.json",
        "job_postings": list(),
        "resume_file_path": "v1.json",
        "user_prev_roles": list(),
        "user_profile": "",
        "user_resume": dict(),
        "user_skills": []

    })

    print("Entities were: ", data)

    with open('recommendations_v0.json', 'w') as f:
        json.dump(data, f)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")


if __name__ == "__main__":
    start_time = time.time()
    start()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")
