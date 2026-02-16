# **RoleRadar:** _Semantic Resume-to-Job Matching Engine_

> :warning: This project is actively under development

## Overview

RoleRadar is a semantic resume-to-job matching engine that combines:

- Structured resume parsing
- LLM-assisted job title refinement
- Embeddings-based semantic similarity
- LLM-based re-ranking with explanation
- Agent-based orchestration

The goal is to move beyong keyword matching and build a context-aware resume-to-job matching system.

## Project Status

---

## Status
|Component | Status|
|---|---|
|Resume Parsing Agent | Working|
|Job Fetching | Working|
|Orchestration(LangGraph) | Partial|
|Job Matching Agent | In Progress|
|Embedding Caching | Planned|
|MongoDB Integration | Planned|
|ChromaDB Integration | Planned|
|FastAPI Backend | In Progress|
|UI | Planned|
|End-to-End Exceution | In Progress|

## Core Idea:

Traditional job portals rely heavily on keyword overlap.

RoleRadar instead:

- Parses resume into structured JSON
- Embeds resume and job postings
- Computes semantic similarity
- Filters top candidates using weighted similarity scoring
- Re-ranks top results using an LLM
- Returns final ranked matches with explanation

## System Architecture
![System_design](assets/Role_Radar_sys_arch.png)

High-level architecture:

- User -> Orchestrator Agent
- Resume Parser Agent
- Job Extractor Agent
- Job Matching Agent
- MongoDB(structured data)
- ChromaDB(vector mbeddings)

(Architecture diagram in repository: )

## Agent Architecture:

- **_Orchestrator Agent_**:
  - Controls full workflow
  - Routes data between agents
- **_Resume Parser Agent_**:
  - Structured extraction
  - Uses LLM-based schema enforecement
- **_Job extractor Agent_**:
  - Cleans and structures job descriptions
- **_Job Matching Agent_**:
  - Embedding similarity
  - Weighted scoring
  - LLM re-ranking

## Application Flow:

1. **User Input**

User provides:

- Resume
- Target job titles
- Preferred location

2. **Resume Parsing**

- Resume Parser Agent:
  - Converts resume -> structured JSON
  - Removes PII
  - Stores anonymized profile
  - Uses hashed email as user ID
  - Profile versioning enabled

3. **Job Title Evaluation**

- Agent:
  - Evaluated selected job titles against resume
  - Suggests additioanl relevant job titles
  - Provides reasoning

User selects final job title(s).

4. **Job-Feching(Adzuna API)**

System:

- Fetches job postings using selected job titles + location
- Returns raw job postings

## LLM Setup:

Currently using differnt Ollama models for:

- Resume parsing
- Structured extraction
- Matching analysis
- Re-ranking
  This setup is for experimentation and rapid iteration.

Model selection is based on:

- Response quality
- Latency trade-offs

In production, models may be replaced with:

- GPT-based models
- Otehr hosted LLM APIs

## Tech Stack:

- **LangGraph** -> Orchestration workflow
- **LangChain** -> Prompt pipelines and LLM integration
- **Ollama** -> Local LLM experimentation
- **FastAPI** -> Backedn API layer(in progress)
- **MongoDB** -> Structured storage(planned)
- **ChromaDB** -> Embedding storage(planned)
- **all-Mini-L6-v2** -> Embedding model

## Scoring Strategy:

Final ranking combines:

- Semantic embedding similarity
- LLM-based contextual reasoning

This hybrid scoring improves:

- Relevance
- Context alignment
- Skill interpretation
- Experience weighting

## Planned Improvements:

- Frontend UI
- Feedback-driven ranking optimization
- Model evaluation framework
- Production-grade LLM integration
- Latency optimization
- Scalable embedding infrastructure
- Better explainability for match reasoning

RoleRadar is an agentic AI system built using LangGraph, where each autonomous agent handles a specialized task—resume parsing, job extraction, and job matching—while a higher-level orchestrator agent manages flow control, state propagation, and human-in-the-loop decisions. The system combines LLM-based semantic understanding with deterministic scoring logic to deliver personalized, high-confidence job recommendations.

## Problem Statement:

The job alerts that we receive from LinkedIn/Naukri/Company career websites are keyword-specific. We create job alerts based on the poosition we want to apply for and the location. However, many times those job postings do not meet our search criteria, or at times the job alert might not alert us about a role that woiuld be a prfect fit for us, just because it failed the keyword match.

## Solution:

Based on users' resume, description of the role they want, the title they want to target, and their preferred location, create a solution that would provide them personalised job recommendations

## Tech Stack:

- LangGraph for Agents orchestration
- Ollama as LLM(free; open-source)

---
