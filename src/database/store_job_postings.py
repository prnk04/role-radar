from pymongo import ReturnDocument, UpdateOne
# from pymongo.database import Database
from pymongo.asynchronous.database import AsyncDatabase
import logging

from schemas.data_models import JobPostingsModel1
from schemas.database_schema import BriefJobPostingsModel, JobPostingsModel_DB


def clean_text_list(thisList):
    data_to_send = list()
    for thisData in thisList:
        modified_data = ",".join([x.strip() for x in thisData.split("(")])
        modified_data = ",".join([x.strip() for x in modified_data.split(")")])
        modified_data = ",".join([x.strip() for x in modified_data.split("&")])
        data_to_send.extend(
            [x.strip() for x in modified_data.split(",") if len(x) > 0])

    data_to_send = [x.strip() for x in data_to_send if len(x.strip()) > 0]

    return data_to_send


async def bulk_store_jobs(job_posting: list[JobPostingsModel_DB], db: AsyncDatabase):
    try:
        job_posting_collection = db.get_collection('job_posting')
        operations = list()
        logging.info(f"Should store {len(job_posting)} documents")

        for job in job_posting:
            data_to_insert = job.model_dump()
            operation = UpdateOne(
                filter={"id": job.id},
                update={"$set": data_to_insert},
                upsert=True
            )
            operations.append(operation)

        await job_posting_collection.bulk_write(operations)

        # now, getting the ids of the documents that were inserted/modified so that we can use those later
        job_ids = [job.id for job in job_posting]
        obj_ids = await job_posting_collection.find(
            {"id": {"$in": job_ids}}, {"_id": 1, "id": 1}).to_list()

        return {doc["id"]: doc["_id"] for doc in obj_ids}
    except Exception as e:
        logging.error(f"Error in bulk storing job postings: {e}")
        logging.info(f"Result was: {job_posting}")
        return None


async def bulk_store_job_profiles(job_profiles: list[BriefJobPostingsModel], db: AsyncDatabase):
    try:
        job_profile_collection = db.get_collection('job_profile')
        operations = list()
        logging.info(f"Should store {len(job_profiles)} profiles")

        for job in job_profiles:
            data_to_insert = job.model_dump()
            operation = UpdateOne(
                filter={"id": job.id},
                update={"$set": data_to_insert},
                upsert=True
            )
            operations.append(operation)

        await job_profile_collection.bulk_write(operations)

        # now, getting the ids of the documents that were inserted/modified so that we can use those later
        job_ids = [job.id for job in job_profiles]
        obj_ids = await job_profile_collection.find(
            {"id": {"$in": job_ids}}, {"_id": 1, "id": 1}).to_list()

        return {doc["id"]: doc["_id"] for doc in obj_ids}
    except Exception as e:
        logging.error(f"Error in bulk storing job profiles: {e}")
        return None


async def store_job_posting(job_posting: list[JobPostingsModel_DB], db: AsyncDatabase):
    try:
        logging.info(f"I want to store: {job_posting}")

        job_posting_collection = db.get_collection('job_posting')
        data_1 = [x.model_dump() for x in job_posting]
        res = await job_posting_collection.insert_many(data_1)
        logging.info(f"Stored job posting in the db: {res}")
        return res
    except Exception as e:
        logging.error(f"Error in storing job posting in db: {e}")


async def store_job_profiles(job_profiles, db: AsyncDatabase):
    try:
        # logging.info(f"Job profiles to store: {job_profiles}")
        job_profiles_collection = db.get_collection('job_profiles')
        res = await job_profiles_collection.insert_many(job_profiles)

        logging.info(f"Stored job_profiles in the db: {res}")
        return res
    except Exception as e:
        logging.error(f"Error in job_profiles in db: {e}")
