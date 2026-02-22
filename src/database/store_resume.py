import datetime
from bson import ObjectId
from pydantic import ValidationError
import pymongo
# from pymongo.database import Database
from pymongo.asynchronous.database import AsyncDatabase
from schemas.data_models import StructuredResume
from schemas.database_schema import UserProfileFullModel, UserProfileShortModel
import logging
import hashlib

from src.utils.commons import clean_text_list, get_hashed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(filename)s:%(funcName)s():%(lineno)d  -  %(levelname)s- %(message)s")
# logger = logging.getLogger("__name__")


async def store_resume(resume: StructuredResume, db: AsyncDatabase, hashed_resume: str):
    try:
        # logging.info(f"Inside store resume: {resume}")

        if isinstance(resume, dict):
            resume = StructuredResume(**resume)
        user_resume = UserProfileFullModel(
            user_id=get_hashed(resume.email_id),
            resume_hash=hashed_resume,
            created_at=datetime.datetime.now(
                datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z"),
            **resume.model_dump(exclude={"email_id"})
        )

        this_version = 0

        resume_collection = db.get_collection('user_profile_extended')

        # first, let's check if the document exists
        existing_user = await resume_collection.find_one(filter={'user_id': get_hashed(
            resume.email_id)}, sort=[('_id', pymongo.DESCENDING)])
        logging.info(f'existing user: {existing_user}')

        if existing_user:
            if existing_user.get('resume_hash') == hashed_resume:
                logging.info('Same resume, so not doing anything')
                return None
            else:
                last_version = existing_user.get('version')
                this_version = last_version + 1

        user_resume.version = this_version
        user_resume_dict = user_resume.model_dump()
        res = await resume_collection.insert_one(user_resume_dict)
        logging.info(f"Stored user profile extended in the db")
        return res.inserted_id
    except ValidationError as ve:
        logging.error(f"Validation error: {ve}")
        raise ve
    except Exception as e:
        logging.error(type(e))
        logging.error(f"Error in storing resume in db: {e}")
        return None


async def store_user_profile(user_email_id: str, user_profile: str, prev_roles: list, all_skills: list, db: AsyncDatabase, full_profile_id: ObjectId):
    try:

        user_profile_to_store = UserProfileShortModel(
            user_id=hashlib.sha256(
                user_email_id.encode('utf-8')).hexdigest(),
            profile=user_profile,
            prev_roles=prev_roles,
            all_skills=all_skills,
            embeddings_status='pending',
            full_profile_id=str(full_profile_id),
            created_at=datetime.datetime.now(
                datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S %z")
        )
        user_profile_dict = user_profile_to_store.model_dump()
        resume_collection = db.get_collection('user_profile_brief')
        res = await resume_collection.insert_one(user_profile_dict, )
        logging.info(f"Insertion res: {res.inserted_id}")

        logging.info(f"Stored user profile brief in the db")
        return res.inserted_id

    except ValidationError as ve:
        logging.error(f"Validation error: {ve}")
        return ve
    except Exception as e:
        logging.error(type(e))
        logging.error(f"Error in storing user profile in db: {e}")
        return None


async def update_embedding_status(status_code, obj_id, db: AsyncDatabase):
    try:
        status_message = 'failed' if status_code == 0 else 'succeeded'
        resume_collection = db.get_collection('user_profile_brief')
        res1 = await resume_collection.find_one({'_id': ObjectId(obj_id)})
        # logging.info(f"res1: {res1}: {ObjectId(obj_id)}")
        res = await resume_collection.find_one_and_update(
            {'_id': ObjectId(obj_id)}, update={'$set': {'embeddings_status': status_message, }})
        # logging.info(f"After updating embedding status: {res}")

    except Exception as e:
        logging.error(f"Error in updating embedding status: {e}")
