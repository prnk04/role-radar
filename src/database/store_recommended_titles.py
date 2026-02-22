import datetime
from bson import ObjectId
from pydantic import ValidationError
import pymongo
from pymongo.database import Database
from schemas.data_models import StructuredResume
from schemas.database_schema import UserProfileFullModel, UserProfileShortModel
import logging
import hashlib

from src.utils.commons import clean_text_list, get_hashed

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(filename)s:%(funcName)s():%(lineno)d  -  %(levelname)s- %(message)s")
# logger = logging.getLogger("__name__")


def store_resume(resume: StructuredResume, db: Database, hashed_resume: str):
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
        existing_user = resume_collection.find_one(filter={'user_id': get_hashed(
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
        res = resume_collection.insert_one(user_resume_dict)
        logging.info(f"Stored user profile extended in the db")
        return res.inserted_id
    except ValidationError as ve:
        logging.error(f"Validation error: {ve}")
        raise ve
    except Exception as e:
        logging.error(type(e))
        logging.error(f"Error in storing resume in db: {e}")
        return None


def update_embedding_status(status_code, obj_id, db: Database):
    try:
        status_message = 'failed' if status_code == 0 else 'succeeded'
        resume_collection = db.get_collection('user_profile_brief')
        res1 = resume_collection.find_one({'_id': ObjectId(obj_id)})
        # logging.info(f"res1: {res1}: {ObjectId(obj_id)}")
        res = resume_collection.find_one_and_update(
            {'_id': ObjectId(obj_id)}, update={'$set': {'embeddings_status': status_message, }})
        # logging.info(f"After updating embedding status: {res}")

    except Exception as e:
        logging.error(f"Error in updating embedding status: {e}")
