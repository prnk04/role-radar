from pymongo import MongoClient
from pymongo import AsyncMongoClient


class MongoConnect:
    client: MongoClient = None

    @classmethod
    def connect(cls, uri: str):
        if cls.client is None:
            cls.client = MongoClient(uri)
            print("Mongo db connected")
        return cls.client

    @classmethod
    def close(cls):
        if cls.client:
            cls.client.close()
            print("Mongo db connection closed")
            cls.client = None


class AsyncMongoConnect:
    client: AsyncMongoClient = None

    @classmethod
    def connect(cls, uri: str):
        if cls.client is None:
            cls.client = AsyncMongoClient(uri)
            print("Mongo db connected")
        return cls.client

    @classmethod
    def close(cls):
        if cls.client:
            x = cls.client.close()
            print("Mongo db connection closed: ", x)
            cls.client = None
