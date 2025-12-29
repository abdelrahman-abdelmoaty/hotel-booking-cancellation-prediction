"""
MongoDB Data Loader Utilities

This module provides functions for connecting to MongoDB Atlas
and loading hotel booking data.
"""

import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_mongodb_connection():
    """
    Establish connection to MongoDB Atlas.
    
    Returns:
        MongoClient: MongoDB client object
        
    Raises:
        ConnectionFailure: If connection to MongoDB fails
    """
    mongodb_uri = os.getenv('MONGODB_URI')
    if not mongodb_uri:
        raise ValueError("MONGODB_URI not found in environment variables. "
                        "Please set it in .env file or as environment variable.")
    
    try:
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        print("✓ Successfully connected to MongoDB Atlas")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"✗ Failed to connect to MongoDB: {e}")
        raise


def load_csv_to_mongodb(csv_path, db_name, collection_name, mongodb_uri=None):
    """
    Load CSV file into MongoDB collection.
    
    Args:
        csv_path (str): Path to CSV file
        db_name (str): MongoDB database name
        collection_name (str): MongoDB collection name
        mongodb_uri (str, optional): MongoDB connection string. 
                                    If None, uses MONGODB_URI from .env
    
    Returns:
        int: Number of documents inserted
    """
    if mongodb_uri:
        client = MongoClient(mongodb_uri)
    else:
        client = get_mongodb_connection()
    
    db = client[db_name]
    collection = db[collection_name]
    
    # Read CSV in chunks to handle large files
    print(f"Reading CSV file: {csv_path}")
    chunk_size = 10000
    total_inserted = 0
    
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Convert DataFrame to list of dictionaries
        records = chunk.to_dict('records')
        # Insert into MongoDB
        result = collection.insert_many(records)
        total_inserted += len(result.inserted_ids)
        print(f"Inserted {total_inserted} documents...")
    
    print(f"✓ Total documents inserted: {total_inserted}")
    client.close()
    return total_inserted


def create_indexes(db_name, collection_name, mongodb_uri=None):
    """
    Create indexes on MongoDB collection for efficient querying.
    
    Args:
        db_name (str): MongoDB database name
        collection_name (str): MongoDB collection name
        mongodb_uri (str, optional): MongoDB connection string
    """
    if mongodb_uri:
        client = MongoClient(mongodb_uri)
    else:
        client = get_mongodb_connection()
    
    db = client[db_name]
    collection = db[collection_name]
    
    # Create indexes
    indexes = [
        ("is_canceled", 1),  # Target variable
        ("hotel", 1),  # Hotel type filter
        ("arrival_date_year", 1),  # Year filter
        ("arrival_date_month", 1),  # Month filter
        ("country", 1),  # Country filter
        ("market_segment", 1),  # Market segment filter
    ]
    
    print("Creating indexes...")
    for field, direction in indexes:
        try:
            collection.create_index([(field, direction)])
            print(f"✓ Created index on {field}")
        except Exception as e:
            print(f"✗ Failed to create index on {field}: {e}")
    
    client.close()


def load_from_mongodb(db_name, collection_name, query=None, mongodb_uri=None):
    """
    Load data from MongoDB collection to pandas DataFrame.
    
    Args:
        db_name (str): MongoDB database name
        collection_name (str): MongoDB collection name
        query (dict, optional): MongoDB query filter
        mongodb_uri (str, optional): MongoDB connection string
    
    Returns:
        pd.DataFrame: DataFrame containing the data
    """
    if mongodb_uri:
        client = MongoClient(mongodb_uri)
    else:
        client = get_mongodb_connection()
    
    db = client[db_name]
    collection = db[collection_name]
    
    # Query MongoDB
    if query:
        cursor = collection.find(query)
    else:
        cursor = collection.find()
    
    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))
    
    # Remove MongoDB _id field if present
    if '_id' in df.columns:
        df = df.drop('_id', axis=1)
    
    print(f"✓ Loaded {len(df)} documents from MongoDB")
    client.close()
    return df


def get_collection_stats(db_name, collection_name, mongodb_uri=None):
    """
    Get statistics about MongoDB collection.
    
    Args:
        db_name (str): MongoDB database name
        collection_name (str): MongoDB collection name
        mongodb_uri (str, optional): MongoDB connection string
    
    Returns:
        dict: Collection statistics
    """
    if mongodb_uri:
        client = MongoClient(mongodb_uri)
    else:
        client = get_mongodb_connection()
    
    db = client[db_name]
    collection = db[collection_name]
    
    stats = {
        'total_documents': collection.count_documents({}),
        'cancelled': collection.count_documents({'is_canceled': 1}),
        'not_cancelled': collection.count_documents({'is_canceled': 0}),
        'cancellation_rate': collection.count_documents({'is_canceled': 1}) / 
                            collection.count_documents({}) * 100
    }
    
    client.close()
    return stats

