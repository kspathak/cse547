import pickle
import tempfile

import boto3

_s3_client = None
def _get_s3_client():
    """ Caches an instance of the AWS S3 client.
    """
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client('s3')
    return _s3_client

def serialize_object(obj, bucket: str, key: str) -> None:
    _, results_file_path = tempfile.mkstemp()
    with open(results_file_path, 'wb') as f:
        pickle.dump(obj, f)
    with open(results_file_path, 'rb') as f:
        _get_s3_client().upload_fileobj(f, bucket, key)

def deserialize_object(bucket, key):
    _, results_file_path = tempfile.mkstemp()
    with open(results_file_path, 'wb') as f:
        _get_s3_client().download_fileobj(bucket, key, f)
    with open(results_file_path, 'rb') as f:
        return pickle.load(f)
