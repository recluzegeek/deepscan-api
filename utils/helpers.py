import os
import uuid

def unique_id():
    return str(uuid.uuid4())

def delete_file(filename):
    os.remove(filename)