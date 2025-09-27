import os
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# If using AWS S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

def upload_file_to_s3(file_name: str, object_name: str = None):
    if object_name is None:
        object_name = os.path.basename(file_name)

    try:
        s3_client.upload_file(file_name, S3_BUCKET_NAME, object_name)
        print(f"File uploaded to {S3_BUCKET_NAME}/{object_name}")
        public_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{object_name}"
        return public_url
    except Exception as e:
        print(f"Upload error: {e}")
        return None

def convert_ifd_rational_to_float(data):
    if isinstance(data, IFDRational):
        try:
            return float(data.numerator) / float(data.denominator)
        except ZeroDivisionError:
            return 0.0
    elif isinstance(data, dict):
        return {k: convert_ifd_rational_to_float(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(convert_ifd_rational_to_float(elem) for elem in data)
    elif isinstance(data, bytes):
        return data.decode('utf-8', errors='ignore')
    elif not isinstance(data, (str, int, float, bool, type(None))): # Catch any remaining non-serializable types
        return str(data)
    return data

def extract_exif_data(image_path: str):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            extracted_data = {
                "GPSInfo": exif_data.get(34853) # GPS information
            }
            if extracted_data.get("GPSInfo"):
                extracted_data["GPSInfo"] = convert_ifd_rational_to_float(extracted_data["GPSInfo"])
            return {k: v for k, v in extracted_data.items() if v is not None}
        else:
            return {}
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return {}