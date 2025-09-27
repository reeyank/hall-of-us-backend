import os
import boto3
from botocore.client import Config
from dotenv import load_dotenv
from PIL import Image

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

def extract_exif_data(image_path: str):
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            # A dictionary to store relevant EXIF data
            # This is a simplified example; you might want to extract more specific tags
            # or use a library that maps EXIF codes to human-readable names.
            # For full EXIF tag reference, see: https://exiftool.org/TagNames/EXIF.html
            extracted_data = {
                "Make": exif_data.get(271),  # Camera Make
                "Model": exif_data.get(272), # Camera Model
                "DateTimeOriginal": exif_data.get(36867), # Date and time of original data generation
                "GPSInfo": exif_data.get(34853) # GPS information
            }
            return {k: v for k, v in extracted_data.items() if v is not None}
        else:
            return {}
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return {}