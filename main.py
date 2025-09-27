from fastapi import FastAPI, UploadFile, File, HTTPException, Form # Import Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from s3_upload import upload_file_to_s3, extract_exif_data
import shutil
import tempfile
import os
import uuid
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Function to get a database connection
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

# Dependency to get a database cursor
def get_db_cursor():
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        yield cur
    finally:
        cur.close()
        conn.close()

# Function to initialize the database table
def initialize_db():
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS photos (
                id VARCHAR(255) PRIMARY KEY,
                filename VARCHAR(255),
                url TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                tags TEXT DEFAULT '',
                user_id VARCHAR(255) DEFAULT NULL,
                likes INTEGER DEFAULT 0
            );
            """
        )
        # Add new columns if they don't exist
        cur.execute("ALTER TABLE photos ADD COLUMN IF NOT EXISTS tags TEXT DEFAULT '';")
        cur.execute("ALTER TABLE photos ADD COLUMN IF NOT EXISTS user_id VARCHAR(255) DEFAULT NULL;")
        cur.execute("ALTER TABLE photos ADD COLUMN IF NOT EXISTS likes INTEGER DEFAULT 0;")
        conn.commit()
        print("Database table 'photos' initialized successfully.")
    except psycopg2.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    initialize_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/photos/upload")
async def upload_photo(
    file: UploadFile = File(...),
    tags: str = Form(''), # Optional tags as a comma-separated string
    user_id: str = Form(None) # Optional user ID
):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name
    
    try:
        uuid_photo = uuid.uuid4().hex
        file_extension = os.path.splitext(file.filename)[1]
        uuid_photo_with_extension = uuid_photo + file_extension
        s3_object_name = "photos/" + uuid_photo_with_extension
        public_url = upload_file_to_s3(temp_file_path, s3_object_name)
        
        if not public_url:
            raise HTTPException(status_code=500, detail="Failed to upload file or get public URL")
        
        # Extract EXIF data
        exif_data = extract_exif_data(temp_file_path)
        print(f"Extracted EXIF Data: {exif_data}")

        # Save photo metadata to the database using psycopg2
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO photos (id, filename, url, tags, user_id, likes) VALUES (%s, %s, %s, %s, %s, %s)",
                (uuid_photo, file.filename, public_url, tags, user_id, 0)
            )
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            cur.close()
            conn.close()
        
        print(f"UUID: {uuid_photo_with_extension}")
        return {"id": uuid_photo, "filename": file.filename, "url": public_url, "tags": tags, "user_id": user_id, "likes": 0}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        os.remove(temp_file_path)

@app.get("/photos/{image_id}")
async def get_photo(image_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, filename, url, tags, user_id, likes FROM photos WHERE id = %s", (image_id,))
        photo = cur.fetchone()
        if photo:
            return {
                "id": photo[0],
                "filename": photo[1],
                "url": photo[2],
                "tags": photo[3],
                "user_id": photo[4],
                "likes": photo[5],
            }
        else:
            raise HTTPException(status_code=404, detail="Image not found in database")
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

@app.get("/photos")
async def get_all_photos():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, filename, url, tags, user_id, likes FROM photos")
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
        photos = []
        for row in rows:
            photos.append({
                "id": row[0],
                "filename": row[1],
                "url": row[2],
                "tags": row[3],
                "user_id": row[4],
                "likes": row[5],
            })
        return {"photos": photos}
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

@app.get("/photos/user/{user_id}")
async def get_photos_by_user(user_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, filename, url, tags, user_id, likes FROM photos WHERE user_id = %s", (user_id,))
        rows = cur.fetchall()
        columns = [col[0] for col in cur.description]
        photos = []
        for row in rows:
            photos.append({
                "id": row[0],
                "filename": row[1],
                "url": row[2],
                "tags": row[3],
                "user_id": row[4],
                "likes": row[5],
            })
        return {"photos": photos}
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()