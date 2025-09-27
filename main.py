from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends # Import Form, Depends
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
import json # Re-introduce the json module
from pydantic import BaseModel # Import BaseModel
from PIL import Image, ImageDraw, ImageFont # Import Pillow components
from io import BytesIO # Import BytesIO
from fastapi.responses import Response # Import Response

load_dotenv()

# Pydantic model for User
class User(BaseModel):
    username: str
    password: str

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def convert_dms_to_decimal(dms, ref):
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    decimal_degrees = degrees + (minutes / 60) + (seconds / 3600)
    if ref in ['S', 'W']:
        decimal_degrees *= -1
    return decimal_degrees

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
                likes INTEGER DEFAULT 0,
                caption TEXT DEFAULT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(255) PRIMARY KEY,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS photo_likes (
                photo_id VARCHAR(255) REFERENCES photos(id) ON DELETE CASCADE,
                user_id VARCHAR(255) REFERENCES users(username) ON DELETE CASCADE,
                PRIMARY KEY (photo_id, user_id),
                liked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        # Add new columns if they don't exist
        cur.execute("ALTER TABLE photos ADD COLUMN IF NOT EXISTS tags TEXT DEFAULT '';")
        cur.execute("ALTER TABLE photos ADD COLUMN IF NOT EXISTS user_id VARCHAR(255) DEFAULT NULL;")
        cur.execute("ALTER TABLE photos ADD COLUMN IF NOT EXISTS likes INTEGER DEFAULT 0;")
        cur.execute("ALTER TABLE photos ADD COLUMN IF NOT EXISTS exif_gps_info TEXT DEFAULT NULL;")
        cur.execute("ALTER TABLE photos ADD COLUMN IF NOT EXISTS caption TEXT DEFAULT NULL;")
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
    user_id: str = Form(None), # Optional user ID
    caption: str = Form(None) # Optional caption
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
        s3_object_name = uuid_photo_with_extension
        public_url = upload_file_to_s3(temp_file_path, s3_object_name)
        
        if not public_url:
            raise HTTPException(status_code=500, detail="Failed to upload file or get public URL")
        
        # Extract EXIF data
        exif_data = extract_exif_data(temp_file_path)
        exif_gps_info_raw = exif_data.get("GPSInfo")
        print(f"DEBUG: Type of exif_gps_info_raw: {type(exif_gps_info_raw)}")
        print(f"DEBUG: Content of exif_gps_info_raw: {exif_gps_info_raw}")
        exif_gps_info = json.dumps(exif_gps_info_raw) if exif_gps_info_raw else None

        # Save photo metadata to the database using psycopg2
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO photos (id, filename, url, tags, user_id, likes, exif_gps_info, caption) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (uuid_photo, file.filename, public_url, tags, user_id, 0, exif_gps_info, caption)
            )
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {e}")
        finally:
            cur.close()
            conn.close()
        
        print(f"UUID: {uuid_photo_with_extension}")
        return {"id": uuid_photo, "filename": file.filename, "url": public_url, "tags": tags, "user_id": user_id, "likes": 0, "exif_gps_info": json.loads(exif_gps_info) if exif_gps_info else None, "caption": caption}
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
        cur.execute("SELECT id, filename, url, tags, user_id, likes, exif_gps_info, caption FROM photos WHERE id = %s", (image_id,))
        photo = cur.fetchone()
        if photo:
            return {
                "id": photo[0],
                "filename": photo[1],
                "url": photo[2],
                "tags": photo[3],
                "user_id": photo[4],
                "likes": photo[5],
                "exif_gps_info": json.loads(photo[6]) if photo[6] else None,
                "caption": photo[7]
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
        cur.execute("SELECT id, filename, url, tags, user_id, likes, exif_gps_info, caption FROM photos")
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
                "exif_gps_info": json.loads(row[6]) if row[6] else None,
                "caption": row[7]
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
        cur.execute("SELECT id, filename, url, tags, user_id, likes, exif_gps_info, caption FROM photos WHERE user_id = %s", (user_id,))
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
                "exif_gps_info": json.loads(row[6]) if row[6] else None,
                "caption": row[7]
            })
        return {"photos": photos}
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

@app.post("/photos/{photo_id}/like")
async def like_photo(photo_id: str, user_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Check if photo exists
        cur.execute("SELECT id FROM photos WHERE id = %s", (photo_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Photo not found")

        # Check if user exists
        cur.execute("SELECT username FROM users WHERE username = %s", (user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        # Check if already liked
        cur.execute("SELECT 1 FROM photo_likes WHERE photo_id = %s AND user_id = %s", (photo_id, user_id))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Photo already liked by this user")

        # Add like and increment count
        cur.execute("INSERT INTO photo_likes (photo_id, user_id) VALUES (%s, %s)", (photo_id, user_id))
        cur.execute("UPDATE photos SET likes = likes + 1 WHERE id = %s", (photo_id,))
        conn.commit()
        return {"message": "Photo liked successfully", "photo_id": photo_id, "user_id": user_id}
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

@app.delete("/photos/{photo_id}/unlike")
async def unlike_photo(photo_id: str, user_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Check if photo exists
        cur.execute("SELECT id FROM photos WHERE id = %s", (photo_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="Photo not found")

        # Check if user exists
        cur.execute("SELECT username FROM users WHERE username = %s", (user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="User not found")

        # Check if liked
        cur.execute("SELECT 1 FROM photo_likes WHERE photo_id = %s AND user_id = %s", (photo_id, user_id))
        if not cur.fetchone():
            raise HTTPException(status_code=400, detail="Photo not liked by this user")

        # Remove like and decrement count
        cur.execute("DELETE FROM photo_likes WHERE photo_id = %s AND user_id = %s", (photo_id, user_id))
        cur.execute("UPDATE photos SET likes = likes - 1 WHERE id = %s", (photo_id))
        conn.commit()
        return {"message": "Photo unliked successfully", "photo_id": photo_id, "user_id": user_id}
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

@app.get("/gps")
async def get_all_photos_gps_data():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT id, exif_gps_info FROM photos WHERE exif_gps_info IS NOT NULL")
        rows = cur.fetchall()
        gps_data_list = []
        for row in rows:
            photo_id = row[0]
            exif_gps_info_str = row[1]
            try:
                gps_info = json.loads(exif_gps_info_str)
                
                latitude_dms = gps_info.get("2") # Key "2" for latitude DMS
                latitude_ref = gps_info.get("1") # Key "1" for latitude reference
                longitude_dms = gps_info.get("4") # Key "4" for longitude DMS
                longitude_ref = gps_info.get("3") # Key "3" for longitude reference

                if latitude_dms and latitude_ref and longitude_dms and longitude_ref:
                    latitude = convert_dms_to_decimal(latitude_dms, latitude_ref)
                    longitude = convert_dms_to_decimal(longitude_dms, longitude_ref)
                    gps_data_list.append({
                        "photo_id": photo_id,
                        "latitude": latitude,
                        "longitude": longitude
                    })
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for photo_id {photo_id}")
                continue # Skip this entry if JSON is malformed
        return {"gps_data": gps_data_list}
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

@app.get("/plaque_text")
async def get_image_with_text(text: str = "Hello, World!"):
    img_width = 800
    img_height = 400
    background_color = (139, 69, 19)  # Brown color (RGB)
    text_color = (255, 255, 255)  # White color (RGB)

    img = Image.new('RGB', (img_width, img_height), color=background_color)
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("./TheSeasons.otf", 40)
    except IOError:
        font = ImageFont.load_default() # Fallback to default font

    # Calculate text size and position
    text_bbox = d.textbbox((0,0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (img_width - text_width) / 2
    y = (img_height - text_height) / 2

    d.text((x, y), text, fill=text_color, font=font)

    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return Response(content=img_byte_arr.getvalue(), media_type="image/png")

@app.post("/signup")
async def signup(user: User):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Check if username already exists
        cur.execute("SELECT username FROM users WHERE username = %s", (user.username,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Username already registered")

        cur.execute(
            "INSERT INTO users (username, password) VALUES (%s, %s)",
            (user.username, user.password)
        )
        conn.commit()
        return {"message": "User registered successfully", "user_id": user.username} # Assuming user_id is the username for now
    except psycopg2.Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

@app.post("/signin")
async def signin(user: User):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT password FROM users WHERE username = %s", (user.username,))
        result = cur.fetchone()

        if not result:
            raise HTTPException(status_code=400, detail="Incorrect username or password")

        stored_password = result[0]
        if user.password == stored_password:
            return {"message": "Signed in successfully", "user_id": user.username}
        else:
            raise HTTPException(status_code=400, detail="Incorrect username or password")
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")
    finally:
        cur.close()
        conn.close()

