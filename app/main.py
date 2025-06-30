from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta, datetime
import asyncio
import logging
import pandas as pd
import numpy as np
from tinkoff.invest import Client, SecurityTradingStatus, CandleInterval
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.utils import quotation_to_decimal, now
import ta
from copy import deepcopy
import joblib
import joblib
import ta
import numpy as np
from copy import deepcopy
from sqlalchemy import desc, distinct
from huggingface_hub import hf_hub_download
import os 


try:
    import models
    import auth
except:
    from app import auth, models

try:
    from database import engine, get_db
except:
    from app.database import engine, get_db

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

models.Base.metadata.create_all(bind=engine)

from get_data3 import get_current_features1

app = FastAPI()


# MODEL_PATH = "randomforest_model.joblib"
# REPO_ID = "IvanBorrow/MLFG"
# FILENAME = "rf.joblib"

# if not os.path.exists(MODEL_PATH):
#     print("📥 Скачиваем модель с Hugging Face...")
#     hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=".", local_dir_use_symlinks=False)
#     print('Скачан')

# # Загружаем модель один раз при старте
# try:
#     ml_model = joblib.load("rf.joblib")
#     logger.info("ML model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load ML model: {e}")
#     raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # URL фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация Tinkoff API
TOKEN = "t.YbAt3ov-iNU4jt9A4l9ML4ga77xB1z_NYKOFEvZZDRv72ilghDUEJVk3B86XRSCeyNz5_do2Go_cAqj2qjH9Jg"


async def update_prediction():
    from concurrent.futures import ThreadPoolExecutor

    executor = ThreadPoolExecutor(max_workers=1)

    while True:
        try:
            # send_prediction_task()
            logger.info("Starting prediction update...")
            # Запускаем get_current_features1 в отдельном потоке
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(executor, get_current_features1)

            
            current_time = datetime.utcnow()

            # Сохраняем в базу
            db = next(get_db())
            new_prediction = models.Prediction(
                value=float(prediction), timestamp=current_time
            )
            db.add(new_prediction)
            db.commit()
            logger.info(f"New prediction saved: {prediction} at {current_time}")
            # else:
            #     logger.error("Failed to create valid features DataFrame")

        except Exception as e:
            logger.error(f"Error in prediction update: {e}")

        await asyncio.sleep(10)  # 5 минут между обновлениями


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_prediction())


@app.post("/register")
def register(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    logger.info(f"Attempting to register user with email: {form_data.username}")
    try:
        user = (
            db.query(models.User)
            .filter(models.User.email == form_data.username)
            .first()
        )
        if user:
            logger.warning(f"User with email {form_data.username} already exists")
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = auth.get_password_hash(form_data.password)
        new_user = models.User(
            email=form_data.username, hashed_password=hashed_password
        )
        db.add(new_user)
        db.commit()
        logger.info(f"Successfully registered user with email: {form_data.username}")
        return {"message": "User created successfully"}
    except Exception as e:
        logger.error(f"Error during registration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    logger.info(f"Login attempt for user: {form_data.username}")
    try:
        user = (
            db.query(models.User)
            .filter(models.User.email == form_data.username)
            .first()
        )
        if not user:
            logger.warning(f"User not found: {form_data.username}")
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not auth.verify_password(form_data.password, user.hashed_password):
            logger.warning(f"Invalid password for user: {form_data.username}")
            raise HTTPException(
                status_code=401,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = auth.create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        logger.info(f"Successful login for user: {form_data.username}")
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        logger.error(f"Error during login: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prediction")
async def get_prediction(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    try:
        latest_prediction = (
            db.query(models.Prediction)
            .order_by(desc(models.Prediction.timestamp))
            .first()
        )

        if not latest_prediction:
            return {
                "status": "initializing",
                "message": "ML model is warming up, please wait...",
            }

        return {
            "status": "ready",
            "prediction": latest_prediction.value,
            "timestamp": latest_prediction.timestamp,
        }
    except Exception as e:
        logger.error(f"Error in get_prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/prediction/history")
async def get_prediction_history(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    try:
        recent_predictions = (
            db.query(models.Prediction)
            .order_by(desc(models.Prediction.timestamp))
            .limit(2000)
            .all()
        )

        return {
            "history": [
                {"prediction": pred.value, "timestamp": pred.timestamp}
                for pred in recent_predictions
            ]
        }
    except Exception as e:
        logger.error(f"Error in get_prediction_history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
