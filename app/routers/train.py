from fastapi import APIRouter, Request, UploadFile, File
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix='/train',tags=['training'])

@router.get('/')
async def upload_form(request:Request):
    print('upload_form')

@router.post('/upload')
async def upload_data(file: UploadFile = File(...)):
    print('upload post')

@router.post('/retrain')
async def retrain_model():
    print('retrain trigger')
