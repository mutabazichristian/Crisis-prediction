from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix='/predict',tags=['prediction'])

@router.get("/")
async def predict_form(request: Request):
    print('predict_form')

@router.post('/')
async def predict(request:Request, feature1: float = Form (...), feature2: float = Form(...)):
    print('make a prediction')
