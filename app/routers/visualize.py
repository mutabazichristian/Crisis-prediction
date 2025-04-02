from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates

router = APIRouter(prefix='/visualize',tags=['visualization'])

@router.get('/')
async def visualize_form():
    print('visualization form')

