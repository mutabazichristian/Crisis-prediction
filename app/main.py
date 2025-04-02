from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title='Crashout 2')

app.mount('/static',StaticFiles(directory="static"),name='static')
template = Jinja2Templates(directory="templates")

from routers import predict, train, visualize

app.include_router(predict.router)
app.include_router(train.router)
app.include_router(visualize.router)

@app.get('/')
async def home():
    """Home pageeeeee"""
