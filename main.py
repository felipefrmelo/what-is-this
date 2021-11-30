from fastapi.param_functions import Depends
from inference import predict
from fastapi import FastAPI, File, UploadFile

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from dataclasses import dataclass
import wikipediaapi
from i18 import i18
wiki_wiki = wikipediaapi.Wikipedia('en')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@dataclass
class Result:
    title: str
    summary: str
    url: str


def get_wiki_page(query, lang):
    page = wiki_wiki.page(query)
    traduction = page.langlinks.get(lang)
    result = traduction if traduction else page
    return Result(result.title, result.summary, result.fullurl)


def extract_language(request: Request):
    return request.headers.get("accept-language", 'en').split(",")[0].split("-")[0].lower()


def save_image(file: UploadFile):
    img = Image.open(file.file)
    img.save("static/img/" + file.filename)
    img.close()


def renderTemplate(request: Request,  lang=Depends(extract_language)):
    def wrapper(filename,  result: Result = None, html="index.html"):
        return templates.TemplateResponse(html,
                                          {"request": request,
                                           "result": result,
                                           "filename": filename,
                                           "i18": i18.get(lang, "pt")
                                           })
    return wrapper


@app.post("/predict/")
async def create_upload_file(render=Depends(renderTemplate),
                             file: UploadFile = File(...),
                             lang=Depends(extract_language)):

    try:
        save_image(file)
        prediction = predict(file.file)
        result = get_wiki_page(prediction, lang)
        return render(file.filename, result, "predict.html")
    except Exception as e:
        return render(file.filename)


@app.get("/")
async def index(render=Depends(renderTemplate)):

    return render(filename="default.jpg")
