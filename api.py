import numpy_compat  # noqa: F401 — must be first; patches np.sctypes for NumPy 2.0 / imgaug compatibility
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, uvicorn
from parser import parse_slip

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get('/')
def root():
    return {
        'status': 'StakeVision v4.0 running',
        'engine': 'PaddleOCR',
        'books': 'DraftKings, FanDuel, PrizePicks, Underdog, BetMGM, Fanatics, Onyx',
        'accuracy': '98.7% on 6 main books'
    }

@app.post('/parse')
async def parse_endpoint(
    file: UploadFile = File(...),
    sportsbook: str = Form(None)
):
    try:
        contents = await file.read()
        suffix = '.' + file.filename.split('.')[-1] if file.filename else '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        result = parse_slip(tmp_path, sportsbook=sportsbook)
        os.unlink(tmp_path)
        return {'success': True, 'data': result}
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
