import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import tempfile, uvicorn
from parser import parse_slip, extract_lines, group_lines

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
        'accuracy': '96.8% on 7 supported books'
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

        # Debug: log raw OCR to Railway logs
        lines = extract_lines(tmp_path)
        groups = group_lines(lines, threshold=20)
        print(f"\n{'='*60}")
        print(f"PARSE REQUEST: sportsbook={sportsbook} file={file.filename}")
        print(f"{'='*60}")
        for i, g in enumerate(groups):
            text = ' '.join(item['text'] for item in g)
            print(f"  {i:02d} [y={g[0]['y']:.0f}] {text}")
        print(f"{'='*60}")

        result = parse_slip(tmp_path, sportsbook=sportsbook)
        os.unlink(tmp_path)

        print(f"RESULT: {len(result.get('selections', result.get('picks', [])))} selections")
        print(f"  bet_type={result.get('bet_type')} odds={result.get('total_odds')}")
        for s in (result.get('selections') or result.get('picks') or []):
            print(f"  -> {s}")
        print(f"{'='*60}\n", flush=True)

        return {'success': True, 'data': result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

@app.post('/debug')
async def debug_endpoint(
    file: UploadFile = File(...),
    sportsbook: str = Form(None)
):
    try:
        contents = await file.read()
        suffix = '.' + file.filename.split('.')[-1] if file.filename else '.jpg'
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Raw OCR lines
        lines = extract_lines(tmp_path)

        # Grouped text
        groups = group_lines(lines, threshold=20)
        grouped = []
        for g in groups:
            text = ' '.join(item['text'] for item in g)
            y = g[0]['y']
            grouped.append({'y': round(y), 'text': text})

        # Parsed result
        result = parse_slip(tmp_path, sportsbook=sportsbook)
        os.unlink(tmp_path)

        return {
            'success': True,
            'raw_lines': [{'text': l['text'], 'y': round(l['y']), 'x': round(l['x']), 'conf': round(l['conf'], 2)} for l in lines],
            'grouped': grouped,
            'parsed': result
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
