"""Hugging Face Space: live review of Tayyibat al-Nashr for a blind reciter.
The phone streams microphone audio over a WebSocket; the server follows the recitation and, on a clear
mistake, tells the page which word to play (in Sheikh Ayman Suwaid's voice, served one word at a time)."""
import asyncio, io, json, os, subprocess, time, wave
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, JSONResponse
from engine import Model, Poem, Follower, SR

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('MODEL_DIR', '/models/omni')
THREADS = int(os.environ.get('THREADS', '2'))

model = Model(MODEL_DIR, threads=THREADS)
poem = Poem(os.path.join(HERE, 'data'))
app = FastAPI()

# ------------------------------------------------------------ the sheikh's recording (kept private)
AUDIO_SR = 24000
audio = None
def load_audio():
    """From a private Hugging Face dataset (AUDIO_REPO + AUDIO_FILE, read with the HF_TOKEN secret),
    or a local file (AUDIO_PATH) when testing. Only single words / verses are ever sent out."""
    global audio
    path = os.environ.get('AUDIO_PATH')
    if not path and os.environ.get('AUDIO_REPO'):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(repo_id=os.environ['AUDIO_REPO'], filename=os.environ.get('AUDIO_FILE', 'taiba-ayman-suwaid.m4a'),
                                   repo_type='dataset', token=os.environ.get('HF_TOKEN'))
        except Exception as e:
            print('audio download failed:', e, flush=True); return
    if not path or not os.path.exists(path): print('no sheikh audio configured', flush=True); return
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(AUDIO_SR), '-f', 's16le', '-'],
                         capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, np.int16)
    print(f'sheikh audio loaded: {len(audio)/AUDIO_SR/60:.1f} min', flush=True)

def wav_bytes(a, b):
    seg = audio[max(0, int(a * AUDIO_SR)):int(b * AUDIO_SR)]
    # short fades so clips don't click
    seg = seg.astype(np.float32); n = min(len(seg) // 4, int(0.012 * AUDIO_SR))
    if n > 0: seg[:n] *= np.linspace(0, 1, n); seg[-n:] *= np.linspace(1, 0, n)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(AUDIO_SR); w.writeframes(seg.astype(np.int16).tobytes())
    return buf.getvalue()

def warm():
    # warm up the model and the aligner so the first check is fast
    f = Follower(model, poem, 0); f.feed(np.random.randn(SR * 2).astype(np.float32) * 0.01); f.anchor = 0; f.step()
    print('ready', flush=True)

@app.on_event('startup')
async def startup():
    if os.environ.get('SKIP_STARTUP'): return       # the Space bootstrap calls load_audio()/warm() itself
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, load_audio)
    await loop.run_in_executor(None, warm)

# ------------------------------------------------------------ http
@app.get('/')
def index(): return FileResponse(os.path.join(HERE, 'static', 'index.html'), headers={'Cache-Control': 'no-cache'})

@app.get('/manifest.webmanifest')
def manifest(): return FileResponse(os.path.join(HERE, 'static', 'manifest.webmanifest'))

@app.get('/icon.png')
def icon(): return FileResponse(os.path.join(HERE, 'static', 'icon.png'))

@app.get('/health')
def health(): return {'ok': True, 'audio': audio is not None}

@app.get('/clip/{i}')
def clip(i: int):
    if audio is None or not (0 <= i < len(poem.W)): return Response(status_code=404)
    w = poem.W[i]
    return Response(wav_bytes(w['s'] - 0.03, w['e'] + 0.05), media_type='audio/wav', headers={'Cache-Control': 'public, max-age=86400'})

@app.get('/verse/{n}')
def verse(n: int):
    if audio is None or not (1 <= n <= poem.N): return Response(status_code=404)
    a = poem.W[poem.first[n]]['s'] - 0.05; b = poem.W[poem.first[n + 1] - 1]['e'] + 0.1
    return Response(wav_bytes(a, b), media_type='audio/wav', headers={'Cache-Control': 'public, max-age=86400'})

# ------------------------------------------------------------ live following
@app.websocket('/ws')
async def ws(sock: WebSocket):
    await sock.accept()
    loop = asyncio.get_running_loop()
    f = None; paused = True; busy = False; last = 0.0
    try:
        while True:
            msg = await sock.receive()
            if msg.get('type') == 'websocket.disconnect': break
            if msg.get('text'):
                m = json.loads(msg['text'])
                if m['type'] == 'start':
                    f = Follower(model, poem, int(m['word']), m.get('strict', 'normal')); paused = False; last = 0.0
                    await sock.send_json({'type': 'listening'})
                elif m['type'] == 'resume' and f:
                    f.resume(int(m['word'])); paused = False; last = f.now
                elif m['type'] == 'pause':
                    paused = True
                elif m['type'] == 'strict' and f:
                    f.set_strict(m['value'])
                continue
            data = msg.get('bytes')
            if not data or f is None or paused: continue
            f.feed(np.frombuffer(data, np.int16).astype(np.float32) / 32768.0)
            if busy or f.now - last < 0.35: continue
            busy = True; last = f.now
            try:
                events = await loop.run_in_executor(None, f.step)
            finally:
                busy = False
            for e in events:
                if e['type'] == 'error':
                    paused = True            # the page plays the word, then sends "resume"
                    w = poem.W[e['word']]; e.update(n=w['n'], disp=w['disp'])
                await sock.send_json(e)
    except WebSocketDisconnect:
        pass
