"""Hugging Face Space: live review of Tayyibat al-Nashr for a blind reciter.
The phone streams microphone audio over a WebSocket; the server follows the recitation and, on a clear
mistake, tells the page which word to play (in Sheikh Ayman Suwaid's voice, served one word at a time)."""
import asyncio, io, json, os, subprocess, time, wave
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, JSONResponse
from engine import Model, Poem, Follower, SR

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get('MODEL_DIR', '/models/omni')
THREADS = int(os.environ.get('THREADS', str(min(4, os.cpu_count() or 2))))

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

def wav_bytes(a, b, tempo=1.0):
    seg = audio[max(0, int(a * AUDIO_SR)):int(b * AUDIO_SR)]
    if abs(tempo - 1.0) > 0.01:                     # slower, same pitch
        seg = np.frombuffer(subprocess.run(['ffmpeg', '-v', 'error', '-f', 's16le', '-ar', str(AUDIO_SR), '-ac', '1', '-i', '-',
                                            '-af', f'atempo={tempo:.2f}', '-f', 's16le', '-'],
                                           input=seg.tobytes(), capture_output=True).stdout, np.int16)
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

STEP_TIMES = deque(maxlen=400)

@app.get('/health')
def health(): return {'ok': True, 'audio': audio is not None, 'threads': THREADS, 'cpus': os.cpu_count()}

@app.get('/stats')
def stats():
    t = sorted(STEP_TIMES)
    return {'checks': len(t), 'median_s': round(t[len(t) // 2], 3) if t else None, 'p90_s': round(t[int(len(t) * .9)], 3) if t else None, 'threads': THREADS}

@app.get('/clip/{i}')
def clip(i: int, n: int = 1, tempo: float = 1.0):
    """word i and up to n-1 following words of the same verse, optionally slowed down"""
    if audio is None or not (0 <= i < len(poem.W)): return Response(status_code=404)
    w = poem.W[i]; j = i
    while j + 1 < len(poem.W) and j + 1 < i + max(1, min(n, 5)) and poem.W[j + 1]['n'] == w['n']: j += 1   # stay inside the verse
    tempo = max(0.6, min(1.0, tempo))
    return Response(wav_bytes(w['s'] - 0.03, poem.W[j]['e'] + 0.05, tempo), media_type='audio/wav', headers={'Cache-Control': 'public, max-age=86400'})

@app.get('/verse/{n}')
def verse(n: int):
    if audio is None or not (1 <= n <= poem.N): return Response(status_code=404)
    a = poem.W[poem.first[n]]['s'] - 0.05; b = poem.W[poem.first[n + 1] - 1]['e'] + 0.1
    return Response(wav_bytes(a, b), media_type='audio/wav', headers={'Cache-Control': 'public, max-age=86400'})

# ------------------------------------------------------------ live following
@app.websocket('/ws')
async def ws(sock: WebSocket):
    """Receiving audio and checking it run side by side: a check always looks at the newest audio,
    so a slow check never makes the follower fall behind the reciter."""
    await sock.accept()
    loop = asyncio.get_running_loop()
    st = {'f': None, 'paused': True, 'alive': True}

    async def worker():
        last = 0.0; cur = None
        while st['alive']:
            f = st['f']
            if f is not cur: cur = f; last = 0.0          # a new start: forget the previous session's clock
            if f is None or st['paused'] or f.now - last < 0.2:
                await asyncio.sleep(0.05); continue
            last = f.now
            t0 = time.time()
            events = await loop.run_in_executor(None, f.step)
            STEP_TIMES.append(time.time() - t0)
            if st['f'] is not f: continue          # restarted meanwhile
            for e in events:
                if e['type'] == 'error':
                    st['paused'] = True             # the page plays the word, then sends "resume"
                    w = poem.W[e['word']]; e.update(n=w['n'], disp=w['disp'])
                try: await sock.send_json(e)
                except Exception: return
                if e['type'] == 'error': break

    task = asyncio.create_task(worker())
    try:
        while True:
            msg = await sock.receive()
            if msg.get('type') == 'websocket.disconnect': break
            if msg.get('text'):
                m = json.loads(msg['text'])
                if m['type'] == 'start':
                    st['f'] = Follower(model, poem, int(m['word']), m.get('strict', 'normal')); st['paused'] = False
                    await sock.send_json({'type': 'listening'})
                elif m['type'] == 'resume' and st['f']:
                    st['f'].resume(int(m['word'])); st['paused'] = False
                elif m['type'] == 'pause':
                    st['paused'] = True
                elif m['type'] == 'strict' and st['f']:
                    st['f'].set_strict(m['value'])
                continue
            data = msg.get('bytes')
            if not data or st['f'] is None or st['paused']: continue
            if os.environ.get('DUMP_AUDIO'):
                with open(os.environ['DUMP_AUDIO'], 'ab') as fh: fh.write(data)
            st['f'].feed(np.frombuffer(data, np.int16).astype(np.float32) / 32768.0)
    except WebSocketDisconnect:
        pass
    finally:
        st['alive'] = False; task.cancel()
