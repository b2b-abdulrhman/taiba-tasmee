import sys, json, random, time, numpy as np, soundfile as sf, glob
sys.path.insert(0, '/home/claude/space')
from engine import Model, Poem, Follower, SR, ratio, letters
MD = glob.glob('/home/claude/models/sherpa-onnx-omni*/')[0].rstrip('/')
model = Model(MD, threads=2); poem = Poem('/home/claude/space/data')
x, _ = sf.read('/home/claude/audio/full16k.wav', dtype='float32')
def clip(i): w = poem.W[i]; return x[int(w['s'] * SR):int(w['e'] * SR)]

def run(v0, v1, seed, n_err=6, strict='normal'):
    rnd = random.Random(seed)
    a, b = poem.first[v0], poem.first[v1 + 1]
    planted = {}
    cand = list(range(a + 3, b - 3)); rnd.shuffle(cand)
    for i in cand:
        if len(planted) >= n_err: break
        if any(abs(i - j) < 6 for j in planted): continue
        if rnd.random() < 0.65:   # substitution by a clearly different word
            while True:
                j = rnd.randrange(len(poem.W))
                if ratio(poem.W[j]['key'], poem.W[i]['key']) < 0.5: break
            planted[i] = ('sub', j)
        else: planted[i] = ('del', None)
    f = Follower(model, poem, a, strict)
    i = a; caught = {}; false_stops = []; lat = []
    feed_t = 0.0; last_step = 0.0
    while i < b:
        kind = planted.get(i)
        audio = clip(kind[1]) if kind and kind[0] == 'sub' else (np.zeros(0, np.float32) if kind else clip(i))
        planted_here = i
        # stream this word in 100 ms chunks
        for c0 in range(0, max(1, len(audio)), 1600):
            f.feed(audio[c0:c0 + 1600])
            if f.now - last_step >= 0.4:
                last_step = f.now
                for e in f.step():
                    if e['type'] == 'error':
                        w = e['word']
                        if w in planted and w not in caught:
                            caught[w] = f.now; lat.append(f.now - err_end.get(w, f.now))
                            planted.pop(w)
                        else: false_stops.append((w, poem.W[w]['disp'], e))
                        f.resume(w); i = w; raise_flag = True
                        break
                else: continue
                break
            else: continue
        else:
            if kind: err_end[planted_here] = f.now
            i += 1; continue
        # an error was flagged: continue from the corrected word (loop picks it up)
        continue
    # flush: a little silence at the end
    for _ in range(10):
        f.feed(np.zeros(1600, np.float32))
        for e in f.step():
            if e['type'] == 'error':
                w = e['word']
                if w in planted: caught[w] = f.now; planted.pop(w)
                else: false_stops.append((w, poem.W[w]['disp'], e))
    return caught, planted, false_stops, lat, f.compute

err_end = {}
tot_c = tot_m = tot_f = 0; lats = []; comp = []; words = 0
for k, (v0, v1) in enumerate([(200, 209), (520, 529), (880, 889)]):
    err_end.clear()
    c, missed, fs, lat, cp = run(v0, v1, seed=k + 1)
    words += poem.first[v1 + 1] - poem.first[v0]
    tot_c += len(c); tot_m += len(missed); tot_f += len(fs); lats += lat; comp += cp
    print(f'verses {v0}-{v1}: caught {len(c)}, missed {len(missed)} {[(poem.W[w]["disp"], kd[0]) for w, kd in missed.items()]}, false stops {len(fs)} {[(d, e["heard"], e["gop"], e["sim"]) for _, d, e in fs]}')
print(f'TOTAL: caught {tot_c}/{tot_c + tot_m}, false stops {tot_f} in {words} words; delay after the mistake: median {np.median(lats):.1f}s, max {max(lats):.1f}s; compute per check median {np.median(comp):.2f}s max {max(comp):.2f}s')
