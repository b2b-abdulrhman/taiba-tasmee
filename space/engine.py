"""Live recitation follower for Tayyibat al-Nashr.

Audio streams in (16 kHz mono). We keep a short window that starts just before the next expected word,
run the Omnilingual CTC model on it, force-align the expected letters (prefix alignment: the reciter may be
mid-verse) and judge every word the reciter has clearly moved past:
  * goodness   = how much worse the expected letters fit than the model's free best guess (per letter)
  * heard text = what the model hears over that word's frames, compared with the text (and the sheikh's reading)
A word is a mistake only when both say so, and the same mistake shows up in two checks in a row."""
import json, re, time
import numpy as np
import onnxruntime as ort
from numba import njit

FR = 0.02
SR = 16000
DIAC = re.compile('[ؐ-ًؚ-ٰٟۖ-ۭـ‌‍‎‏]')
AR = 'ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي'

def letters(w):
    return re.sub('[^ء-ي]', '', DIAC.sub('', w or ''))

def norm(w):
    w = re.sub('[أإآٱ]', 'ا', w).replace('ى', 'ي').replace('ة', 'ه').replace('ؤ', 'و').replace('ئ', 'ي').replace('ء', '')
    return w

def ratio(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    # Ratcliff/Obershelp like difflib, cheap enough for words
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio()

# ---------------------------------------------------------------- model
class Model:
    def __init__(self, model_dir, threads=2):
        toks = {}
        for line in open(model_dir + '/tokens.txt', encoding='utf-8'):
            t, i = line.rstrip('\n').rsplit(' ', 1); toks[int(i)] = t
        inv = {t: i for i, t in toks.items()}
        self.keep = np.array([0, inv['|'], inv[' ']] + [inv[c] for c in AR])
        self.names = ['<blk>', '|', ' '] + list(AR)
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_dir + '/model.int8.onnx', so, providers=['CPUExecutionProvider'])

    def logprobs(self, x):
        x = x.astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        lg = self.sess.run(None, {'x': x[None]})[0][0].astype(np.float32)
        lg -= lg.max(-1, keepdims=True)
        lsm = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        return lsm[:, self.keep]

# ---------------------------------------------------------------- letter classes
ALEF = 'اأإآ'
CLASS = {'ا': ALEF + 'ىء', 'أ': ALEF + 'ء', 'إ': ALEF + 'ء', 'آ': ALEF + 'ء', 'ء': 'ءأإئؤ', 'ة': 'ةهت', 'ه': 'هة',
         'ى': 'ىيا', 'ي': 'يى', 'ئ': 'ئيء', 'ؤ': 'ؤوء', 'ت': 'تة', 'ذ': 'ذز', 'ث': 'ثس', 'ظ': 'ظضذ', 'ض': 'ضظ'}
OPTIONAL = set('اأإآء')
NAMES = ['<blk>', '|', ' '] + list(AR)
COL = {c: i for i, c in enumerate(NAMES)}
_classes, _cid = [], {}
def cid(ch):
    if ch not in _cid:
        members = ['|', ' '] if ch == '|' else list(CLASS.get(ch, ch))
        _cid[ch] = len(_classes); _classes.append(np.array([COL[m] for m in members if m in COL], np.int64))
    return _cid[ch]
for _c in AR + '|': cid(_c)

def emissions(lp):
    E = np.empty((lp.shape[0], len(_classes)), np.float32)
    for c, cols in enumerate(_classes): E[:, c] = lp[:, cols].max(1)
    return E

@njit(cache=True)
def viterbi_prefix(E, blank, garb, lab, opt):
    """Lead garbage state 0, then CTC states 1..2L+1. The path may END in any state (the reciter is mid-way)."""
    T = E.shape[0]; L = len(lab); NS = 2 * L + 2
    NEG = -1e30
    dp = np.full(NS, NEG, np.float64); bp = np.zeros((T, NS), np.int32)
    dp[0] = garb[0]; dp[1] = blank[0]
    k = 0
    while k < L:
        dp[2 + 2 * k] = E[0, lab[k]]
        if not opt[k]: break
        k += 1
    for t in range(1, T):
        nd = np.full(NS, NEG, np.float64)
        for s in range(NS):
            best = dp[s]; arg = s
            if s >= 1:
                k2 = s - 1
                if k2 % 2 == 0:
                    if s - 1 >= 1 and dp[s - 1] > best: best = dp[s - 1]; arg = s - 1
                    if s == 1 and dp[0] > best: best = dp[0]; arg = 0
                else:
                    li = (k2 - 1) // 2
                    if dp[s - 1] > best: best = dp[s - 1]; arg = s - 1
                    if li == 0 and dp[0] > best: best = dp[0]; arg = 0
                    pj = li - 1
                    while pj >= 0:
                        ps = 2 + 2 * pj
                        if lab[pj] != lab[li] and dp[ps] > best: best = dp[ps]; arg = ps
                        if pj < li - 1 and dp[ps + 1] > best: best = dp[ps + 1]; arg = ps + 1
                        if not opt[pj]: break
                        pj -= 1
                    if pj < 0 and li > 0:
                        if dp[0] > best: best = dp[0]; arg = 0
                        if dp[1] > best: best = dp[1]; arg = 1
            if s == 0: e = garb[t]
            elif (s - 1) % 2 == 0: e = blank[t]
            else: e = E[t, lab[(s - 2) // 2]]
            nd[s] = best + e; bp[t, s] = arg
        dp = nd
    s = 0; best = dp[0]
    for q in range(1, NS):
        if dp[q] > best: best = dp[q]; s = q
    path = np.empty(T, np.int32)
    for t in range(T - 1, -1, -1):
        path[t] = s; s = bp[t, s]
    return path

# ---------------------------------------------------------------- the poem
class Poem:
    def __init__(self, data_dir):
        data = json.load(open(data_dir + '/taiba_data.json', encoding='utf-8'))
        timing = json.load(open(data_dir + '/timing.json'))['verses']
        alts = json.load(open(data_dir + '/alts.json', encoding='utf-8'))
        self.V = data['verses']; self.N = len(self.V)
        self.W = []; self.first = [0] * (self.N + 2)
        for n in range(1, self.N + 1):
            self.first[n] = len(self.W); k = 0
            for h, hem in enumerate(self.V[n - 1]):
                for disp in hem.split():
                    L = letters(disp)
                    if L:
                        al = alts.get(str(n), {}).get(str(k), '')
                        s, e = timing[n - 1][k] if k < len(timing[n - 1]) else (0, 0)
                        self.W.append(dict(n=n, h=h, k=k, disp=disp, L=L, key=norm(L), alt=norm(al), s=s, e=e)); k += 1
                    elif self.W and self.W[-1]['n'] == n:
                        self.W[-1]['disp'] += ' ' + disp
        self.first[self.N + 1] = len(self.W)

    def hem_start(self, i):
        j = i
        while j > 0 and self.W[j - 1]['n'] == self.W[i]['n'] and self.W[j - 1]['h'] == self.W[i]['h']: j -= 1
        return j

# ---------------------------------------------------------------- follower
GOP_THR = -1.5      # calibrated on the sheikh's recording: ~90% of mistakes caught, ~0.5% false stops
HEARD_THR = 0.5
AHEAD = 10          # expected words considered per check
MAXWIN = 6.0        # seconds of audio per check at most

class Follower:
    def __init__(self, model, poem, start_word, strict='normal'):
        self.m, self.p = model, poem
        self.ptr = start_word            # next word to confirm
        self.free_from = start_word      # earlier words he may repeat (after a correction)
        self.buf = np.zeros(0, np.float32); self.buf_t0 = 0.0   # stream time of buf[0]
        self.anchor = 0.0                # stream time where word `ptr` (or the restart point) begins, roughly
        self.pending = None
        self.lost = 0                    # >0: we lost track; the next few words may be skipped silently
        self.forgiven = set()            # words he was already stopped on once: never stop twice on the same word
        self.set_strict(strict)
        self.last_check = 0.0; self.compute = []

    def set_strict(self, strict):
        self.gop_thr, self.heard_thr = {'strict': (-1.0, 0.6), 'normal': (GOP_THR, HEARD_THR), 'lenient': (-2.5, 0.4)}.get(strict, (GOP_THR, HEARD_THR))

    @property
    def now(self): return self.buf_t0 + len(self.buf) / SR

    def feed(self, pcm):
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])

    def resume(self, word):
        """after a correction: forget what was heard, expect `word` (or a restart of its hemistich)"""
        self.buf_t0 = self.now; self.buf = np.zeros(0, np.float32)
        self.anchor = self.now; self.ptr = word; self.free_from = self.p.hem_start(word); self.pending = None

    def step(self):
        """run one check; returns a list of events"""
        ev = []
        if self.ptr >= len(self.p.W): return [{'type': 'done'}]
        a = max(self.buf_t0, self.anchor - 0.3)
        if self.now - a > MAXWIN:
            # stuck: the start of the expected word is leaving the window, so resync instead of waiting forever
            a = self.now - MAXWIN; self.anchor = max(self.anchor, a); self.lost = 4; self.pending = None
        if self.now - a < 0.9: return ev
        seg = self.buf[int((a - self.buf_t0) * SR):]
        if seg.std() < 2e-3: return ev                     # silence
        t1 = time.time()
        lp = self.m.logprobs(seg)
        # expected letters: (optional) words he may repeat, then the words ahead
        W = self.p.W
        i0 = min(self.free_from, self.ptr); i1 = min(len(W), self.ptr + AHEAD)
        lab, opt, owner = [], [], []
        for wi in range(i0, i1):
            if wi > i0: lab.append(cid('|')); opt.append(True); owner.append(-1)
            for ch in W[wi]['L']:
                skippable = wi < self.ptr or wi < self.ptr + self.lost
                lab.append(cid(ch)); opt.append(skippable or (ch in OPTIONAL and len(W[wi]['L']) > 1)); owner.append(wi)
        E = emissions(lp); blank = lp[:, 0]; garb = lp.max(1) - 3.0
        path = viterbi_prefix(E, blank, garb, np.array(lab, np.int64), np.array(opt, np.bool_))
        self.compute.append(time.time() - t1)
        # frames per word, emission along the path
        best = lp.max(1)
        fw = np.full(len(path), -1); emit = np.empty(len(path), np.float32); cur = -1
        for t, s in enumerate(path):
            if s == 0: emit[t] = garb[t]; continue
            k = s - 1
            if k % 2 == 0: emit[t] = blank[t]; fw[t] = cur
            else:
                li = (k - 1) // 2; o = owner[li]; emit[t] = E[t, lab[li]]
                if o >= 0: cur = o; fw[t] = o
        reached = cur                                   # furthest word the alignment got to
        if reached >= self.ptr and reached != getattr(self, '_last_reach', -1):
            self._last_reach = reached
            ev.append({'type': 'reach', 'word': int(reached)})   # for the screen: where he is right now
        am = lp.argmax(1)
        verdicts = []
        for wi in range(self.ptr, max(self.ptr, reached)):  # words he has moved past
            fr = np.where(fw == wi)[0]
            nxt = np.where(fw > wi)[0]
            if len(nxt) < 5: break                          # not clearly past it yet
            L = W[wi]['L']
            g = float((emit[fr] - best[fr]).sum() / max(len(L), 1)) if len(fr) else -99.0
            heard = ''; prev = -1
            for t in fr:
                c = int(am[t])
                if c != prev and c >= 3: heard += NAMES[c]
                prev = c
            heard = norm(heard)
            sim = max(ratio(W[wi]['key'], heard), ratio(W[wi]['alt'], heard) if W[wi]['alt'] else 0.0)
            if self.lost and wi < self.ptr + self.lost and len(fr) < 2:
                verdicts.append((wi, False, a + nxt[0] * FR, a + nxt[0] * FR, 0.0, 1.0, '', False)); continue   # lost word: no judgement
            squeezed = len(fr) < 1.6 * len(L)               # far too short for its letters: skipped
            bad = (g < self.gop_thr and sim < self.heard_thr) or (squeezed and g < self.gop_thr * 0.6)
            strong = bad and (g < 2 * self.gop_thr or squeezed) and sim < 0.35
            end_t = a + (fr[-1] + 1) * FR if len(fr) else a + nxt[0] * FR
            start_t = a + fr[0] * FR if len(fr) else end_t
            verdicts.append((wi, bad, start_t, end_t, g, sim, heard, strong))
        err = next((v for v in verdicts if v[1] and v[0] not in self.forgiven), None)
        if err:
            if self.pending == err[0] or err[7]:
                ev.append({'type': 'error', 'word': err[0], 'heard': err[6], 'gop': round(err[4], 2), 'sim': round(err[5], 2)})
                self.pending = None; self.ptr = err[0]; self.forgiven.add(err[0])
            else:
                self.pending = err[0]
            # confirm the good words before the mistake
            good = [v for v in verdicts if v[0] < err[0]]
        else:
            self.pending = None
            good = verdicts[:-1] if len(verdicts) > 1 else []   # keep the last judged word in the window
        if good and not any(x['type'] == 'error' for x in ev):
            last = good[-1]
            self.ptr = last[0] + 1; self.free_from = self.ptr; self.anchor = last[3]; self.lost = 0
            # drop old audio
            cut = max(0.0, self.anchor - 1.0 - self.buf_t0)
            if cut > 0: self.buf = self.buf[int(cut * SR):]; self.buf_t0 += cut
            ev.append({'type': 'progress', 'ptr': self.ptr})
        return ev
