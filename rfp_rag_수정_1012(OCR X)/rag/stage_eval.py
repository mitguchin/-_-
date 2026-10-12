"""단계별 평가 — 검색(01) · 생성(02·03) · End-to-End(04).   GPT 호출 없음 → $0

    01_rag_build      : 검색 과정 자체 평가   evaluate_retrieval · retrieval_summary · score_separation
    02_generate_eval  : 생성 과정 자체 평가   gen_checks · gen_table
    03_chatbot        : 생성 과정 자체 평가   gen_checks (턴마다) · 대본 대화(DEFAULT_RFP_CONVERSATIONS)
    04_e2e_eval       : 01·02·03 결과를 합쳐 단계별 종합 · 실패 원인 분류   attribute · stage_summary

결과 파일은 모두 eval/results/ 에 JSON 으로 저장합니다 (save_result / load_result).
이 모듈은 pandas · numpy · 표준 라이브러리만 씁니다 (rag 의 다른 모듈에 의존하지 않음).
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import time
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

RESULTS_DIR = "eval/results"
RETRIEVAL_GOLD_PATH = "eval/retrieval_golden.jsonl"

# ═══════════════════════════════════════════════════════════════════════ 정답 세트 (RFP)
#   docs     : 정답 문서 목록. 문서 하나 = 파일명에 모두 들어 있어야 하는 조각 목록. 조각 안의 'A|B' = 둘 중 하나
#   keywords : 검색된 문단(근거)에 들어 있어야 할 말 ('A|B' = 둘 중 하나) — 생성 평가에서는 핵심 사실로도 씀
#   type     : in(문서 안 질문) | ood(문서 밖 질문 — 정답 문서 없음, 점수 분리·거절 평가용)
DEFAULT_RFP_GOLD = [
    dict(id="R01", question="국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘.",
         docs=[["국민연금공단", "이러닝"]], keywords=["요구사항", "이러닝"]),
    dict(id="R02", question="국민연금공단 이러닝시스템 사업의 콘텐츠 개발 관리 요구사항을 자세히 알려 줘.",
         docs=[["국민연금공단", "이러닝"]], keywords=["콘텐츠", "개발"]),
    dict(id="R03", question="기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?",
         docs=[["기초과학연구원|중이온가속기"]], keywords=["극저온", "예측|AI|인공지능"]),
    dict(id="R04", question="기초과학연구원 극저온시스템 사업에서 모니터링 업무에 대한 요청사항이 있나?",
         docs=[["기초과학연구원|중이온가속기"]], keywords=["모니터링"]),
    dict(id="R05", question="한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘.",
         docs=[["한국원자력연구원"]], keywords=["선량", "목적|배경|필요"]),
    dict(id="R06", question="고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?",
         docs=[["고려대학교", "포털"], ["광주과학기술원", "학사"]], keywords=["포털", "학사"]),
    dict(id="R07", question="고려대학교랑 광주과학기술원 각각 응답 시간에 대한 요구사항이 있나? 문서를 기반으로 정확하게 답변해 줘.",
         docs=[["고려대학교", "포털"], ["광주과학기술원", "학사"]], keywords=["응답시간|응답 시간|응답속도"]),
    dict(id="R08", question="고려대 차세대 포탈 학사시스템 구축 사업 예산이 얼마야?",          # 줄임말 · 오타 (GPT 해석 없이 버티는지)
         docs=[["고려대학교", "포털"]], keywords=["예산|사업비|금액|원"]),
    dict(id="X01", question="월세액 세액공제 한도가 얼마예요?", docs=[], keywords=[], type="ood"),
    dict(id="X02", question="연말정산은 언제 하나요?", docs=[], keywords=[], type="ood"),
    dict(id="X03", question="오늘 서울 날씨 어때?", docs=[], keywords=[], type="ood"),
]

# 03 대본 대화 — 같은 대화 안에서는 앞 질문을 기억. gold = 위 정답 세트의 id (후속 질문은 '혼자 읽어도 되는' 질문의 정답을 씀)
DEFAULT_RFP_CONVERSATIONS = [
    dict(cid="C1", turns=[
        dict(q="국민연금공단이 발주한 이러닝시스템 관련 사업 요구사항을 정리해 줘.", gold="R01"),
        dict(q="콘텐츠 개발 관리 요구 사항에 대해서 더 자세히 알려 줘.", gold="R02"),
        dict(q="교육이나 학습 관련해서 다른 기관이 발주한 사업은 없나?", gold=None),
    ]),
    dict(cid="C2", turns=[
        dict(q="기초과학연구원 극저온시스템 사업 요구에서 AI 기반 예측에 대한 요구사항이 있나?", gold="R03"),
        dict(q="그럼 모니터링 업무에 대한 요청사항이 있는지 찾아보고 알려 줘.", gold="R04"),
    ]),
    dict(cid="C3", turns=[
        dict(q="한국 원자력 연구원에서 선량 평가 시스템 고도화 사업을 발주했는데, 이 사업이 왜 추진되는지 목적을 알려 줘.", gold="R05"),
    ]),
    dict(cid="C4", turns=[
        dict(q="고려대학교 차세대 포털 시스템 사업이랑 광주과학기술원의 학사 시스템 기능개선 사업을 비교해 줄래?", gold="R06"),
        dict(q="고려대학교랑 광주과학기술원 각각 응답 시간에 대한 요구사항이 있나? 문서를 기반으로 정확하게 답변해 줘.", gold="R07"),
    ]),
]


# ═══════════════════════════════════════════════════════════════════════ 공용 유틸
def norm(s) -> str:
    """비교용 정규화: NFC · 소문자 · 공백/쉼표/가운뎃점/따옴표 제거"""
    s = unicodedata.normalize("NFC", str(s or "")).lower()
    return re.sub(r"[\s,·ㆍ'\"“”‘’`]+", "", s)


def qkey(q) -> str:
    """질문 매칭 키 (문장부호·공백 차이 무시)"""
    return re.sub(r"[^\w가-힣]+", "", norm(q))


def has_alt(text_norm: str, fact: str) -> bool:
    """fact = 'A|B' → 정규화된 text 에 A 또는 B 가 들어 있는가"""
    return any(norm(a) and norm(a) in text_norm for a in str(fact).split("|"))


def is_num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not (isinstance(x, float) and math.isnan(x))


def mean_or_none(vals):
    v = [float(x) for x in vals if is_num(x)]
    return sum(v) / len(v) if v else None


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return [jsonable(r) for r in o.to_dict("records")]
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    return str(o)


def save_result(name: str, payload: dict, results_dir: str = RESULTS_DIR) -> str:
    """eval/results/<name>.json 로 저장 (임시 파일에 쓰고 교체 → 저장 중 중단돼도 안전)"""
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{name}.json")
    payload = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), **payload}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable(payload), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def load_result(name_or_path: str, results_dir: str = RESULTS_DIR):
    path = name_or_path if name_or_path.endswith(".json") else os.path.join(results_dir, f"{name_or_path}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_results(prefix: str, results_dir: str = RESULTS_DIR):
    return sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.json")), key=os.path.getmtime)


def save_gold(gold, path: str = RETRIEVAL_GOLD_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for g in gold:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    return path


def load_gold(path: str = RETRIEVAL_GOLD_PATH, default=None):
    """저장된 정답 세트가 있으면 그것을, 없으면 default(기본 DEFAULT_RFP_GOLD)"""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    return [dict(g) for g in (DEFAULT_RFP_GOLD if default is None else default)]


def gold_index(gold):
    """id · 질문 → 정답 항목"""
    idx = {}
    for g in gold or []:
        idx[g["id"]] = g
        idx[qkey(g["question"])] = g
    return idx


# ═══════════════════════════════════════════════════════════════════════ 검색 결과 정규화
def _src_name(s) -> str:
    return os.path.basename(str(s or "").replace("\\", "/"))


def hits_from_retriever(hits):
    """retriever.search() · chat.ask()['hits'] → [{source, text, score, section, page}]"""
    out = []
    for h in hits or []:
        out.append(dict(source=_src_name(h.get("source")), text=h.get("body") or h.get("text") or "",
                        score=h.get("score"), section=h.get("section"), page=h.get("page")))
    return out


def hits_from_langchain(pairs):
    """FAISS.similarity_search_with_score() → 같은 형식"""
    out = []
    for doc, score in pairs or []:
        md = getattr(doc, "metadata", {}) or {}
        out.append(dict(source=_src_name(md.get("source")), text=getattr(doc, "page_content", ""),
                        score=float(score), section=md.get("section"), page=md.get("page")))
    return out


def corpus_sources(retriever) -> list:
    """인덱스에 들어 있는 문서(파일) 이름 목록 — 정답 패턴 점검용"""
    names = set()
    by_id = getattr(retriever, "by_id", None) or {}
    vals = by_id.values() if isinstance(by_id, dict) else by_id
    for c in vals:
        src = c.get("source") if isinstance(c, dict) else (getattr(c, "metadata", {}) or {}).get("source")
        if src:
            names.add(_src_name(src))
    if not names:                                              # 대안: FAISS docstore
        store = getattr(retriever, "store", None)
        ds = getattr(getattr(store, "docstore", None), "_dict", {}) or {}
        for d in ds.values():
            src = (getattr(d, "metadata", {}) or {}).get("source")
            if src:
                names.add(_src_name(src))
    return sorted(names)


def text_source_map(retriever) -> dict:
    """청크 본문(정규화) → 문서 이름. 02·03 에서 답변 근거가 어느 문서에서 왔는지 찾을 때 사용"""
    m = {}
    by_id = getattr(retriever, "by_id", None) or {}
    vals = by_id.values() if isinstance(by_id, dict) else by_id
    for c in vals:
        if not isinstance(c, dict):
            continue
        src = _src_name(c.get("source"))
        for k in ("body", "text"):
            if c.get(k):
                m[norm(c[k])[:300]] = src
    return m


def sources_of(contexts, tmap: dict) -> list:
    return [tmap.get(norm(c)[:300]) for c in contexts or []]


# ═══════════════════════════════════════════════════════════════════════ ① 검색 평가 (01)
def doc_match(source: str, pattern) -> bool:
    """pattern = ['국민연금공단', '이러닝'] → 파일명에 모두 들어 있으면 True ('A|B' = 둘 중 하나)"""
    s = norm(source)
    return bool(pattern) and all(has_alt(s, p) for p in pattern)


def check_gold(gold, sources) -> pd.DataFrame:
    """정답 문서 패턴마다 실제로 맞는 문서 수 — 0개면 패턴(파일명 조각)을 고쳐야 함"""
    rows = []
    for g in gold:
        for pat in g.get("docs") or []:
            m = [s for s in sources if doc_match(s, pat)]
            rows.append({"id": g["id"], "정답 문서 패턴": " + ".join(pat), "맞는 문서 수": len(m),
                         "맞는 문서": " / ".join(x[:40] for x in m[:3]) + (" …" if len(m) > 3 else ""),
                         "상태": "✅" if len(m) == 1 else ("⚠️ 여러 개 — 패턴을 좁히세요" if m else "❌ 없음 — 패턴 수정")})
    return pd.DataFrame(rows)


def retrieval_row(item, hits, ks=(1, 3, 5, 10)) -> dict:
    """질문 1개의 검색 결과 → @k 지표"""
    pats = item.get("docs") or []
    kws = item.get("keywords") or []
    srcs = [h["source"] for h in hits]
    rel = [any(doc_match(s, p) for p in pats) for s in srcs]           # 순위별: 정답 문서의 청크인가
    first = next((i + 1 for i, r in enumerate(rel) if r), None)
    row = {"id": item["id"], "질문": item["question"], "type": item.get("type", "in"),
           "첫 정답 순위": first, "MRR": (1.0 / first) if first else 0.0,
           "top1 점수": hits[0].get("score") if hits else None,
           "top 문서": " / ".join(dict.fromkeys(s[:28] for s in srcs[:3]))}
    for k in ks:
        top = hits[:k]
        found = [any(doc_match(h["source"], p) for h in top) for p in pats]
        row[f"Hit@{k}"] = float(any(found)) if pats else None
        row[f"Recall@{k}"] = (sum(found) / len(found)) if pats else None        # 정답 문서 여러 개(비교 질문)일 때 의미
        row[f"Precision@{k}"] = (sum(rel[:k]) / len(top)) if (pats and top) else None
        if kws:
            t = norm(" ".join(h["text"] for h in top))
            row[f"Keyword@{k}"] = sum(has_alt(t, kw) for kw in kws) / len(kws)
        else:
            row[f"Keyword@{k}"] = None
    if item.get("type") == "ood":
        row["MRR"] = None
    return row


def evaluate_retrieval(gold, search_fn, ks=(1, 3, 5, 10), mode: str = "", on_progress=None):
    """search_fn(question, k) → hits_from_* 형식 목록.  → (질문별 DataFrame, 원본 hits dict)"""
    kmax = max(ks)
    rows, raw = [], {}
    for i, g in enumerate(gold):
        t0 = time.time()
        hits = search_fn(g["question"], kmax) or []
        r = retrieval_row(g, hits, ks)
        r["검색(s)"] = round(time.time() - t0, 3)
        r["mode"] = mode
        rows.append(r)
        raw[g["id"]] = hits
        if on_progress:
            on_progress(i + 1, len(gold), g["id"])
    return pd.DataFrame(rows), raw


def retrieval_summary(df: pd.DataFrame, ks=(1, 3, 5, 10)) -> dict:
    """문서 안 질문(type=in)만 평균"""
    d = df[df["type"] != "ood"] if "type" in df else df
    out = {"질문 수": int(len(d)), "MRR": mean_or_none(d["MRR"])}
    for k in ks:
        for m in ("Hit", "Recall", "Precision", "Keyword"):
            c = f"{m}@{k}"
            if c in d:
                out[c] = mean_or_none(d[c])
    out["평균 검색(s)"] = mean_or_none(df.get("검색(s)", []))
    return out


def ranking_signature(raw: dict) -> str:
    """검색 결과 순서 지문 — 설정을 바꿨는데 결과가 그대로인지 확인용"""
    return "|".join(f"{k}:" + ",".join(f"{h['source'][:20]}#{norm(h['text'])[:20]}" for h in v) for k, v in sorted(raw.items()))


def auc(pos, neg):
    """Mann-Whitney AUC: 문서 안 질문 점수가 문서 밖 질문 점수보다 높을 확률"""
    pos = [float(x) for x in pos if is_num(x)]
    neg = [float(x) for x in neg if is_num(x)]
    if not pos or not neg:
        return None
    s = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def score_separation(in_scores, ood_scores) -> dict:
    """관련도 점수로 문서 밖 질문을 가를 수 있는가 + 추천 임계값(문서 안 최솟값과 문서 밖 최댓값 사이)"""
    pin = [float(x) for x in in_scores if is_num(x)]
    pood = [float(x) for x in ood_scores if is_num(x)]
    res = {"문서 안 평균": mean_or_none(pin), "문서 안 최소": min(pin) if pin else None,
           "문서 밖 평균": mean_or_none(pood), "문서 밖 최대": max(pood) if pood else None, "AUC": auc(pin, pood)}
    if pin and pood:
        lo, hi = max(pood), min(pin)
        res["추천 임계값"] = (lo + hi) / 2 if hi > lo else None
        res["판정"] = "✅ 완전히 분리됨" if hi > lo else "⚠️ 겹침 — 점수만으로 문서 밖 질문을 거를 수 없음"
    return res


# ═══════════════════════════════════════════════════════════════════════ ② 생성 평가 (02 · 03)
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_UNIT = (r"(?:%|퍼센트|만\s*원|억\s*원|천\s*원|원|만|억|년|개월|월|일|시간|초|분|명|건|개|배|㎡|회|차|"
         r"gb|tb|mb|ms|mbps|tps|percent)")
_DATE = re.compile(r"\d{1,2}\s*월\s*\d{1,2}\s*일|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}", re.I)
_NUMUNIT = re.compile(rf"(?:{_NUM})\s*{_UNIT}", re.I)

REFUSAL = re.compile(r"(확인할\s*수\s*없|찾을\s*수\s*없|문서에\s*(?:는\s*)?(?:없|나와\s*있지\s*않|포함되어\s*있지\s*않)|"
                     r"제공된\s*(?:문서|자료|맥락)에(?:서)?(?:는)?\s*(?:관련|해당)?.{0,12}(?:없|않)|알\s*수\s*없|"
                     r"정보가\s*없|답변(?:드리기|하기)\s*어렵)")
PREMISE_FIX = re.compile(r"(아니(?:요|에요|오|다|며|고)|아닙니다|아님|아닌|사실과\s*다르|사실이\s*아니|틀린|틀립|잘못|정정|바로잡|"
                         r"그렇지\s*않|않습니다|없습니다|\bno\b|\bnot\b|incorrect|actually)", re.I)


def premise_corrected(answer) -> bool:
    """틀린 전제 질문('3% 맞죠?')에 답변이 전제를 바로잡았는가 (아닙니다 · 사실과 다릅니다 …)"""
    return bool(PREMISE_FIX.search(str(answer or "")))


OUT_OF_DOC = re.compile(r"\[\s*문서\s*외|문서\s*외\s*일반\s*지식|일반\s*지식\s*(?:기준|으로)")


def extract_facts(text: str, limit: int = 8) -> list:
    """정답·답변에서 숫자 사실(금액·비율·기간·날짜) 뽑기 → 정규화 문자열"""
    t = unicodedata.normalize("NFC", str(text or ""))
    out = []
    for m in list(_DATE.finditer(t)) + list(_NUMUNIT.finditer(t)):
        f = norm(m.group(0))
        if f and f not in out and not any(f in o for o in out):
            out.append(f)
    return out[:limit]


def _sentences(text: str):
    t = re.sub(r"[#*_>`|]+", " ", str(text or ""))
    parts = re.split(r"(?<=[.!?。])\s+|\n+", t)
    out = []
    for p in parts:
        p = re.sub(r"^\s*(?:[-•·▪◦○□■※]|\d+[.)]|[가-하][.)])\s*", "", p).strip()
        if len(re.sub(r"\W", "", p)) >= 12 and not re.match(r"^(근거|출처|참고)\s*[:：]", p):
            out.append(p)
    return out


def _bigrams(s: str):
    s = norm(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def gen_checks(answer, contexts, key_facts=None, reference=None, completion_tokens=None, max_tokens=None,
               sentence_threshold: float = 0.5, doc_hit=None) -> dict:
    """생성 과정 자체 점검 ($0, 결정적). 검색이 가져온 근거(contexts)를 답변이 얼마나 충실히 썼는지를 봄.
    key_facts 가 없으면 reference(정답) 문장에서 숫자 사실을 뽑아 씀."""
    ans = str(answer or "")
    ctx = [str(c or "") for c in (contexts or [])]
    ctx_n, ans_n = norm(" ".join(ctx)), norm(ans)
    facts = list(key_facts or []) or extract_facts(reference or "")
    r = {"답변 길이": len(ans)}
    r["거절"] = bool(REFUSAL.search(ans))
    r["문서 외 표시"] = bool(OUT_OF_DOC.search(ans))
    r["잘림"] = bool(completion_tokens and max_tokens and completion_tokens >= 0.98 * max_tokens)
    # ① 숫자 근거율 — 답변 속 숫자 사실이 근거에 있는 비율 (환각 의심 지표)
    nums = extract_facts(ans, limit=30)
    r["숫자 사실 수"] = len(nums)
    r["숫자 근거율"] = (sum(n in ctx_n for n in nums) / len(nums)) if nums else None
    # ② 문장 근거율 — 답변 문장마다 가장 비슷한 근거 문단과의 글자 2-gram 포함률
    sents = _sentences(ans)
    cgrams = [_bigrams(c) for c in ctx if c.strip()]
    sup = []
    for s in sents:
        sg = _bigrams(s)
        sup.append(max((len(sg & cg) / len(sg) for cg in cgrams), default=0.0) if sg else 0.0)
    r["문장 수"] = len(sents)
    r["문장 근거율"] = (sum(x >= sentence_threshold for x in sup) / len(sup)) if sup else None
    r["근거 겹침(평균)"] = (sum(sup) / len(sup)) if sup else None
    # ③ 핵심 사실 — 근거에 있었나 / 답변에 썼나 / 근거에 있던 것을 답변이 활용했나
    if facts:
        in_ctx = [has_alt(ctx_n, f) for f in facts]
        in_ans = [has_alt(ans_n, f) for f in facts]
        r["핵심 사실"] = " · ".join(facts)
        r["핵심사실(근거)"] = sum(in_ctx) / len(facts)
        r["핵심사실(답변)"] = sum(in_ans) / len(facts)
        r["근거 활용률"] = (sum(a and c for a, c in zip(in_ans, in_ctx)) / sum(in_ctx)) if any(in_ctx) else None
        r["근거 밖 사실률"] = (sum(a and not c for a, c in zip(in_ans, in_ctx)) / sum(in_ans)) if any(in_ans) else None
    else:
        r.update({"핵심 사실": "", "핵심사실(근거)": None, "핵심사실(답변)": None, "근거 활용률": None, "근거 밖 사실률": None})
    r["정답 문서 근거"] = doc_hit
    r["진단"] = gen_diagnosis(r, doc_hit)
    return r


def gen_diagnosis(r: dict, doc_hit=None) -> str:
    """생성 단계 한 줄 진단 (검색이 준 근거 대비 답변이 어땠는지)"""
    kc, ka, used = r.get("핵심사실(근거)"), r.get("핵심사실(답변)"), r.get("근거 활용률")
    evidence = (kc is not None and kc > 0) or bool(doc_hit)
    if not r.get("답변 길이"):
        return "❌ 빈 답변"
    if r.get("거절"):
        return "🙅 근거 있는데 거절" if evidence else "🙅 거절(근거 없음)"
    if r.get("문서 외 표시"):
        return "📚 문서 외 일반지식으로 답변"
    if kc is not None:
        if kc == 0 and ka:
            return "📚 근거 밖 정답(일반지식)"
        if kc > 0 and used is not None and used < 1:
            return "🧠 근거 누락(근거엔 있는데 답변에 없음)"
        if kc > 0:
            return "✅ 근거 반영"
        return "🔎 근거·답변 모두 핵심 사실 없음"
    if r.get("숫자 근거율") is not None and r["숫자 근거율"] < 0.5:
        return "⚠️ 숫자 근거 부족(환각 의심)"
    if r.get("문장 근거율") is not None and r["문장 근거율"] < 0.3:
        return "⚠️ 근거와 무관한 답변 의심"
    return "✅ 근거 기반" + (" · ⚠️ 잘림" if r.get("잘림") else "")


GEN_NUM_COLS = ["숫자 근거율", "문장 근거율", "근거 겹침(평균)", "핵심사실(근거)", "핵심사실(답변)", "근거 활용률", "근거 밖 사실률"]
GEN_FLAG_COLS = ["거절", "문서 외 표시", "잘림"]


def gen_summary(df: pd.DataFrame) -> pd.DataFrame:
    """생성 점검 요약 (지표 평균 · 비율 · 진단 분포)"""
    recs = []
    for c in GEN_NUM_COLS:
        if c in df:
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            recs.append({"지표": c, "값": v.mean() if len(v) else None, "계산된 질문": f"{len(v)}/{len(df)}", "종류": "평균(0~1)"})
    for c in GEN_FLAG_COLS:
        if c in df:
            v = df[c].fillna(False).astype(bool)
            recs.append({"지표": f"{c} 비율", "값": float(v.mean()) if len(v) else None, "계산된 질문": f"{int(v.sum())}/{len(df)}", "종류": "비율"})
    if "진단" in df:
        for k, n in Counter(df["진단"]).most_common():
            recs.append({"지표": f"진단: {k}", "값": n / len(df), "계산된 질문": f"{n}/{len(df)}", "종류": "진단 분포"})
    return pd.DataFrame(recs, columns=["지표", "값", "계산된 질문", "종류"])


# ═══════════════════════════════════════════════════════════════════════ ③ End-to-End (04)
def pick(row: dict, *names):
    """여러 이름 후보 중 처음으로 값이 있는 것 (노트북마다 열 이름이 조금씩 달라서)"""
    for n in names:
        for k, v in row.items():
            if k == n or (n.endswith("*") and k.startswith(n[:-1])):
                if is_num(v):
                    return float(v)
    return None


def retrieval_ok(row: dict, th: float = 0.5):
    """검색 성공 여부 — 정답 문서 적중(01) > 핵심사실(근거) > Key Fact(근거) > Context Recall > Context Precision 순"""
    for v in (row.get("정답 문서 근거"), row.get("01 Hit@k")):
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if is_num(v):
            return float(v) >= th
    v = pick(row, "핵심사실(근거)", "Key Fact (근거)", "Key Fact(근거)", "Context Recall", "Context Precision")
    return None if v is None else v >= th


def answer_ok(row: dict, th: float = 0.5, sim_th: float = 0.8):
    """정답 여부 — 틀린 전제를 못 바로잡으면 오답. 그 외 Answer Correctness > 핵심사실(답변) > Key Fact(답변) > Semantic Sim"""
    if row.get("틀린 전제") is True and row.get("전제 교정") is False:
        return False
    v = pick(row, "Answer Correctness", "핵심사실(답변)", "Key Fact (답변)", "Key Fact(답변)")
    if v is not None:
        return v >= th
    v = pick(row, "Semantic Sim", "Semantic Similarity")
    return None if v is None else v >= sim_th


def grounded_ok(row: dict, th: float = 0.5):
    """근거 충실 여부 — Faithfulness > 숫자 근거율 > 문장 근거율"""
    v = pick(row, "Faithfulness", "숫자 근거율", "문장 근거율")
    return None if v is None else v >= th


def attribute(row: dict, r_th=0.5, a_th=0.5, f_th=0.5) -> str:
    """질문 1개의 E2E 결과를 검색/생성 중 어디서 틀렸는지로 분류"""
    expect_ood = row.get("type") == "ood"
    refused = bool(row.get("거절"))
    if expect_ood:
        return "✅ 올바른 거절" if refused else "⚠️ 문서 밖 질문에 답변"
    r, a, g = retrieval_ok(row, r_th), answer_ok(row, a_th), grounded_ok(row, f_th)
    if a is None:                                         # 정답이 없는 질문: 검색 × 근거 충실도로만
        if r is None and g is None:
            return "❔ 판단 불가(정답·근거 지표 없음)"
        if r is False:
            return "🔎 검색 실패"
        if refused:
            return "🙅 근거 있는데 거절" if r else "🔎 검색 실패"
        if g is False:
            return "🧠 생성 실패(근거와 다른 답)"
        return "✅ 성공(근거 충실)" if r else "⚠️ 근거 충실하나 검색 미확인"
    if a and (r or r is None):
        return "✅ 성공" if g is not False else "✅ 정답(근거 충실도 낮음)"
    if a and r is False:
        return "📚 근거 없이 정답(일반지식)"
    if not a and r is False:
        return "🔎 검색 실패"
    if not a and refused:
        return "🙅 근거 있는데 거절"
    return "🧠 생성 실패(근거는 있음)"


STAGE_METRICS = {
    "① 검색": ["01 Hit@k", "01 Recall@k", "01 MRR", "핵심사실(근거)", "Key Fact (근거)", "Context Precision", "Context Recall"],
    "② 생성": ["Faithfulness", "Answer Relevancy", "숫자 근거율", "문장 근거율", "근거 활용률", "근거 밖 사실률", "전제 교정",
              "거절", "문서 외 표시", "잘림"],
    "③ E2E": ["Answer Correctness", "핵심사실(답변)", "Key Fact (답변)", "Semantic Sim", "E2E 성공"],
    "④ 운영": ["전체 응답(s)", "합계 $"],
}


def stage_summary(df: pd.DataFrame, by: str = "출처") -> pd.DataFrame:
    """단계 × 지표 × 출처(02 일괄 · 03 챗봇) 평균표"""
    groups = list(df[by].dropna().unique()) if by in df else []
    rows = []
    for stage, metrics in STAGE_METRICS.items():
        for m in metrics:
            if m not in df:
                continue
            rec = {"단계": stage, "지표": m}
            for gname in groups + ["전체"]:
                d = df if gname == "전체" else df[df[by] == gname]
                v = d[m]
                if v.dtype == bool or set(v.dropna().unique()) <= {True, False}:
                    v = v.dropna().astype(float)
                else:
                    v = pd.to_numeric(v, errors="coerce").dropna()
                rec[gname] = v.mean() if len(v) else None
                rec[f"{gname} n"] = int(len(v))
            if any(rec.get(g) is not None for g in groups + ["전체"]):
                rows.append(rec)
    return pd.DataFrame(rows)
