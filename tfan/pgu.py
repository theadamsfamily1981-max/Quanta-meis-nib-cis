import re, hashlib, json, time
from z3 import Solver, parse_smt2_string, is_const, Z3_OP_TRUE, Z3_OP_FALSE

def canonical_key(ast):
    syms = []
    def walk(a):
        try: kids = a.children()
        except Exception: kids = []
        if is_const(a) and a.decl().kind() not in (Z3_OP_TRUE, Z3_OP_FALSE):
            s = str(a.decl().name())
            if s not in syms: syms.append(s)
        for k in kids: walk(k)
    walk(ast)
    rename = {old: f"x{idx+1}" for idx, old in enumerate(syms)}
    sexpr = ast.sexpr()
    for old, new in rename.items():
        sexpr = re.sub(rf"\b{re.escape(old)}\b", new, sexpr)
    return hashlib.sha1(sexpr.encode()).hexdigest()

class PGUCache:
    def __init__(self, timeout_ms=120):
        self.timeout = timeout_ms
        self.cache = {}

    def check(self, smt2: str):
        ast = parse_smt2_string(smt2)
        key = canonical_key(ast)
        if key in self.cache:
            return {"cached": True, "ms": 0.0, "res": self.cache[key]}
        s = Solver(); s.set("timeout", self.timeout)
        t0 = time.perf_counter()
        res = s.check(ast)
        ms = (time.perf_counter()-t0)*1000.0
        self.cache[key] = str(res)
        return {"cached": False, "ms": ms, "res": str(res)}
