"""Replay a frozen frontend corpus without importing the Python compiler or LLVM."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def relocate(value, work):
    if isinstance(value,str): return value.replace("$FIXTURES",str(work)).replace("$REPO",str(ROOT))
    if isinstance(value,list): return [relocate(x,work) for x in value]
    if isinstance(value,dict): return {k:relocate(v,work) for k,v in value.items()}
    return value


def comparable(result):
    if result["ok"]: return dict(ok=True,resolution=result["resolution"],sections=result["sections"])
    error=result["diagnostics"][0]
    if error["phase"]=="source":
        categories=["Cyclic","nesting exceeds","outside","ambiguous","missing import","sectionless","duplicate @","invalid import","duplicate config","invalid config"]
        return dict(ok=False,phase="source",category=next((s for s in categories if s in error["message"]),"preprocessing"))
    return dict(ok=False,phase=error["phase"],message=error["message"],line=error.get("line",0),column=error.get("column",0))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("helper",type=Path)
    parser.add_argument("reference",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with zipfile.ZipFile(args.reference) as archive, tempfile.TemporaryDirectory(prefix="replay-",dir=args.output.parent.resolve()) as temp:
        work=Path(temp)
        manifest=json.loads(archive.read("manifest.json"))
        for collection in ["referenceFiles","dependencyHashes"]:
            for relative,digest in manifest.get(collection,{}).items():
                actual=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
                if actual!=digest: raise RuntimeError("Reference provenance changed: "+relative)
        for member in archive.namelist():
            if member.startswith("fixtures/"):
                file=(work/member[len("fixtures/"):]).resolve()
                if not file.is_relative_to(work): raise RuntimeError("Unsafe fixture path")
                file.parent.mkdir(parents=True,exist_ok=True)
                file.write_bytes(archive.read(member))
        for member in sorted(archive.namelist()):
            if not member.endswith(".json") or member=="manifest.json" or "/" in member: continue
            fixture=relocate(json.loads(archive.read(member)),work)
            request,result=work/"request.json",work/"result.json"
            request.write_text(json.dumps(fixture["request"],ensure_ascii=False),encoding="utf-8")
            started=time.perf_counter()
            process=subprocess.run([str(args.helper.resolve()),str(request),str(result)],env=dict(os.environ,PATH=""),capture_output=True,text=True,timeout=120)
            native=json.loads(result.read_text(encoding="utf-8"))
            passed=comparable(native)==fixture["expected"] and process.returncode==(0 if native["ok"] else 1)
            row=dict(name=fixture["name"],passed=passed,accepted=native["ok"],nativeProcessSeconds=time.perf_counter()-started,nativePhaseSeconds=native.get("phaseSeconds"))
            rows.append(row)
            print(json.dumps(row),flush=True)
    report=dict(passed=all(r["passed"] for r in rows),cases=rows,provenanceVerified=True,
                helperSha256=hashlib.sha256(args.helper.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__=="__main__": raise SystemExit(main())
