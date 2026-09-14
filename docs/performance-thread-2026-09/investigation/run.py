import sys,subprocess,time,json,hashlib,pathlib
root=pathlib.Path('/tmp/picvec-regression-20260914'); repo=pathlib.Path('/home/hidek/git/picvec')
version,binary,case,limit=sys.argv[1:]; dest=root/f'{version}-{case}'; output=dest.with_suffix('.svg'); log=dest.with_suffix('.log')
source=repo/'sample/input'/({'booster':'booster-layout.jpg','sheet':'cliparts-6x6.png'}[case])
cmd=[binary,str(source),str(output),'--threads','4','--verbose']
if case=='sheet':cmd+=['--remove-chroma-key-background']
start=time.monotonic(); status='complete'
with log.open('w') as f:
 try:
  result=subprocess.run(cmd,stdout=f,stderr=f,timeout=float(limit));code=result.returncode
  if code:status='failed'
 except subprocess.TimeoutExpired: status='timeout';code=None
record=dict(version=version,case=case,command=cmd,status=status,exit_code=code,seconds=time.monotonic()-start,binary_sha256=hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest())
if status=='complete':
 record['svg_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
 txt=log.read_text();record['summary']=json.JSONDecoder().raw_decode(txt[txt.index('{\n'):])[0]
dest.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['summary','command'] }),flush=True)
