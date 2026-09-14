from pathlib import Path
import json,subprocess,sys
root=Path('/home/hidek/git/picvec')
base=Path('/tmp/picvec-generic-speed')
names=['booster-layout.jpg','cliparts-6x6.png','viewport2.jpg','car.png','vectorization-stress-still-life.png','cliparts.png','boy_and_turtle.png','viewport1.jpg','wikipedia_logo_1_0.png']
rows=[]
for i,name in enumerate(names):
    row={'input':name}
    for version in (['before','after'] if i%2==0 else ['after','before']):
        dest=base/'comparison'/f'{i:02}-{version}'
        command=[sys.executable,str(root/'scripts/benchmark_committed_svgs.py'),'--output-dir',str(dest),'--binary',str(base/version),'--inputs',name]
        if version=='after':command.append('--update-output')
        print(f'PAIR {name} {version}',flush=True)
        result=subprocess.run(command,cwd=root)
        path=dest/'results.json'
        if not path.exists():raise SystemExit(result.returncode or 1)
        row[version]=json.loads(path.read_text())['runs'][0]
        if version=='after' and result.returncode:
            rows.append(row)
            (base/'comparison/results.json').write_text(json.dumps(rows,indent=2)+'\n')
            raise SystemExit(result.returncode)
    rows.append(row)
    (base/'comparison/results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('COMPARISON',name,row['before']['seconds'],row['after']['seconds'],flush=True)
