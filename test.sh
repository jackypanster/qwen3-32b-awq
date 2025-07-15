# python3 -c "print(' '.join(['token'] * 32000))" > prompt32k.txt

python3 -c "
import json, sys, subprocess, concurrent.futures, time
payload = {
    'model': 'coder',
    'messages': [{'role':'user','content':open('prompt32k.txt').read()}],
    'max_tokens': 32,
    'temperature': 0
}
with open('payload.json','w') as f: json.dump(payload, f)

def hit():
    return subprocess.run([
        'curl','-s','http://localhost:8888/v1/chat/completions',
        '-H','Content-Type: application/json',
        '-d','@payload.json'
    ], capture_output=True, text=True).stdout

start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    for f in concurrent.futures.as_completed([ex.submit(hit) for _ in range(10)]):
        print('🏁', time.time()-start, 's')
"