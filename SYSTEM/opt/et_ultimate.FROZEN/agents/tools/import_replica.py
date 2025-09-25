import requests, os
url = "http://IP_DA_REPLICA:9898/ultimos"
r = requests.get(url)
if r.status_code == 200:
    with open("/opt/et_ultimate/workspace/replica_feed.log", "a") as f:
        f.write(r.text + "\n")
