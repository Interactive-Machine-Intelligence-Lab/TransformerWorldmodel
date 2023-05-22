import json
from dotmap import DotMap

def read_json(src):
    with open(src, 'r') as f:
        cfg = json.load(f)
        cfg = DotMap(cfg)
    return cfg


train_cfg = read_json('config/train.json')
col_cfg = read_json('config/collector.json')
env_cfg = read_json('config/environment.json')
tok_cfg = read_json('config/tokenizer.json')
worldmodel_cfg = read_json('config/worldmodel.json')
ac_cfg = read_json('config/actorcritic.json')
replay_cfg = read_json('config/replay.json')