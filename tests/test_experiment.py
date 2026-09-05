import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from looped_transformer_comparison.model import ModelConfig, LanguageModel
from looped_transformer_comparison.data import prepare, load_data, batch
from looped_transformer_comparison.engine import evaluate, train, load_checkpoint, comparison

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def dataset(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    for s in ('train', 'validation', 'test'):
        (raw / f'{s}.txt').write_text(('small cats learn words and small dogs read books.\n' * 8) if s == 'train' else ('unseen quantum zebras read books.\n' * 3))
    out = tmp_path / 'data'
    prepare(out, 280, raw)
    return out

@pytest.fixture
def config(tmp_path):
    c = json.loads((ROOT / 'configs/smoke.json').read_text())
    c['model'].update(seq_len=8, width=16, heads=2, dropout=0.2)
    p = tmp_path / 'config.json'
    p.write_text(json.dumps(c))
    return p

@pytest.mark.parametrize('updates', [{'loops': 3}, {'width': 7}, {'depth': 0}, {'dropout': 1}])
def test_invalid_configs(updates):
    c = dict(vocab_size=20, seq_len=8, width=16, heads=2, depth=4, loop_layers=2, loops=2)
    c.update(updates)
    with pytest.raises(ValueError):
        ModelConfig(**c)


def models(loops=2):
    c = ModelConfig(vocab_size=20, seq_len=8, width=16, heads=2, depth=4, loop_layers=4//loops, loops=loops)
    torch.manual_seed(9)
    a = LanguageModel(c, 'standard')
    torch.manual_seed(9)
    b = LanguageModel(c, 'looped')
    return a, b


def test_depth_shared_initialization_and_loop_one():
    a,b = models()
    assert len(a.blocks)==4 and len(b.blocks)==2
    assert sum(p.numel() for p in b.parameters()) < sum(p.numel() for p in a.parameters())
    for key,value in b.state_dict().items():
        assert torch.equal(value, a.state_dict()[key]), key
    calls=[]
    handles=[block.register_forward_hook(lambda *args: calls.append(1)) for block in b.blocks]
    b(torch.tensor([[1,2,3]]))
    assert len(calls)==4
    for handle in handles: handle.remove()
    a,b = models(1)
    assert torch.equal(a(torch.tensor([[1,2,3]])), b(torch.tensor([[1,2,3]])))

@pytest.mark.parametrize('arch', ['standard','looped'])
def test_causal_mask(arch):
    a,b = models()
    m = a if arch=='standard' else b
    x=torch.tensor([[1,2,3,4,5]])
    y=x.clone(); y[:,3:]=9
    torch.testing.assert_close(m(x)[:,:3],m(y)[:,:3], rtol=0, atol=0)


def test_shared_gradient_matches_sum_of_unrolled_independent_blocks():
    unrolled, shared=models()
    for i, block in enumerate(unrolled.blocks):
        block.load_state_dict(shared.blocks[i%2].state_dict())
    x=torch.tensor([[1,2,3,4]])
    target=torch.tensor([[2,3,4,5]])
    for model in (unrolled,shared):
        F.cross_entropy(model(x).reshape(-1,20),target.flatten()).backward()
    for i, block in enumerate(shared.blocks):
        grads=[dict(unrolled.blocks[j].named_parameters()) for j in (i,i+2)]
        for name,p in block.named_parameters():
            torch.testing.assert_close(p.grad, grads[0][name].grad+grads[1][name].grad, rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(shared.token.weight.grad,unrolled.token.weight.grad,rtol=1e-5,atol=1e-7)


def test_train_only_tokenizer_checksums_and_occupied(dataset,tmp_path):
    before=json.loads((dataset/'tokenizer.json').read_text())
    raw=tmp_path/'raw'
    (raw/'validation.txt').write_text('totally distinct foreign validation content '*30)
    other=tmp_path/'other'
    prepare(other,280,raw)
    assert json.loads((other/'tokenizer.json').read_text()) == before
    streams,manifest=load_data(dataset,8)
    assert all(len(streams[s])==manifest['splits'][s]['tokens'] for s in streams)
    with pytest.raises(ValueError,match='not empty'): prepare(dataset,280,raw)
    with (dataset/'test.bin').open('ab') as f: f.write(b'1234')
    with pytest.raises(ValueError,match='checksum'): load_data(dataset,8)


def test_batch_deterministic_and_shifted():
    stream=np.arange(100,dtype=np.uint32)
    a=torch.Generator().manual_seed(42); b=torch.Generator().manual_seed(42)
    for _ in range(3):
        x,y=batch(stream,3,8,a,'cpu'); u,v=batch(stream,3,8,b,'cpu')
        assert torch.equal(x,u) and torch.equal(y,v) and torch.equal(x+1,y)


def test_eval_token_weighted_tail_and_training_mode():
    class Toy(torch.nn.Module):
        config=type('Config',(),{'seq_len':4})()
        def forward(self,x):
            return F.one_hot(x%3,3).float()*2
    model=Toy()
    stream=np.array([0,1,1,1,2,1,1,0,2,2,0,1],dtype=np.uint32)
    expected=F.cross_entropy(model(torch.tensor(stream[:-1].astype('int64'))),torch.tensor(stream[1:].astype('int64'))).item()
    for bs in (1,2,8):
        result=evaluate(model,stream,bs,torch.device('cpu'),'fp32')
        assert result['tokens']==len(stream)-1
        assert result['loss']==pytest.approx(expected,abs=1e-6)
        assert model.training
    model.eval(); evaluate(model,stream,2,torch.device('cpu'),'fp32'); assert not model.training

@pytest.mark.parametrize('arch',['standard','looped'])
def test_exact_dropout_resume(dataset,config,tmp_path,arch):
    full=tmp_path/'full'; paused=tmp_path/'paused'
    result=train(config,dataset,full,arch)
    assert result['status']=='complete' and math.isfinite(result['test']['loss'])
    assert train(config,dataset,paused,arch,stop_after=1)['status']=='paused'
    train(config,dataset,paused,arch,resume=True)
    a=load_checkpoint(full/'last.pt'); b=load_checkpoint(paused/'last.pt')
    assert a['step']==b['step']==4
    for key in a['model']: assert torch.equal(a['model'][key],b['model'][key]),key
    assert torch.equal(a['batch_rng'],b['batch_rng'])
    assert torch.equal(a['torch_rng'],b['torch_rng'])
    with pytest.raises(ValueError,match='not empty'): train(config,dataset,full,arch)
    changed=json.loads(config.read_text()); changed['training']['lr']*=2; config.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='mismatch'): train(config,dataset,paused,arch,resume=True)


def cli(*args,success=True):
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false'}
    result=subprocess.run([sys.executable,'-m','looped_transformer_comparison.cli',*map(str,args)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
    assert (result.returncode==0)==success,result.stdout+result.stderr
    return result


def test_paired_cli_evaluate_generate_report(dataset,config,tmp_path):
    out=tmp_path/'pair'
    cli('compare','--config',config,'--data',dataset,'--output',out)
    report=comparison(out)
    assert report['runs'][0]['train_tokens']==report['runs'][1]['train_tokens']==128
    assert (out/'comparison.md').exists()
    cli('report','--output',out)
    result=cli('evaluate','--checkpoint',out/'looped/best.pt','--data',dataset,'--device','cpu','--batch-size','3')
    assert json.loads(result.stdout)['tokens']>0
    args=('generate','--checkpoint',out/'looped/best.pt','--tokenizer',dataset/'tokenizer.json','--prompt','small cats','--max-new-tokens','3','--temperature','0','--device','cpu')
    assert cli(*args).stdout==cli(*args).stdout
    cli('compare','--config',config,'--data',dataset,'--output',out,success=False)
    cli('compare','--config',config,'--data',dataset,'--output',out,'--resume')
    r=json.loads((out/'looped/result.json').read_text()); r['train_tokens']+=1
    (out/'looped/result.json').write_text(json.dumps(r))
    with pytest.raises(ValueError,match='token budgets'): comparison(out)


def test_shell_syntax_and_lock():
    for path in [*ROOT.glob('scripts/*.sh'),*ROOT.glob('scripts/*.sbatch')]:
        subprocess.run(['bash','-n',str(path)],check=True)
    subprocess.run(['uv','lock','--check'],cwd=ROOT,check=True)
    cli('check')
    if not torch.cuda.is_available(): cli('check','--require-cuda',success=False)

@pytest.mark.parametrize('name',['auto','cuda','cuda:2'])
def test_cuda_device_selection_mocked(monkeypatch,name):
    from looped_transformer_comparison.engine import device_for
    seen=[]
    monkeypatch.setattr(torch.cuda,'is_available',lambda: True)
    monkeypatch.setattr(torch.cuda,'current_device',lambda: 1)
    monkeypatch.setattr(torch.cuda,'set_device',lambda d: seen.append(d))
    expected=torch.device('cuda:2' if name=='cuda:2' else 'cuda:1')
    assert device_for(name)==expected
    assert seen==[expected]
