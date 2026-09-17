import ast
import copy
import subprocess
import torch
import pytest
from cmgm.models.formal_baselines_v2 import TCN,CausalConv,TCNBlock,input_view
from third_party.baselines.tcn.tcn import TemporalConvNet,TemporalBlock,Chomp1d


def test_faithful_structure_full_input_and_initialization():
    torch.manual_seed(42);m=TCN(284).eval()
    assert not hasattr(m,'projection') and isinstance(m.tcn,TemporalConvNet)
    assert not any(isinstance(v,(CausalConv,TCNBlock)) for v in m.modules())
    for i,b in enumerate(m.tcn.network):
        assert isinstance(b,TemporalBlock)
        assert [type(v).__name__ for v in b.net]==['ParametrizedConv1d','Chomp1d','ReLU','Dropout','ParametrizedConv1d','Chomp1d','ReLU','Dropout']
        assert b.conv1.in_channels==(5964 if i==0 else 128)
        for c in (b.conv1,b.conv2):
            assert c.kernel_size==(3,) and c.padding==(2*2**i,) and c.dilation==(2**i,)
            v=c.parametrizations.weight.original1;g=c.parametrizations.weight.original0
            torch.testing.assert_close(c.weight,g*v/v.norm(dim=(1,2),keepdim=True))
            assert .009<c.weight.std()<.011
        assert (b.downsample is not None)==(i==0)
        if b.downsample is not None:assert .009<b.downsample.weight.std()<.011
    x=torch.randn(2,20,284,21)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    with torch.no_grad():
        hidden=m.temporal_states(x);assert hidden.shape==(2,20,128)
        assert m(x).shape==(2,4,24)
        altered=x.clone();altered[:,10:]+=100
        torch.testing.assert_close(hidden[:,:10],m.temporal_states(altered)[:,:10],atol=0,rtol=0)
        for i,b in enumerate(m.tcn.network):
            assert .009<b.conv1.weight.std()<.011 and .009<b.conv2.weight.std()<.011
    assert all(torch.equal(p,before[n]) for n,p in m.named_parameters())


def test_chomp_and_checkpoint_roundtrip():
    c=Chomp1d(4);x=torch.arange(20).reshape(1,1,20);assert torch.equal(c(x),x[:,:,:16])
    torch.manual_seed(42);a=TCN(30).eval();b=copy.deepcopy(a)
    b.load_state_dict(a.state_dict());x=torch.randn(2,20,30,21)
    with torch.no_grad():torch.testing.assert_close(a(x),b(x),atol=0,rtol=0)
    loss=a(x).square().mean();g=torch.autograd.grad(loss,tuple(a.parameters()))
    assert all(torch.isfinite(t).all() for t in g)


def test_only_tcn_class_changed_in_model_file():
    from cmgm.scripts.formal_v2_audit import ROOT
    path='cmgm/models/formal_baselines_v2.py'
    old=ast.parse(subprocess.check_output(['git','show','HEAD:'+path],cwd=ROOT,text=True))
    new=ast.parse((ROOT/path).read_text())
    def rest(tree):return [ast.dump(n,include_attributes=False) for n in tree.body if not(isinstance(n,ast.ClassDef) and n.name=='TCN')]
    assert rest(old)==rest(new)


def test_official_provenance():
    from cmgm.scripts.formal_tcn_repair import verify_provenance
    assert verify_provenance()['commit']=='2f8c2b817050206397458dfd1f5a25ce8a32fe65'
