import io,json
from pathlib import Path
import torch
from torch import nn
from ultralytics.nn.tasks import WorldModel
from smol_vla_mod.vision.yolo_world_extractor import YoloWorldFeatureExtractor
from smol_vla_mod.fusion.observation_fusion import ObservationFusion
from smol_vla_mod.policy.flow_policy import FlowActionExpert,RobotPolicy

torch.manual_seed(42);torch.set_num_threads(2)
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
d='cuda';report={'status':'running','passed':[]}
path=Path('reports/flow_policy_validation.json');path.parent.mkdir(exist_ok=True)
def passed(name):
    report['passed'].append(name);print('PASS:',name,flush=True)
    path.write_text(json.dumps(report,indent=2))
try:
    expert=FlowActionExpert().to(d)
    context=torch.randn(2,98,256,device=d)
    actions=torch.randn(2,16,7,device=d)
    noise=torch.randn_like(actions);t=torch.tensor([.25,.75],device=d)
    mask=torch.ones(2,16,device=d,dtype=torch.bool);mask[1,5:]=False
    loss=expert.loss(context,actions,mask,noise,t)
    assert loss.ndim==0 and torch.isfinite(loss)
    altered=actions.clone();altered[~mask]=float('nan')
    changed_noise=noise.clone();changed_noise[~mask]=1e6
    other=expert.loss(context,altered,mask,changed_noise,t)
    torch.testing.assert_close(loss,other,atol=0,rtol=0)
    differentiable=actions.clone().requires_grad_()
    expert.loss(context,differentiable,mask,noise,t).backward()
    assert (differentiable.grad[~mask]==0).all()
    passed('masked loss ignores padded targets and padded noise, including NaNs')
    try:expert.loss(context,actions,torch.zeros_like(mask))
    except ValueError:pass
    else:raise AssertionError('All-padding mask accepted')
    passed('all-padding batches rejected')
    # Analytic constant field checks the sampler direction and integration interval.
    class Oracle(FlowActionExpert):
        def __init__(self):nn.Module.__init__(self);self.horizon=16;self.action_dim=7
        def forward(self,x,t,context,valid=None):return actions-noise
    oracle=Oracle()
    for steps in (1,4,10):
        torch.testing.assert_close(oracle.sample(context,steps,noise),actions,atol=2e-6,rtol=2e-6)
    passed('Euler sampler integrates noise-to-action direction over [0,1]')
    expert.eval()
    a=expert.sample(context,4,noise);b=expert.sample(context,4,noise)
    assert a.shape==(2,16,7) and torch.isfinite(a).all()
    torch.testing.assert_close(a,b)
    assert not torch.allclose(a,expert.sample(context.flip(0),4,noise))
    passed('finite reproducible sampling and context dependence')
    extractor=YoloWorldFeatureExtractor(WorldModel(cfg='yolov8s-worldv2.yaml',nc=1,verbose=False)).to(d)
    policy=RobotPolicy(extractor,ObservationFusion().to(d),expert).train()
    images=torch.rand(2,2,3,256,256,device=d)
    instructions=torch.nn.functional.normalize(torch.randn(2,512,device=d),dim=-1)
    state=torch.randn(2,8,device=d)
    policy.zero_grad(set_to_none=True)
    value=policy(images,instructions,state,actions,mask,noise,t)
    value.backward()
    for name,group in [('backbone',policy.extractor.layers[:10]),('neck',policy.extractor.layers[10:]),('fusion',policy.fusion),('expert',policy.expert)]:
        grads=[p.grad for p in group.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads),name
        assert any(torch.count_nonzero(g).item() for g in grads),name
    passed('complete policy masked loss and gradients through all four trainable groups')
    policy.eval()
    before=policy.sample(images,instructions,state,steps=4,noise=noise)
    buffer=io.BytesIO();torch.save(policy.state_dict(),buffer);buffer.seek(0)
    # Reload into the same graph after perturbing one parameter to test restoration.
    with torch.no_grad():next(policy.parameters()).add_(.1)
    policy.load_state_dict(torch.load(buffer,map_location=d,weights_only=True))
    after=policy.sample(images,instructions,state,steps=4,noise=noise)
    torch.testing.assert_close(before,after)
    assert torch.isfinite(after).all()
    passed('complete policy sampling and checkpoint restoration')
    report.update(status='passed',horizon=16,action_dim=7,sampling_steps_tested=[1,4,10],
        expert_parameters=sum(p.numel() for p in expert.parameters()),
        limitation='Random-weight visual integration with synthetic inputs; demonstration learning and pretrained integrated run pending')
except Exception as exc:report.update(status='failed',error=repr(exc));raise
finally:path.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
