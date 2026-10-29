"""Run explicit-download pretrained perception checks; write a JSON result."""
import hashlib
import json
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.nn.text_model import build_text_model
from smol_vla_mod.vision.yolo_world_extractor import YoloWorldFeatureExtractor
from smol_vla_mod.language.clip_instruction_encoder import ClipInstructionEncoder

torch.manual_seed(42)
# Use a reproducible FP32 reference before testing deployment precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(2)
device = 'cuda'
assert torch.cuda.is_available()
report = {'status': 'running', 'passed': []}
output = Path('reports/pretrained_validation.json')
output.parent.mkdir(exist_ok=True)
def passed(name):
    report['passed'].append(name)
    print('PASS:', name, flush=True)
    output.write_text(json.dumps(report, indent=2))
try:
    # Ultralytics resolves this official asset; retain a checksum for provenance.
    loaded = YOLO('yolov8s-worldv2.pt')
    world = loaded.model.float().to(device).eval()
    weight_path = Path(loaded.ckpt_path)
    report['checkpoint_sha256'] = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    extractor = YoloWorldFeatureExtractor(world).eval()
    for i in range(22):
        for key, value in world.model[i].state_dict().items():
            torch.testing.assert_close(extractor.layers[i].state_dict()[key], value, rtol=0, atol=0)
    passed('pretrained load and retained weights')
    helper = build_text_model('clip:ViT-B/32', device=device)
    helper.model.float()
    assert all(p.dtype == torch.float32 for p in helper.model.parameters() if p.is_floating_point())
    print('CLIP validation dtype: float32; TF32 disabled', flush=True)
    encoder = ClipInstructionEncoder(helper)
    instructions = ['Pick up the red cup.', 'Place the blue bowl on the plate.']
    embedding = encoder(instructions)
    assert not torch.is_inference(embedding) and not embedding.requires_grad
    torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(2,device=device), atol=1e-5, rtol=1e-5)
    ref = helper.encode_text(helper.tokenize(instructions, truncate=False))
    torch.testing.assert_close(embedding, ref, atol=1e-6, rtol=1e-5)
    for i, text in enumerate(instructions):
        single = encoder([text])[0]
        print(f'Instruction {i} batch/single max absolute difference: {(single - embedding[i]).abs().max().item():.8g}', flush=True)
        torch.testing.assert_close(single, embedding[i], atol=1e-4, rtol=1e-3)
    passed('CLIP helper parity, norms, ordinary tensors and batching')
    try:
        encoder(['word ' * 200])
    except RuntimeError:
        pass
    else:
        raise AssertionError('Overlong instruction was silently accepted')
    passed('overlong instruction rejected')
    encoder.train()
    assert not encoder.training and not helper.training
    assert all(not p.requires_grad for p in encoder.parameters())
    projection = torch.nn.Linear(512, 32).to(device)
    projection(embedding).square().mean().backward()
    assert projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
    passed('frozen encoder supports trainable downstream projection')
    images = torch.rand(1,3,256,256,device=device).repeat(2,1,1,1)
    guides = embedding[:,None,:]
    # Reference detector must match one guide per image.
    world.model[-1].nc = 1
    taps = (4,6,9,15,18,21)
    captured, handles = {}, []
    def hook(i):
        def capture(module, args, result): captured[i] = result.detach().clone()
        return capture
    try:
        for i in taps: handles.append(world.model[i].register_forward_hook(hook(i)))
        with torch.no_grad(): world.predict(images, txt_feats=guides)
    finally:
        for h in handles: h.remove()
    with torch.no_grad():
        features = extractor(images,guides,return_backbone=True)
        swapped = extractor(images,guides.flip(0))
        for maps, indices in ((features.backbone,taps[:3]), (features.neck,taps[3:])):
            for value,i in zip(maps,indices):
                assert torch.isfinite(value).all()
                torch.testing.assert_close(value,captured[i],atol=1e-5,rtol=1e-4)
        for a,b in zip(features.neck,swapped.neck):
            torch.testing.assert_close(a.flip(0),b,atol=1e-5,rtol=1e-4)
        assert any(not torch.equal(a[0],a[1]) for a in features.neck)
    passed('pretrained reference parity and per-image instruction alignment')
    # Two views per sample, flattened sample-major.
    multi = images[:,None].expand(-1,2,-1,-1,-1).reshape(4,3,256,256)
    with torch.no_grad():
        views = extractor(multi,guides.repeat_interleave(2,dim=0))
        for v,f in zip(views.neck,features.neck):
            torch.testing.assert_close(v,f.repeat_interleave(2,dim=0),atol=1e-5,rtol=1e-4)
    passed('two-camera guide alignment')
    extractor.train()
    extractor.zero_grad(set_to_none=True)
    result = extractor(images,guides)
    sum(x.square().mean() for x in result.neck).backward()
    for group in (extractor.layers[:10], extractor.layers[10:]):
        grads = [p.grad for m in group for p in m.parameters() if p.requires_grad]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g).item() for g in grads)
    assert all(p.grad is None for p in encoder.parameters())
    passed('pretrained backbone and neck backward with frozen CLIP')
    report['status'] = 'passed'
except Exception as exc:
    report['status'] = 'failed'
    report['error'] = repr(exc)
    raise
finally:
    output.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
