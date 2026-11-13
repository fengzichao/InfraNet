"""Verify the paired InfraNet-IR C training loss and gradient propagation."""
from pathlib import Path
import contextlib,io,sys
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from models.yolo import Model,DDetect,DualDDetect
from utils.loss_tal import ComputeLoss as SingleLoss
from utils.loss_tal_dual import ComputeLoss as DualLoss


def main():
    torch.set_num_threads(1)
    hyp=yaml.safe_load((ROOT/'data/hyps/hyp.scratch-high.yaml').read_text())
    hyp.update(w_main=1.0,w_aux=0.25)
    for file in ('yolov8-c.yaml',):
        conf=yaml.safe_load((ROOT/'models/detect'/file).read_text());conf.update(nc=1, width_multiple=0.25,depth_multiple=0.5)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            model=Model(conf,ch=3,nc=1)
        model.hyp=hyp
        model.train()
        loss_fn=DualLoss(model) if isinstance(model.model[-1],DualDDetect) else SingleLoss(model)
        x=torch.rand(1,6,64,64)
        targets=torch.tensor([[0,0,0.5,0.5,0.3,0.3]])
        loss,parts=loss_fn(model(x),targets)
        assert torch.isfinite(loss).all() and loss.requires_grad, (file,loss)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters()), file
        print('PASS',file,type(model.model[-1]).__name__,round(float(loss),4))

if __name__=='__main__':main()
