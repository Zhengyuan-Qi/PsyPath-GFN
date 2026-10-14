"""Isolated correction: trajectory-entropy score-function gradient only."""
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))

def corrected_loss(trajectories,rewards,*,entropy_weight):
    logp=torch.stack([x.log_forward for x in trajectories])
    reward=torch.stack([x.log_reward for x in rewards]).detach()
    advantage=reward-reward.mean()
    # This surrogate has gradient log P(tau) * grad log P(tau).
    # Its on-policy expectation is the negative trajectory-entropy gradient.
    entropy_surrogate=(logp.detach()*logp).mean()
    return -(advantage*logp).mean()+float(entropy_weight)*entropy_surrogate

def test_entropy_gradient():
    logits=torch.tensor([-.7,.2,1.1],dtype=torch.float64,requires_grad=True)
    logp=logits.log_softmax(0);p=logp.exp()
    exact=torch.autograd.grad((p*logp).sum(),logits,retain_graph=True)[0]
    surrogate=torch.autograd.grad((p.detach()*logp.detach()*logp).sum(),logits,retain_graph=True)[0]
    legacy=torch.autograd.grad((p.detach()*logp).sum(),logits)[0]
    assert torch.allclose(exact,surrogate,atol=1e-12)
    assert torch.allclose(legacy,torch.zeros_like(legacy),atol=1e-12)
    assert exact.abs().max()>.01
    print('ENTROPY_GRADIENT_TEST_PASS',exact.tolist(),flush=True)

if __name__=='__main__':
    test_entropy_gradient()
    if '--unit-test' not in sys.argv:
        import validate_alternating_psychometric_gfn_cached as runner
        runner.reinforce_loss=corrected_loss
        raise SystemExit(runner.main())
