"""Run with:  python -m unittest discover -s tests -v"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spp.bank import EliteArchive, ReplayBank, niche_of  # noqa: E402
from spp.bf import run_programs  # noqa: E402
from spp.data import RandomPCFG, sample_pcfg_rows, sample_uniform_programs  # noqa: E402
from spp.generator import ProgramGenerator, generator_losses, log_prior  # noqa: E402
from spp.icl import TASKS, make_instance  # noqa: E402
from spp.model import build_model, per_sequence_cross_entropy  # noqa: E402
from spp.reward import adamw_preconditioner, directional_losses, make_tangent  # noqa: E402
from spp.structure import classify_output  # noqa: E402
from spp.tokens import ALPHABET_SIZE, BODY_ALPHABET, PROGRAM_ALPHABET  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class TestInterpreter(unittest.TestCase):
    def test_paper_examples(self):
        r = run_programs([b"+++[>+.<-]", b"+[.++]", b"+[.L>]"], T=8, step_budget=10_000, tape_size=64)
        self.assertEqual(r.outputs[0].tolist(), [1, 2, 3, 0, 0, 0, 0, 0])
        self.assertEqual(r.outputs[1].tolist(), [1, 3, 5, 7, 9, 11, 13, 15])
        self.assertEqual(r.outputs[2].tolist(), [1, 3, 9, 27, 81, 243, 217, 139])
        self.assertEqual(r.n_emitted[0], 3)

    def test_c_matches_python(self):
        rng = np.random.default_rng(0)
        progs = [bytes(rng.choice(list(BODY_ALPHABET.encode()), size=int(rng.integers(0, 40)))) for _ in range(300)]
        a = run_programs(progs, T=64, step_budget=2000, tape_size=32, seed=123)
        b = run_programs(progs, T=64, step_budget=2000, tape_size=32, seed=123, force_python=True)
        np.testing.assert_array_equal(a.outputs, b.outputs)
        np.testing.assert_array_equal(a.max_depth, b.max_depth)
        np.testing.assert_array_equal(a.steps, b.steps)
        np.testing.assert_array_equal(a.n_emitted, b.n_emitted)

    def test_macro_heavy_programs_do_not_overflow(self):
        # regression: X expands to 19 bytes; the old buffer assumed 17 per token (heap overflow -> glibc abort)
        progs = [b"X" * n for n in range(1, 65)] + [b"XXX", b"XXXX+", b"X" * 64 + b"[.]"] * 50
        r = run_programs(progs * 20, T=32, step_budget=5000, tape_size=16)
        rp = run_programs(progs, T=32, step_budget=5000, tape_size=16, force_python=True)
        np.testing.assert_array_equal(r.outputs[: len(progs)], rp.outputs)
        self.assertEqual(r.outputs[2].tolist()[:1], [0])      # XXX emits nothing
        self.assertEqual(run_programs([b"X."], T=2).outputs[0].tolist(), [16, 0])

    def test_random_tape_differs_by_seed(self):
        a = run_programs([b",.,.,."], T=3, seed=1).outputs
        b = run_programs([b",.,.,."], T=3, seed=2).outputs
        self.assertFalse(np.array_equal(a, b))

    def test_unmatched_brackets_and_budget(self):
        r = run_programs([b"]]]+.[[[", b"+[]"], T=4, step_budget=50)
        self.assertEqual(r.outputs[0].tolist(), [1, 0, 0, 0])
        self.assertEqual(r.steps[1], 50)


class TestReward(unittest.TestCase):
    def test_jvp_equals_gradient_inner_product(self):
        torch.manual_seed(0)
        model = build_model("tiny", max_seq_len=32).to(DEV)
        params = dict(model.named_parameters())
        tangent = {k: torch.randn_like(v) for k, v in params.items()}
        x = torch.randint(0, 256, (5, 32), device=DEV)
        losses, d = directional_losses(model, params, dict(model.named_buffers()), tangent, x, chunk_size=2)
        for i in range(5):
            model.zero_grad()
            li = per_sequence_cross_entropy(model(x[i:i + 1, :-1]), x[i:i + 1, 1:])[0]
            grads = torch.autograd.grad(li, list(params.values()))
            ip = sum(float((g * tangent[k]).sum()) for g, k in zip(grads, params))
            self.assertAlmostEqual(float(losses[i]), float(li), places=4)
            self.assertAlmostEqual(float(d[i]) / (abs(ip) + 1e-6), ip / (abs(ip) + 1e-6), places=2)

    def test_preconditioner_uses_adam_state(self):
        model = build_model("tiny", max_seq_len=16).to(DEV)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, eps=1e-8)
        P0 = adamw_preconditioner(opt, dict(model.named_parameters()))
        self.assertTrue(all(torch.allclose(v, torch.full_like(v, 1e-2 / 1e-8)) for v in P0.values()))
        x = torch.randint(0, 256, (2, 16), device=DEV)
        per_sequence_cross_entropy(model(x[:, :-1]), x[:, 1:]).mean().backward()
        opt.step()
        P1 = adamw_preconditioner(opt, dict(model.named_parameters()))
        self.assertTrue(all(torch.isfinite(v).all() and (v > 0).all() for v in P1.values()))
        t = make_tangent(P1, {k: v.detach().clone() for k, v in model.named_parameters()}, dict(model.named_parameters()))
        self.assertTrue(all(float(v.abs().max()) == 0 for v in t.values()))


class TestGenerator(unittest.TestCase):
    def test_sampled_logprob_matches_evaluated(self):
        torch.manual_seed(0)
        model = build_model("tiny", max_seq_len=18).to("cpu")
        gen = ProgramGenerator(model, max_len=16, device="cpu")
        bodies, lp = gen.sample(32)
        self.assertTrue(all(set(b) <= set(BODY_ALPHABET.encode()) for b in bodies))
        self.assertTrue(all(len(b) <= 16 for b in bodies))
        with torch.no_grad():
            lp2 = gen.log_probs(bodies)
        self.assertTrue(torch.allclose(lp, lp2, atol=1e-4), (lp, lp2))

    def test_prior_and_losses(self):
        lp0 = log_prior([b"", b"+++"])
        self.assertAlmostEqual(float(lp0[0]), -math.log(ALPHABET_SIZE))
        self.assertAlmostEqual(float(lp0[1]), -4 * math.log(ALPHABET_SIZE), places=5)
        logp = torch.tensor([-3.0, -4.0, -5.0, -6.0], requires_grad=True)
        r = torch.tensor([1.0, 0.0, 2.0, -1.0])
        is_mut = torch.tensor([False, True, False, False])
        loss, stats = generator_losses(logp, logp.detach(), lp0.new_tensor([-1, -1, -1, -1.0]), r, is_mut, beta=0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(logp.grad[1]), 0.0)   # mutated row has zero reward weight and no PG term


class TestBank(unittest.TestCase):
    def test_archive(self):
        a = EliteArchive(per_niche=2, decay=0.5, seed=0)
        self.assertFalse(a.add(b"+", 0.0, 0))
        for i in range(5):
            a.add(bytes([43] * (i + 1)), float(i + 1), 0)
        self.assertEqual(a.n_occupied, 1)
        self.assertEqual(a.n_programs, 2)
        self.assertEqual([e.reward for e in a.niches[niche_of(5, 0)]], [5.0, 4.0])
        a.step_decay()
        self.assertEqual(a.niches[niche_of(5, 0)][0].reward, 2.5)
        kids = a.propose(20, max_len=8)
        self.assertEqual(len(kids), 20)
        self.assertTrue(all(len(k) <= 8 and set(k) <= set(BODY_ALPHABET.encode()) for k in kids))
        self.assertEqual(niche_of(9, 20), (8, 1))
        self.assertEqual(niche_of(33, 2), (2, 3))

    def test_replay_bank(self):
        from spp.bank import Program
        b = ReplayBank(capacity=5, seed=0)
        b.add([Program(bytes([43 + i]), -1.0 * i, i) for i in range(8)])
        self.assertEqual(len(b), 5)
        s = b.sample(3)
        self.assertEqual(len(s), 3)
        self.assertTrue(all(p.source == "replay" for p in s))
        self.assertEqual(len({p.body for p in s}), 3)


class TestStructure(unittest.TestCase):
    def test_families(self):
        progs = [b"+[.++]", b"+[.L>]", b"+[[-.L>L>-]-]", b",.,.,.,.", b"+[.]"]
        r = run_programs(progs, T=200, step_budget=50_000, tape_size=64, seed=3)
        fams = [classify_output(r.outputs[i], int(r.n_emitted[i])) for i in range(len(progs))]
        self.assertEqual(fams[0], "arithmetic")
        self.assertEqual(fams[1], "geometric")
        self.assertEqual(fams[2], "cubic")
        self.assertIsNone(fams[3])
        self.assertIsNone(fams[4])   # constant output: period 1 is degenerate
        fib = [1, 1]
        for _ in range(100):
            fib.append((fib[-1] + fib[-2]) % 256)
        self.assertEqual(classify_output(bytes([7, 7, 7] + fib)), "fibonacci")
        quad = [(3 * n * n + n + 9) % 256 for n in range(120)]
        self.assertEqual(classify_output(bytes(quad)), "quadratic")


class TestDataAndICL(unittest.TestCase):
    def test_icl_instances(self):
        rng = np.random.default_rng(0)
        for task in TASKS:
            p, a = make_instance(task, 3, rng, k=4)
            self.assertEqual(p[0], 0)
            self.assertTrue(len(a) >= 1)
        p, a = make_instance("sum", 0, rng)
        self.assertEqual(a[0], (p[1] + p[2]) % 256)

    def test_baselines(self):
        rng = np.random.default_rng(0)
        bodies, lp = sample_uniform_programs(100, 32, rng)
        self.assertTrue(all(len(b) <= 32 and set(b) <= set(BODY_ALPHABET.encode()) for b in bodies))
        rows = sample_pcfg_rows(3, 300, rng)
        self.assertEqual(rows.shape, (3, 300))
        self.assertTrue((rows != 0).all())
        g = RandomPCFG(rng)
        self.assertTrue(1 <= len(g.derive_word()) <= 64)


class TestTraining(unittest.TestCase):
    def test_smoke_selfplay_and_baselines(self):
        from spp.train import SelfPlayTrainer, TrainConfig
        with tempfile.TemporaryDirectory() as d:
            for data in ("selfplay", "uniform", "pcfg"):
                cfg = TrainConfig(out_dir=f"{d}/{data}", preset="tiny", data=data, ctx=64, rounds=4,
                                  n_programs=16, n_replay=4, max_program_len=12, step_budget=500,
                                  warmup_rounds=2, micro_batch=8, reward_chunk=8, eval_every=2,
                                  eval_records=8, icl_every=3, icl_instances=4, ckpt_every=4,
                                  sample_every=2, device=DEV, lookback_storage="memory")
                tr = SelfPlayTrainer(cfg)
                tr.train()
                lines = (Path(d) / data / "log.jsonl").read_text().splitlines()
                self.assertEqual(len(lines), 4)
                self.assertTrue((Path(d) / data / "ckpt_last.pt").exists())
                if data == "selfplay":
                    import json
                    rec = json.loads(lines[-1])
                    self.assertIn("reward/mean", rec)
                    self.assertGreater(rec["bank/size"], 0)
                    self.assertEqual(sorted(tr.lookback.keys()), [2, 3, 4])   # theta_2,3 needed by rounds 4..7
                    tr2 = SelfPlayTrainer(cfg)
                    tr2.load_checkpoint(Path(d) / data / "ckpt_last.pt")
                    self.assertEqual(tr2.round, 4)


if __name__ == "__main__":
    unittest.main()
