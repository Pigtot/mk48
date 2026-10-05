from dataclasses import asdict

import numpy as np
import torch

import entity_policy as ep


L = ep.V6


def contacts_of(obs: np.ndarray) -> np.ndarray:
    return obs[:, L.self_dim : L.self_dim + ep.K * L.contact_dim].reshape(len(obs), ep.K, L.contact_dim)


def random_obs(n: int, present: int = 5) -> np.ndarray:
    obs = np.zeros((n, L.obs_dim), np.float32)
    obs[:, L.own_type] = 17  # FairmileD
    contacts = contacts_of(obs)
    contacts[:, :present, L.contact_type] = 9  # ArleighBurke + 1
    contacts[:, :present, 0] = 1.0
    contacts[:, :present, 1:3] = np.random.uniform(-0.8, 0.8, (n, present, 2))
    return obs


def test_forward_shapes_and_masking():
    policy = ep.Policy()
    obs = torch.as_tensor(random_obs(6, present=3))
    logits = policy(obs)
    for name, k in L.heads().items():
        assert logits[name].shape == (6, k)
    actions = ep.sample(logits)
    assert actions.shape == (6, len(L.head_names))
    # Absent contacts (index > 3 in the pointer) are never sampled.
    for _ in range(20):
        assert ep.sample(logits)[:, ep.H["target"]].max() <= 3
    assert torch.isfinite(ep.log_prob(logits, actions)).all()
    assert torch.isfinite(ep.entropy(logits)).all()
    assert torch.allclose(ep.kl(logits, logits), torch.zeros(6), atol=1e-5)


def test_expert_roundtrip():
    """Expert continuous actions -> discrete labels -> server actions keeps the important parts."""
    obs = random_obs(4, present=4)
    contacts = contacts_of(obs)
    expert = np.zeros((4, 11), np.float32)
    expert[:, 0] = [0.0, 0.5, -0.99, 0.26]  # steer
    expert[:, 1] = 0.8
    expert[:, 2:4] = contacts[:, 2, 1:3] + 0.01  # aiming at contact 2
    expert[:, 4] = [1, -1, 1, -1]  # fire
    expert[:, 5] = L.class_values[[0, 1, 6, 2]]
    expert[:, 9] = 1  # valid
    expert[:, 10] = 1  # aim valid
    labels, masks = ep.from_expert(expert, obs)
    assert (labels[:, ep.H["target"]] == 3).all()  # contact 2 -> pointer index 3
    assert list(masks[:, ep.H["weapon"]]) == [1, 0, 1, 0]
    env = ep.to_env(labels, obs)
    assert np.allclose(env[:, 2:4], contacts[:, 2, 1:3])
    assert np.allclose(env[:, 0], [0.0, 0.5, -1.0, 0.25])
    assert list(env[:, 4]) == [1, -1, 1, -1]
    assert np.allclose(env[[0, 2], 5], L.class_values[[0, 6]])


def test_no_target_when_aim_invalid():
    obs = random_obs(2)
    expert = np.zeros((2, 11), np.float32)
    expert[:, 9] = 1
    labels, _ = ep.from_expert(expert, obs)
    assert (labels[:, ep.H["target"]] == 0).all()


def test_v5_checkpoints_still_load(tmp_path):
    policy = ep.Policy(ep.V5)
    ep.save(tmp_path / "p.pt", policy)
    state = __import__("torch").load(tmp_path / "p.pt", weights_only=False)
    del state["layout"]  # checkpoints written before the layout field existed
    __import__("torch").save(state, tmp_path / "p.pt")
    loaded, _, _ = ep.load(tmp_path / "p.pt")
    assert loaded.layout == ep.V5


def test_v7_upgrade_adds_salvo():
    v6 = ep.Policy(ep.V6)
    v7 = ep.upgrade(v6, ep.V7)
    obs = torch.as_tensor(random_obs(5))
    logits6, logits7 = v6(obs), v7(obs)
    # Shared heads are unchanged; the new salvo head starts out preferring single shots.
    assert torch.allclose(logits6["steer"], logits7["steer"])
    assert (logits7["salvo"].softmax(-1)[:, 0] > 0.8).all()
    actions = ep.sample(logits7)
    assert actions.shape == (5, len(ep.V7.head_names))
    env = ep.to_env(actions.numpy(), obs.numpy(), ep.V7)
    assert env.shape == (5, 10) and set(np.unique(env[:, 9])) <= {-1.0, 1.0}
    # KL against an older reference only covers the heads both have.
    assert torch.allclose(ep.kl(logits6, logits7), torch.zeros(5), atol=1e-5)
    assert torch.isfinite(ep.log_prob(logits7, actions)).all()


def test_v7_expert_labels_mask_salvo():
    obs = random_obs(3)
    expert = np.zeros((3, ep.V7.act_dim + 2), np.float32)
    expert[:, -2] = 1  # valid
    labels, masks = ep.from_expert(expert, obs, ep.V7)
    assert labels.shape == (3, len(ep.V7.head_names))
    assert (masks[:, ep.H["salvo"]] == 0).all()


def test_defense_labels():
    """SAM against an incoming missile a ready SAM can hit; decoy against a closing torpedo."""
    obs = torch.zeros(3, ep.V7.obs_dim)

    def contact(row, k, features):
        for i, v in features.items():
            obs[row, ep.V7.self_dim + k * ep.V7.contact_dim + i] = v

    contact(0, 2, {0: 1, 14: 1, 21: 1, 36: 1, 1: 0.3, 3: -1.0, 7: 0.3})  # missile, SAM can hit
    contact(1, 5, {0: 1, 14: 1, 19: 1, 1: 0.2, 3: -1.0, 7: 0.2})  # torpedo closing
    obs[1, 19] = 1  # own decoy ready
    contact(2, 1, {0: 1, 9: 1, 7: 0.1})  # a ship: nothing to defend against
    rows, target, weapon = ep.defense_labels(obs)
    assert rows.tolist() == [True, True, False]
    assert target[:2].tolist() == [3, 6] and weapon[:2].tolist() == [5, 6]


def test_doctrine_labels():
    """Carriers launch aircraft at ships in reach; submerged submarines keep active sonar off."""
    obs = torch.zeros(3, ep.V7.obs_dim)

    def contact(row, k, features):
        for i, v in features.items():
            obs[row, ep.V7.self_dim + k * ep.V7.contact_dim + i] = v

    obs[0, 34] = 1  # own family: carrier
    contact(0, 4, {0: 1, 9: 1, 34: 1, 7: 0.4})  # enemy ship a ready aircraft can reach
    obs[1, 11] = obs[1, 10] = 1  # a submarine, submerged
    contact(2, 1, {0: 1, 9: 1, 34: 1, 7: 0.2})  # not a carrier: nothing to teach
    air, target, stealth = ep.doctrine_labels(obs)
    assert air.tolist() == [True, False, False] and target[0].item() == 5
    assert stealth.tolist() == [False, True, False]


def test_phased_policy_switches_by_level(tmp_path):
    """The opening policy drives small ships, the main policy the rest; recipes load from JSON."""
    torch.manual_seed(0)
    opening, main = ep.Policy(ep.V7), ep.Policy(ep.V7)
    torch.save({"layout": asdict(ep.V7), "policy": opening.state_dict()}, tmp_path / "opening.pt")
    torch.save({"layout": asdict(ep.V7), "policy": main.state_dict()}, tmp_path / "main.pt")
    (tmp_path / "phased.json").write_text('{"opening": "opening.pt", "main": "main.pt", "until_level": 4}')
    phased = ep.load(tmp_path / "phased.json")[0]
    obs = torch.zeros(2, ep.V7.obs_dim)
    obs[:, 0] = 1
    obs[0, 8], obs[1, 8] = 0.4, 0.5  # levels 4 and 5
    with torch.no_grad():
        got, a, b = phased(obs), opening(obs), main(obs)
    for head in got:  # each network computes only its rows: equal up to float rounding
        assert torch.allclose(got[head][0], a[head][0], atol=1e-5)
        assert torch.allclose(got[head][1], b[head][1], atol=1e-5)
