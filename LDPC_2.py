import time
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import dimod
import commpy.modulation as mod


def gf2_rref(matrix):
    """تحويل مصفوفة ثنائية إلى الصيغة الصفية المختصرة في GF(2)"""
    A = matrix.copy().astype(int) % 2
    rows, cols = A.shape
    pivots = []
    r = 0
    for c in range(cols):
        pivot_row = None
        for i in range(r, rows):
            if A[i, c] == 1:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        A[[r, pivot_row]] = A[[pivot_row, r]]
        for i in range(rows):
            if i != r and A[i, c] == 1:
                A[i] = (A[i] + A[r]) % 2
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return A, pivots

def build_regular_ldpc(N=24, M=12, wc=2, wr=4, seed=42):
    """
    بناء مصفوفة تحقق متفرقة منتظمة تقريباً H بأبعاد (M x N)
    واستخراج مصفوفة التوليد G بحيث H * G^T = 0 (mod 2).
    """
    rng = np.random.default_rng(seed)
    H = np.zeros((M, N), dtype=int)

    total_ones = M * wr
    col_counts = np.zeros(N, dtype=int)
    row_counts = np.zeros(M, dtype=int)

    for j in range(M):
        candidate_cols = np.argsort(col_counts + rng.random(N) * 0.1)[:wr]
        H[j, candidate_cols] = 1
        row_counts[j] = wr
        col_counts[candidate_cols] += 1

    rref_H, pivots = gf2_rref(H)
    rank = len(pivots)
    if rank < M:
        for j in range(rank, M):
            idx = rng.choice(N, size=wr, replace=False)
            H[j] = 0
            H[j, idx] = 1
        rref_H, pivots = gf2_rref(H)
        rank = len(pivots)

    K = N - rank
    free_cols = [c for c in range(N) if c not in pivots]

    perm = pivots + free_cols
    H_perm = H[:, perm]
    rref_perm, _ = gf2_rref(H_perm)

    G_perm = np.zeros((K, N), dtype=int)
    rref_clean, piv_clean = gf2_rref(H_perm)
    A_part = rref_clean[:len(piv_clean), len(piv_clean):]
    if A_part.shape[1] == K and A_part.shape[0] == len(piv_clean):
        G_perm[:, :len(piv_clean)] = A_part.T % 2
        G_perm[:, len(piv_clean):] = np.eye(K, dtype=int)
    else:
        basis = []
        for x_int in range(1, 2**N):
            vec = np.array([(x_int >> b) & 1 for b in range(N)], dtype=int)
            if np.all((H @ vec) % 2 == 0):
                if not basis:
                    basis.append(vec)
                else:
                    B = np.vstack(basis + [vec])
                    _, p_b = gf2_rref(B)
                    if len(p_b) > len(piv_clean) - 1 and len(p_b) == len(basis) + 1:
                        basis.append(vec)
                if len(basis) == K:
                    break
        G = np.vstack(basis[:K]) % 2
        return H, G, K

    inv_perm = np.argsort(perm)
    G = G_perm[:, inv_perm] % 2

    assert np.all((H @ G.T) % 2 == 0), "فشل التحقق من أن H * G^T = 0"
    return H, G, K

def build_ldpc_bqm(H, llr, penalty_weight=4.0):
    """
    يبني نموذج BQM لفك تشفير LDPC:
    - الحد الخطي: طاقة القناة المستمدة من LLR
    - الحدود التربيعية: عقوبة عدم تحقق معادلات الباريتي في H
    """
    bqm = dimod.BinaryQuadraticModel(vartype=dimod.BINARY)
    M, N = H.shape

    for i in range(N):
        bqm.add_variable(f"c_{i}", float(llr[i]))

    def add_xor_eq(v1, v2, v3, w):
        bqm.add_linear(v1, w)
        bqm.add_linear(v2, w)
        bqm.add_linear(v3, w)
        bqm.add_quadratic(v1, v2, 2.0 * w)
        bqm.add_quadratic(v1, v3, -2.0 * w)
        bqm.add_quadratic(v2, v3, -2.0 * w)

    aux_id = 0
    for j in range(M):
        cols = np.where(H[j] == 1)[0]
        v_names = [f"c_{idx}" for idx in cols]
        k = len(v_names)

        if k == 0 or k == 1:
            continue
        elif k == 2:
            v1, v2 = v_names
            bqm.add_linear(v1, penalty_weight)
            bqm.add_linear(v2, penalty_weight)
            bqm.add_quadratic(v1, v2, -2.0 * penalty_weight)
        else:
            curr = v_names[0]
            for idx in range(1, k - 2):
                aux = f"aux_{aux_id}"
                aux_id += 1
                add_xor_eq(curr, v_names[idx], aux, penalty_weight)
                curr = aux
            add_xor_eq(curr, v_names[-2], v_names[-1], penalty_weight)

    return bqm


def sqa_inspired_decode(H, llr, n_trotter=8, n_sweeps=80, penalty_factor=1.5, seed=None):
    """
    خوارزمية مستوحاة من التلدين الكمي (Path-Integral / Trotter-inspired SQA)
    """
    rng = np.random.default_rng(seed)
    M, N = H.shape
    penalty_weight = max(np.mean(np.abs(llr)) * penalty_factor, 1.0)
    bqm = build_ldpc_bqm(H, llr, penalty_weight=penalty_weight)

    var_names = list(bqm.variables)
    var_to_idx = {name: i for i, name in enumerate(var_names)}
    V = len(var_names)

    linear = np.zeros(V, dtype=float)
    for name, bias in bqm.linear.items():
        linear[var_to_idx[name]] = bias

    quad = np.zeros((V, V), dtype=float)
    for (u, v), val in bqm.quadratic.items():
        iu, iv = var_to_idx[u], var_to_idx[v]
        quad[iu, iv] += val
        quad[iv, iu] += val

    init_state = (llr < 0).astype(int)
    spins = np.tile(init_state[:N], (n_trotter, 1))
    if V > N:
        aux_init = rng.integers(0, 2, size=(n_trotter, V - N))
        spins = np.hstack([spins, aux_init])

    betas = np.linspace(0.5, 6.0, n_sweeps)
    j_perp_schedule = np.linspace(0.5, 4.0, n_sweeps)

    for step in range(n_sweeps):
        beta = betas[step]
        j_perp = j_perp_schedule[step]

        for t in range(n_trotter):
            t_prev = (t - 1) % n_trotter
            t_next = (t + 1) % n_trotter

            for i in range(V):
                s_curr = spins[t, i]
                s_flip = 1 - s_curr

                neighbor_sum = np.dot(quad[i], spins[t]) - quad[i, i] * s_curr
                e_curr = linear[i] * s_curr + s_curr * neighbor_sum
                e_flip = linear[i] * s_flip + s_flip * neighbor_sum
                delta_e = e_flip - e_curr

                trotter_curr = j_perp * ((s_curr != spins[t_prev, i]) + (s_curr != spins[t_next, i]))
                trotter_flip = j_perp * ((s_flip != spins[t_prev, i]) + (s_flip != spins[t_next, i]))
                delta_trotter = trotter_flip - trotter_curr

                total_delta = delta_e + delta_trotter
                if total_delta <= 0.0 or rng.random() < np.exp(-beta * total_delta):
                    spins[t, i] = s_flip

    energies = np.zeros(n_trotter)
    for t in range(n_trotter):
        s = spins[t]
        energies[t] = np.dot(linear, s) + 0.5 * (s @ quad @ s)

    best_t = int(np.argmin(energies))
    c_hat = spins[best_t, :N].astype(int)
    return c_hat


def hard_decision_decode(llr):
    return (llr < 0).astype(int)

def plot_4x4(eb_n0_dbs, ber_hard, ber_sqa, time_hard, time_sqa, N, K):
    """يعرض 16 رسماً بيانياً شاملاً في شبكة 4×4."""
    ber_hard = np.array(ber_hard)
    ber_sqa = np.array(ber_sqa)
    time_hard = np.array(time_hard)
    time_sqa = np.array(time_sqa)
    eb = np.array(eb_n0_dbs)
    rate = K / N

    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    fig.suptitle(
        f"LDPC Decoding Analysis "
        f"(N={N}, K={K}, Rate={rate:.2f})",
        fontsize=16, fontweight="bold"
    )

    ax = axes[0, 0]
    ax.semilogy(eb, ber_hard, "o--", color="tab:blue", label="Hard")
    ax.semilogy(eb, ber_sqa, "s-", color="tab:red", label="SQA")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("BER (log)")
    ax.set_title("1. BER Comparison")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(eb, ber_hard, "o--", color="tab:blue", label="Hard")
    ax.plot(eb, ber_sqa, "s-", color="tab:red", label="SQA")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("BER (linear)")
    ax.set_title("2. BER Comparison (Linear)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()

    ax = axes[0, 2]
    improvement = (ber_hard - ber_sqa) / ber_hard * 100
    ax.plot(eb, improvement, "^-", color="tab:green", markersize=8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Improvement (%)")
    ax.set_title("3. SQA BER Improvement over Hard")
    ax.grid(True, ls=":", alpha=0.6)

    ax = axes[0, 3]
    ax.semilogy(eb, ber_hard, "o--", color="tab:blue", label="Hard")
    ax.semilogy(eb, ber_sqa, "s-", color="tab:red", label="SQA")
    ax.fill_between(eb, ber_sqa, ber_hard, where=(ber_hard >= ber_sqa),
                    color="tab:green", alpha=0.2, label="SQA gain")
    ax.fill_between(eb, ber_sqa, ber_hard, where=(ber_hard < ber_sqa),
                    color="tab:orange", alpha=0.2, label="SQA loss")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("BER (log)")
    ax.set_title("4. SQA Gain/Loss Region")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.semilogy(eb, time_hard * 1e6, "o--", color="tab:blue", label="Hard (µs)")
    ax.semilogy(eb, time_sqa * 1e3, "s-", color="tab:red", label="SQA (ms)")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Time (log)")
    ax.set_title("5. Decoding Time Comparison")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(eb, time_hard * 1e6, "o--", color="tab:blue", label="Hard (µs)")
    ax.plot(eb, time_sqa * 1e3, "s-", color="tab:red", label="SQA (ms)")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Time")
    ax.set_title("6. Time Comparison (Linear)")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()

    ax = axes[1, 2]
    speedup = time_sqa / time_hard
    ax.semilogy(eb, speedup, "D-", color="tab:purple", markersize=8)
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("t_SQA / t_Hard (log)")
    ax.set_title("7. Speedup Ratio (x times slower)")
    ax.grid(True, which="both", ls=":", alpha=0.6)

    ax = axes[1, 3]
    ax.bar(eb - 0.15, time_hard * 1e6, width=0.3, color="tab:blue", label="Hard (µs)")
    ax.bar(eb + 0.15, time_sqa * 1e3, width=0.3, color="tab:red", label="SQA (ms)")
    ax.set_yscale("log")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Time per frame (log)")
    ax.set_title("8. Time per Frame (Bar Chart)")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[2, 0]
    ax.scatter(time_hard * 1e6, ber_hard, c=eb, cmap="Blues", s=120,
               marker="o", edgecolors="k", label="Hard")
    ax.scatter(time_sqa * 1e3, ber_sqa, c=eb, cmap="Reds", s=120,
               marker="s", edgecolors="k", label="SQA")
    for i, txt in enumerate(eb):
        ax.annotate(f"{txt:g}dB", (time_hard[i] * 1e6, ber_hard[i]), fontsize=7, ha="right")
        ax.annotate(f"{txt:g}dB", (time_sqa[i] * 1e3, ber_sqa[i]), fontsize=7, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time (log) [µs Hard / ms SQA]")
    ax.set_ylabel("BER (log)")
    ax.set_title("9. BER-Time Trade-off")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[2, 1]
    eff_hard = 1.0 / (ber_hard * time_hard + 1e-12)
    eff_sqa = 1.0 / (ber_sqa * time_sqa + 1e-12)
    ax.semilogy(eb, eff_hard, "o--", color="tab:blue", label="Hard")
    ax.semilogy(eb, eff_sqa, "s-", color="tab:red", label="SQA")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Efficiency = 1/(BER×t) (log)")
    ax.set_title("10. Efficiency Metric")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[2, 2]
    ax.plot(eb, ber_hard * N, "o--", color="tab:blue", label="Hard (errors/frame)")
    ax.plot(eb, ber_sqa * N, "s-", color="tab:red", label="SQA (errors/frame)")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Avg errors per frame")
    ax.set_title("11. Errors per Frame")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()

    ax = axes[2, 3]
    ax.semilogy(eb, 1.0 / ber_hard, "o--", color="tab:blue", label="1/BER Hard")
    ax.semilogy(eb, 1.0 / ber_sqa, "s-", color="tab:red", label="1/BER SQA")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("1 / BER (log)")
    ax.set_title("12. Inverse BER (higher = better)")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[3, 0]
    ber_hard_n = ber_hard / ber_hard[0]
    ber_sqa_n = ber_sqa / ber_sqa[0]
    ax.semilogy(eb, ber_hard_n, "o--", color="tab:blue", label="Hard (norm.)")
    ax.semilogy(eb, ber_sqa_n, "s-", color="tab:red", label="SQA (norm.)")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Normalized BER (log)")
    ax.set_title("13. BER Normalized to 0 dB")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[3, 1]
    time_hard_n = time_hard / time_hard[0]
    time_sqa_n = time_sqa / time_sqa[0]
    ax.plot(eb, time_hard_n, "o--", color="tab:blue", label="Hard (norm.)")
    ax.plot(eb, time_sqa_n, "s-", color="tab:red", label="SQA (norm.)")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Normalized Time")
    ax.set_title("14. Time Normalized to 0 dB")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend()

    ax = axes[3, 2]
    cost_hard = ber_hard * time_hard
    cost_sqa = ber_sqa * time_sqa
    ax.semilogy(eb, cost_hard, "o--", color="tab:blue", label="BER × t Hard")
    ax.semilogy(eb, cost_sqa, "s-", color="tab:red", label="BER × t SQA")
    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("BER × Time (log, lower = better)")
    ax.set_title("15. Combined Cost (BER × Time)")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[3, 3]
    categories = ["BER @ last", "t @ last (norm.)", "log₁₀(Speedup)"]
    hard_vals = [ber_hard[-1], 1.0, 0]
    sqa_vals = [ber_sqa[-1], time_sqa[-1] / time_hard[-1], np.log10(speedup[-1])]
    x = np.arange(len(categories))
    ax.bar(x - 0.2, hard_vals, width=0.4, color="tab:blue", label="Hard")
    ax.bar(x + 0.2, sqa_vals, width=0.4, color="tab:red", label="SQA")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("Value (log)")
    ax.set_title(f"16. Summary @ {eb[-1]:g} dB")
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    for row in axes:
        for ax in row:
            ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig("ldpc_4x4_analysis.png", dpi=130, bbox_inches="tight")
    plt.show()
   
def run_simulation():
    N, M = 24, 12
    H, G, K = build_regular_ldpc(N=N, M=M, wc=2, wr=4, seed=42)
    rate = K / N

    modem = mod.PSKModem(2)

    eb_n0_dbs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    num_frames = 150
    rng = np.random.default_rng(123)

    ber_hard, ber_sqa = [], []
    time_hard, time_sqa = [], []   # متوسط زمن فك التشفير لكل إطار (ثانية)

    for eb_n0_db in eb_n0_dbs:
        eb_n0_lin = 10.0 ** (eb_n0_db / 10.0)
        noise_var = 1.0 / (2.0 * rate * eb_n0_lin)
        sigma = np.sqrt(noise_var)

        err_bits_hard = 0
        err_bits_sqa = 0
        total_bits = num_frames * N

        t_hard_acc = 0.0
        t_sqa_acc = 0.0

        for frame_idx in range(num_frames):
            u = rng.integers(0, 2, size=K)
            c = (u @ G) % 2

            tx_sym = modem.modulate(c).real
            noise = rng.normal(0.0, sigma, size=N)
            rx_sym = tx_sym + noise
            llr = (2.0 * rx_sym) / noise_var

            # فك التشفير: Hard Decision
            t0 = time.perf_counter()
            c_hard = hard_decision_decode(llr)
            t_hard_acc += (time.perf_counter() - t0)

            # فك التشفير: SQA
            seed_i = int(rng.integers(0, 10**9))
            t0 = time.perf_counter()
            c_sqa = sqa_inspired_decode(H, llr, n_trotter=6, n_sweeps=60, seed=seed_i)
            t_sqa_acc += (time.perf_counter() - t0)

            err_bits_hard += np.sum(c != c_hard)
            err_bits_sqa += np.sum(c != c_sqa)

        ber_hard.append(max(err_bits_hard / total_bits, 1e-6))
        ber_sqa.append(max(err_bits_sqa / total_bits, 1e-6))

        mean_t_hard = t_hard_acc / num_frames
        mean_t_sqa = t_sqa_acc / num_frames
        time_hard.append(mean_t_hard)
        time_sqa.append(mean_t_sqa)

        print(f"Eb/N0 = {eb_n0_db:3.1f} dB | "
              f"BER Hard = {ber_hard[-1]:.4f} | BER SQA = {ber_sqa[-1]:.4f} | "
              f"t_Hard = {mean_t_hard*1e6:8.2f} µs/frame | "
              f"t_SQA  = {mean_t_sqa*1e3:8.2f} ms/frame")

    # جدول تفصيلي
    print("\n" + "=" * 90)
    print(f"{'Eb/N0 (dB)':>10} | {'BER Hard':>10} | {'BER SQA':>10} | "
          f"{'t_Hard (µs)':>12} | {'t_SQA (ms)':>12} | {'Speedup':>10}")
    print("=" * 90)
    for k in range(len(eb_n0_dbs)):
        speedup = time_sqa[k] / time_hard[k]
        print(f"{eb_n0_dbs[k]:>10.1f} | {ber_hard[k]:>10.4f} | {ber_sqa[k]:>10.4f} | "
              f"{time_hard[k]*1e6:>12.2f} | {time_sqa[k]*1e3:>12.2f} | "
              f"{speedup:>9.1f}x")
    print("=" * 90)

    # عرض 4×4 الشامل
    plot_4x4(eb_n0_dbs, ber_hard, ber_sqa, time_hard, time_sqa, N, K)

if __name__ == "__main__":
    run_simulation()