import numpy as np
from scipy import signal
import colorednoise as cn
from numba import njit


@njit()
def hann_sample(x, intensity=1.0):
    return intensity * 0.6 * np.sin(2 * np.pi * (x - 0.75) / 4) ** 3 * -1


@njit()
def phase_distort(x, intensity=1.0):
    y = np.zeros_like(x)
    for jj in range(x.size):
        y[jj] = x[jj] + hann_sample(x[jj], intensity=intensity)
    return y


class ConfigGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generate_config():

        activate = np.random.binomial(1, 0.2, 5) > 0.5

        names = ["betas", "gammas", "beta_sharpness", "pac", "phase", "cross-pac"]
        config = {}
        for jj, active in enumerate(list(activate)):
            if active:
                vals = np.random.uniform(0, 1, 2)
                config[names[jj]] = list(vals)
            else:
                config[names[jj]] = None
        if np.array([val == None for val in config.values()]).all():
            return ConfigGenerator.generate_config()
        return config


Fs = 2048
T = 10
import colorednoise as cn


class DataGenerator:
    def __init__(
        self,
        cfg,
        snr=10,
        transition_width=0.05,
        T: int = 60 * 10,
        Fs=2048,
        phase_method="window",
        gamma_mode="burst",
    ):

        if T % 10 != 0:
            raise ValueError("T must be a multiple of 10 seconds (10, 20, 30, ...)")

        self.cfg = cfg

        self.t = np.linspace(0, T, T * Fs)
        self.envelope = np.zeros_like(self.t)

        # Create state time-series:
        self.inertia0, self.inertia1 = np.random.uniform(0.6, 0.7), np.random.uniform(
            0.6, 0.7
        )
        self.transitions = np.array(
            [[self.inertia0, 1 - self.inertia0], [1 - self.inertia1, self.inertia1]]
        )

        state = 0
        for n in range(1, T // 10):
            if state == 0:
                new_state = np.random.binomial(1, self.inertia0) > 0.5
            elif state == 1:
                new_state = np.random.binomial(1, self.inertia1) > 0.5
            else:
                raise RuntimeError("Error generating Markov chain")

            self.envelope[n * 10 * Fs : (n + 1) * 10 * Fs] = new_state

            if new_state > state:
                self.envelope[
                    n * 10 * Fs
                    - int(transition_width / 2 * Fs) : n * 10 * Fs
                    + int(transition_width / 2 * Fs)
                ] = np.linspace(
                    0, 1, int(transition_width * Fs)
                )  # sigmoid(np.linspace(-transition_width/2,transition_width/2,int(transition_width*Fs))/transition_width*8)
            elif new_state < state:
                self.envelope[
                    n * 10 * Fs
                    - int(transition_width / 2 * Fs) : n * 10 * Fs
                    + int(transition_width / 2 * Fs)
                ] = np.linspace(
                    1, 0, int(transition_width * Fs)
                )  # sigmoid(-1*np.linspace(-transition_width/2,transition_width/2,int(transition_width*Fs))/transition_width*8)

            state = new_state

        # Generate waveforms:
        self.sigs, self.sigs2 = [], []
        if self.cfg["betas"] is not None:
            beta0 = np.sin(2 * np.pi * self.t * 25) * self.cfg["betas"][0]
            beta1 = np.sin(2 * np.pi * self.t * 25) * self.cfg["betas"][1]

            self.sigs.append(beta1 * self.envelope + beta0 * (1 - self.envelope))

        if self.cfg["gammas"] is not None:
            gamma0 = np.sin(2 * np.pi * self.t * 70) * self.cfg["gammas"][0]
            gamma1 = np.sin(2 * np.pi * self.t * 70) * self.cfg["gammas"][1]

            self.sigs.append(gamma1 * self.envelope + gamma0 * (1 - self.envelope))

        if self.cfg["beta_sharpness"] is not None:
            triangle = signal.sawtooth(2 * np.pi * 21 * self.t + np.pi / 2, width=0.5)
            sinusoid = np.sin(2 * np.pi * self.t * 21)

            sharp0 = triangle * self.cfg["beta_sharpness"][0] + sinusoid * (
                1 - self.cfg["beta_sharpness"][0]
            )
            sharp1 = triangle * self.cfg["beta_sharpness"][1] + sinusoid * (
                1 - self.cfg["beta_sharpness"][1]
            )

            w0, h0 = signal.periodogram(sharp0, fs=Fs)
            w1, h1 = signal.periodogram(sharp1, fs=Fs)

            assert np.array_equal(w0, w1), "Error generating sharp waveforms"

            beta_ix = np.argmin(np.abs(w0 - 21))
            ratio = h0[beta_ix] / h1[beta_ix]

            sharp1 *= np.sqrt(ratio)

            self.sigs.append(sharp1 * self.envelope + sharp0 * (1 - self.envelope))

        if self.cfg["phase"] is not None:
            base = signal.sawtooth(2 * np.pi * 28 * self.t, width=1)
            if phase_method == "sine":
                base_sine = 0.25 * np.sin(0.5 * 2 * np.pi * self.t + np.pi)
                bases = [
                    base + intensity * base_sine for intensity in self.cfg["phase"]
                ]

            elif phase_method == "window":
                bases = [
                    phase_distort(base, intensity=intensity)
                    for intensity in self.cfg["phase"]
                ]
            bases = [(base + 1) * np.pi for base in bases]

            phase0, phase1 = np.sin(bases[0]), np.sin(bases[1])

            w0, h0 = signal.periodogram(phase0, fs=Fs)
            w1, h1 = signal.periodogram(phase1, fs=Fs)
            beta_ix = np.argmin(np.abs(w0 - 28))

            ratio = h0[beta_ix] / h1[beta_ix]
            phase1 *= np.sqrt(ratio)

            # distorted = np.sin(base)
            # clean = np.sin(2 * np.pi * self.t * 28)

            # phase0 = distorted * self.cfg["phase"][0] + clean * (
            #     1 - self.cfg["phase"][0]
            # )
            # phase1 = distorted * self.cfg["phase"][1] + clean * (
            #     1 - self.cfg["phase"][1]
            # )

            self.sigs.append(phase1 * self.envelope + phase0 * (1 - self.envelope))

        if self.cfg["pac"] is not None:

            theta = np.sin(2 * np.pi * 8 * self.t)
            gamma = 0.2 * np.sin(2 * np.pi * 70 * self.t)

            # Generate Gamma waveform:
            gamma_pac = gamma * theta * (theta > 0).astype(float)

            if gamma_mode == "burst":
                b, a = signal.butter(4, [7, 15], "bandpass", fs=Fs)
                gamma_burst = signal.filtfilt(b, a, 5 * np.random.randn(*self.t.shape))
                gamma_burst += 0
                gamma_burst = gamma * (gamma_burst * (gamma_burst > 0).astype(float))
            elif gamma_mode == "continuous":
                gamma_burst = gamma
            else:
                raise ValueError("gamma_mode (burst or continuous) must bu specified")

            w0, h0 = signal.periodogram(gamma_pac, fs=Fs)
            w1, h1 = signal.periodogram(gamma_burst, fs=Fs)

            gamma_ix = np.argmin(np.abs(w0 - 70))
            ratio = h1[gamma_ix] / h0[gamma_ix]

            gamma_pac *= np.sqrt(ratio)

            pac = theta + gamma_pac
            no_pac = theta + gamma_burst

            # w0, h0 = signal.periodogram(pac, fs=Fs)
            # w1, h1 = signal.periodogram(no_pac, fs=Fs)
            # gamma_ix = np.argmin(np.abs(w0 - 70))

            # ratio = h1[gamma_ix] / h0[gamma_ix]
            # gamma *= np.sqrt(ratio)
            # pac = theta + gamma * theta * (theta > 0).astype(float)

            pac0 = pac * self.cfg["pac"][0] + no_pac * (1 - self.cfg["pac"][0])
            pac1 = pac * self.cfg["pac"][1] + no_pac * (1 - self.cfg["pac"][1])

            self.sigs.append(pac1 * self.envelope + pac0 * (1 - self.envelope))

        if self.cfg["cross-pac"] is not None:
            theta = np.sin(2 * np.pi * 8 * self.t)
            gamma = 0.2 * np.sin(2 * np.pi * 70 * self.t)

            gamma_pac = gamma * theta * (theta > 0).astype(float)

            if gamma_mode == "burst":
                b, a = signal.butter(4, [7, 15], "bandpass", fs=Fs)
                gamma_burst = signal.filtfilt(b, a, 5 * np.random.randn(*self.t.shape))
                gamma_burst += 0
                gamma_burst = gamma * (gamma_burst * (gamma_burst > 0).astype(float))
            elif gamma_mode == "continuous":
                gamma_burst = gamma
            else:
                raise ValueError("gamma_mode (burst or continuous) must bu specified")

            w0, h0 = signal.periodogram(gamma_pac, fs=Fs)
            w1, h1 = signal.periodogram(gamma_burst, fs=Fs)

            gamma_ix = np.argmin(np.abs(w0 - 70))
            ratio = h1[gamma_ix] / h0[gamma_ix]

            gamma_pac *= np.sqrt(ratio)

            gamma0 = (
                self.cfg["cross-pac"][0] * gamma_pac
                + (1 - self.cfg["cross-pac"][0]) * gamma_burst
            )
            gamma1 = (
                self.cfg["cross-pac"][1] * gamma_pac
                + (1 - self.cfg["cross-pac"][1]) * gamma_burst
            )

            # w0, h0 = signal.periodogram(gamma0, fs=Fs)
            # w1, h1 = signal.periodogram(gamma1, fs=Fs)
            # gamma_ix = np.argmin(np.abs(w0 - 70))

            # ratio = h1[gamma_ix] / h0[gamma_ix]
            # gamma0 *= np.sqrt(ratio)

            self.sigs.append(theta)
            self.sigs2.append(self.envelope * gamma1 + (1 - self.envelope) * gamma0)

        if self.cfg["phase-shift"] is not None:

            b, a = signal.butter(3, 9, fs=2048)
            f = signal.filtfilt(b, a, 50 * np.random.rand(*self.t.shape))

            base = np.sin(2 * np.pi * self.t * f)

            v0 = np.sin(2 * np.pi * self.t * f + np.pi * self.cfg["phase-shift"][0])
            v1 = np.sin(2 * np.pi * self.t * f + np.pi * self.cfg["phase-shift"][1])

            self.sigs.append(base)
            self.sigs2.append(self.envelope * v1 + (1 - self.envelope) * v0)

        if self.cfg["burst-length"] is not None:
            beta = np.sin(2 * np.pi * 22 * self.t)

            # Bursts::
            burst0 = self.cfg["burst-length"][0]
            b, a = signal.butter(2, [2 - burst0, (2 - burst0) * 5], "bandpass", fs=Fs)
            burst_envelope_0 = signal.filtfilt(b, a, 7 * np.random.randn(*self.t.shape))
            burst_envelope_0 += 0.45 * burst0
            bursty_beta_0 = beta * (
                burst_envelope_0 * (burst_envelope_0 > 0).astype(float)
            )

            burst1 = self.cfg["burst-length"][1]
            b, a = signal.butter(2, [2 - burst1, (2 - burst1) * 5], "bandpass", fs=Fs)
            burst_envelope_1 = signal.filtfilt(b, a, 7 * np.random.randn(*self.t.shape))
            burst_envelope_1 += 0.45 * burst1
            bursty_beta_1 = beta * (
                burst_envelope_1 * (burst_envelope_1 > 0).astype(float)
            )

            w0, h0 = signal.periodogram(bursty_beta_0, fs=Fs)
            w1, h1 = signal.periodogram(bursty_beta_1, fs=Fs)

            beta_ix = np.argmin(np.abs(w0 - 22))
            ratio = h1[beta_ix] / h0[beta_ix]

            bursty_beta_0 *= np.sqrt(ratio)

            self.sigs.append(
                bursty_beta_1 * self.envelope + (1 - self.envelope) * bursty_beta_0
            )

        # Merge signals into final sig:

        self.signal_clean = np.stack(self.sigs).sum(0)
        noise_level = (
            np.sqrt((self.signal_clean**2).sum() / self.signal_clean.size) / snr
        )
        noise = cn.powerlaw_psd_gaussian(1, self.signal_clean.shape) * noise_level
        self.signal = self.signal_clean + noise

        if self.sigs2:
            self.signal2_clean = np.stack(self.sigs2).sum(0)
            noise_level = (
                np.sqrt((self.signal2_clean**2).sum() / self.signal2_clean.size) / snr
            )
            noise = cn.powerlaw_psd_gaussian(1, self.signal2_clean.shape) * noise_level
            self.signal2 = self.signal2_clean + noise

            self.signal = np.stack((self.signal, self.signal2))

        if self.signal.ndim < 2:
            self.signal = self.signal[np.newaxis, ...]

        self.label = (self.envelope > 0.5).astype(float)
