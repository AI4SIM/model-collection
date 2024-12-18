# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from matplotlib import rc, rcParams
import matplotlib.pyplot as plt
import numpy as np

from metrics import bias, mae, mape, percentiles


LONGNAMES = {"sw": "Shortwave", "lw": "Longwave"}


class Plotter:

    plt.style.use("fivethirtyeight")
    rc("text", usetex=True)
    # rc("font", weight="bold")
    rc("font", size=16)
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Computer Modern Roman"]
    rcParams["mathtext.fontset"] = "custom"
    rcParams["mathtext.it"] = "STIXGeneral:italic"
    rcParams["mathtext.bf"] = "STIXGeneral:italic:bold"

    def __init__(self, predictions, batch):
        self.predictions = predictions
        self.batch = batch
        self.level = np.arange(self.batch["dn_sw"].shape[1])
        self.half_level = np.arange(self.batch["hr_sw"].shape[1])

    def random_samples(self, flux="sw", n_samples=5):
        indexes = np.random.randint(
            0, self.predictions["dn_" + flux].shape[0], n_samples
        )

        fig, ax = plt.subplots(3, n_samples, figsize=(15, 15))
        for idx, i in enumerate(indexes):
            ax[0, idx].plot(
                self.batch["dn_" + flux][i, :], self.level, lw=2.5, label="Truth"
            )
            ax[0, idx].plot(
                self.predictions["dn_" + flux][i, :], self.level, lw=1.5, label="Pred"
            )
            ax[0, idx].set_xlabel(rf"Down {flux.upper()} Flux ($W m^{{-2}}$)")
            ax[0, idx].set_title(f"Sample {i}")
            ax[0, idx].invert_yaxis()

            ax[1, idx].plot(
                self.batch["up_" + flux][i, :], self.level, lw=2.5, label="Truth"
            )
            ax[1, idx].plot(
                self.predictions["up_" + flux][i, :], self.level, lw=1.5, label="Pred"
            )
            ax[1, idx].set_xlabel(rf"Up {flux.upper()} Flux ($W m^{{-2}}$)")
            ax[1, idx].invert_yaxis()

            ax[2, idx].plot(
                self.batch["hr_" + flux][i, :], self.half_level, lw=2.5, label="Truth"
            )
            ax[2, idx].plot(
                self.predictions["hr_" + flux][i, :],
                self.half_level,
                lw=1.5,
                label="Pred",
            )
            ax[2, idx].set_xlabel(r"Heat. Rate ($K d^{-1}$)")
            ax[2, idx].invert_yaxis()

            if idx == 0:
                for i in range(3):
                    ax[i, idx].set_ylabel("Level")

            if idx == 3:
                ax[2, idx].legend(loc="best")

            for ax_ in ax.flatten():
                ax_.ticklabel_format(axis="x", useMathText=True)

        return fig

    def mae_bias_profile(self, flux="sw"):
        fig, ax = plt.subplots(1, 3, figsize=(15, 7))

        metrics = {}
        for metric in [mae, bias]:
            for key in ["dn_" + flux, "up_" + flux, "hr_" + flux]:
                metrics[f"{metric.__name__}_{key}"] = metric(
                    self.batch[key], self.predictions[key]
                )

        quantiles = {}
        for key in ["dn_" + flux, "up_" + flux, "hr_" + flux]:
            quantiles[key] = percentiles(self.batch[key], self.predictions[key])

        ax[0].fill_betweenx(
            self.level,
            quantiles["dn_" + flux][0],
            quantiles["dn_" + flux][-1],
            color="#008fd5",
            alpha=0.25,
        )
        ax[0].fill_betweenx(
            self.level,
            quantiles["dn_" + flux][1],
            quantiles["dn_" + flux][-2],
            color="#008fd5",
            alpha=0.50,
        )
        ax[0].plot(metrics["bias_dn_" + flux], self.level, lw=2.0, label="bias")
        ax[0].plot(metrics["mae_dn_" + flux], self.level, lw=2.0, label="MAE")
        ax[0].set_xlim(-0.3, 0.3)
        ax[0].xaxis.set_ticks(np.arange(-0.3, 0.3, 0.1))
        ax[0].invert_yaxis()
        ax[0].set_xlabel(r"Flux ($W m^{-2}$)")
        ax[0].set_ylabel("Level")
        ax[0].set_title("Downward " + LONGNAMES[flux] + " Flux", fontweight="bold")

        ax[1].fill_betweenx(
            self.level,
            quantiles["up_" + flux][0],
            quantiles["up_" + flux][-1],
            color="#008fd5",
            alpha=0.25,
        )
        ax[1].fill_betweenx(
            self.level,
            quantiles["up_" + flux][1],
            quantiles["up_" + flux][-2],
            color="#008fd5",
            alpha=0.50,
        )
        ax[1].plot(metrics["bias_dn_" + flux], self.level, lw=2.0, label="bias")
        ax[1].plot(metrics["mae_dn_" + flux], self.level, lw=2.0, label="MAE")
        ax[0].set_xlim(-0.3, 0.3)
        ax[0].xaxis.set_ticks(np.arange(-0.3, 0.3, 0.1))
        ax[1].invert_yaxis()
        ax[1].set_xlabel(r"Flux ($W m^{-2}$)")
        ax[1].set_title("Upward " + LONGNAMES[flux] + " Flux", fontweight="bold")

        ax[2].fill_betweenx(
            self.half_level,
            quantiles["hr_" + flux][0],
            quantiles["hr_" + flux][-1],
            color="#008fd5",
            alpha=0.25,
        )
        ax[2].fill_betweenx(
            self.half_level,
            quantiles["hr_" + flux][1],
            quantiles["hr_" + flux][-2],
            color="#008fd5",
            alpha=0.50,
        )
        ax[2].plot(metrics["bias_hr_" + flux], self.half_level, lw=2.0, label="bias")
        ax[2].plot(metrics["mae_hr_" + flux], self.half_level, lw=2.0, label="MAE")
        ax[2].set_xlim(-0.05, 0.05)
        ax[2].ticklabel_format(
            axis="x", style="sci", scilimits=[-2, 2], useMathText=True
        )
        ax[2].invert_yaxis()
        ax[2].set_xlabel(r"Heat. Rate ($K d^{-1}$)")
        ax[2].set_title(LONGNAMES[flux] + " Heating Rate", fontweight="bold")

        for ax_ in ax:
            ax_.legend(loc="best")

        return fig

    def hist2d(self):
        pass
