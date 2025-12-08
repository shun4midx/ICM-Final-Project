import numpy as np
from utils import load_image, save_image
from kernels import gaussian_kernel, motion_kernel, box_kernel, random_kernel, key_image_kernel_rgb
from fft_deconv import apply_kernel_fft, deconv_fft
from metrics import mse, psnr
import csv
import os

IMAGES = ["antimeme.png", "this_is_fine.png", "save_me.png"]
IMAGE_DIR = "../images/"
RESULT_DIR = "../results/images/"
CSV_PATH = "../results/metrics.csv"

for img_name in IMAGES:
    img_folder = f"{RESULT_DIR}/{img_name[:-4]}"
    os.makedirs(img_folder, exist_ok=True)

def run_experiment():
    kernel_functions = {
        "Gaussian": lambda: gaussian_kernel(size=25, sigma=5),
        "Motion": lambda: motion_kernel(length=31, angle=60),
        "Box": lambda: box_kernel(size=20),
        "RandomSeed": lambda: random_kernel(size=21, seed=123),
        "KeyImage": lambda: key_image_kernel_rgb("../images/private_key.png", size=21)
    }

    noise_sigma = 0.03
    lam = 1e-2

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "kernel", "correct_key", "mse", "psnr"])

        for img_name in IMAGES:
            x = load_image(IMAGE_DIR + img_name)

            for kernel_name, kernel_fn in kernel_functions.items():
                K = kernel_fn()

                # FORWARD OPERATOR
                y = apply_kernel_fft(x, K)  # blurred/encrypted image
                z = y + noise_sigma * np.random.randn(*y.shape) # noisy observed image

                # Save the FORWARD IMAGES (new)
                save_image(y, f"{RESULT_DIR}/{img_name[:-4]}/{kernel_name}_forward_clean.png")
                save_image(z, f"{RESULT_DIR}/{img_name[:-4]}/{kernel_name}_forward_noisy.png")

                # CORRECT-KEY RECOVERY
                x_hat = deconv_fft(z, K, lam)
                m = mse(x, x_hat)
                p = psnr(m)

                save_image(x_hat, f"{RESULT_DIR}/{img_name[:-4]}/{kernel_name}_recovered_correct.png")
                writer.writerow([img_name, kernel_name, "True", m, p])

                # WRONG-KEY ONLY FOR ENCRYPTION KERNELS
                if kernel_name in ["RandomSeed", "KeyImage"]:
                    wrong_K = gaussian_kernel(size=11, sigma=2)
                    x_wrong = deconv_fft(z, wrong_K, lam)
                    m2 = mse(x, x_wrong)
                    p2 = psnr(m2)

                    save_image(x_wrong, f"{RESULT_DIR}/{img_name[:-4]}/{kernel_name}_recovered_wrong.png")
                    writer.writerow([img_name, kernel_name, "False", m2, p2])

if __name__ == "__main__":
    run_experiment()