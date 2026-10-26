import numpy as np
import matplotlib.pyplot as plt
from src.utils.fourier_correlation_data import find_intercept

if __name__ == '__main__':
    for parent_folder in ['D:/BaiduNetdiskDownload/3DFSC-output/', 'D:/BaiduNetdiskDownload/3DFSC-output/Hn/']:
        for i in range(1, 11):
            folder = f'{parent_folder}Results_{i}-0.5-3DFSC/'
            xf = np.load(f'{folder}Plots{i}-0.5-3DFSC_x.npy')
            yf = np.load(f'{folder}Plots{i}-0.5-3DFSC_y.npy')
            zf = np.load(f'{folder}Plots{i}-0.5-3DFSC_z.npy')
            frequency = np.load(f'{folder}Plots{i}-0.5-3DFSC_frequency.npy')
            fixed_threshold = 0.143 * np.ones_like(xf)
            fig, ax = plt.subplots()
            ax.plot(frequency, xf, color='#E67373', label='X')
            ax.plot(frequency, yf, color='#B84A4A', linestyle='--', label='Y')
            ax.plot(frequency, zf, color='#8C3939', linestyle=':', label='Z')
            ax.plot(frequency, fixed_threshold, 'k', label='1/7 Threshold')
            ax.legend(loc='upper right')
            x_resolution, _ = find_intercept(xf, fixed_threshold, 20)
            y_resolution, _ = find_intercept(yf, fixed_threshold, 20)
            z_resolution, _ = find_intercept(zf, fixed_threshold, 20)
            ax.text(0.8, 0.5, f'X: {x_resolution:.1f}nm', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.8, 0.4, f'Y: {y_resolution:.1f}nm', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.8, 0.3, f'Z: {z_resolution:.1f}nm', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plt.savefig(f'{folder}plot_3dirs.svg')
            plt.savefig(f'{folder}plot_3dirs.png')
            plt.close()

            folder = f'{parent_folder}Results_{i}-0.143-3DFSC/'
            xf = np.load(f'{folder}Plots{i}-0.143-3DFSC_x.npy')
            yf = np.load(f'{folder}Plots{i}-0.143-3DFSC_y.npy')
            zf = np.load(f'{folder}Plots{i}-0.143-3DFSC_z.npy')
            frequency = np.load(f'{folder}Plots{i}-0.143-3DFSC_frequency.npy')
            fixed_threshold = 0.143 * np.ones_like(xf)
            fig, ax = plt.subplots()
            ax.plot(frequency, xf, color='#E67373', label='X')
            ax.plot(frequency, yf, color='#B84A4A', linestyle='--', label='Y')
            ax.plot(frequency, zf, color='#8C3939', linestyle=':', label='Z')
            ax.plot(frequency, fixed_threshold, 'k', label='1/7 Threshold')
            ax.legend(loc='upper right')
            x_resolution, _ = find_intercept(xf, fixed_threshold, 20)
            y_resolution, _ = find_intercept(yf, fixed_threshold, 20)
            z_resolution, _ = find_intercept(zf, fixed_threshold, 20)
            ax.text(0.8, 0.5, f'X: {x_resolution:.1f}nm', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.8, 0.4, f'Y: {y_resolution:.1f}nm', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.text(0.8, 0.3, f'Z: {z_resolution:.1f}nm', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plt.savefig(f'{folder}plot_3dirs.svg')
            plt.savefig(f'{folder}plot_3dirs.png')
            plt.close()
        
        


