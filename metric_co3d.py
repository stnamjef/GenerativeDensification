import json

# 读取 JSON 文件
def read_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# Here please use the evaluation matrix path for hydrant and teddybear
mvsplat_path = "/workspace/xiangyu/GaussianGen/GDM-object-release/outputs/metrics/release_co3d_hydrant_4_views.json"
pointdec_path = "/workspace/xiangyu/GaussianGen/GDM-object-release/outputs/metrics/release_co3d_teddybear_4_views.json"

hydrant_data = read_json(mvsplat_path)
teddybear_data = read_json(pointdec_path)

num_hydrant = 0
num_teddybear = 0
psnr_all = 0.0
ssim_all = 0.0
lpips_all = 0.0
for i in range(len(hydrant_data['psnr'])):
    num_hydrant = num_hydrant + 1
    psnr_all = psnr_all + hydrant_data['psnr'][i]
    ssim_all = ssim_all + hydrant_data['ssim'][i]
    lpips_all = lpips_all + hydrant_data['lpips_vgg'][i]

for i in range(len(teddybear_data['psnr'])):
    num_teddybear = num_teddybear + 1
    psnr_all = psnr_all + teddybear_data['psnr'][i]
    ssim_all = ssim_all + teddybear_data['ssim'][i]
    lpips_all = lpips_all + teddybear_data['lpips_vgg'][i]

print(num_hydrant)
print(num_teddybear)
print(num_hydrant + num_teddybear)

mean_psnr = psnr_all / (num_hydrant + num_teddybear)
mean_ssim = ssim_all / (num_hydrant + num_teddybear)
mean_lpips = lpips_all / (num_hydrant + num_teddybear)

print('mean value:')
print(mean_psnr, mean_ssim, mean_lpips)