import depthai as dai
from datetime import datetime
import os

print("从设备读取校准数据并保存为JSON文件...")

try:
    # 连接设备
    with dai.Device() as device:
        print("设备连接成功")
        
        # 获取设备信息
        device_info = device.getDeviceInfo()
        mx_id = device_info.getMxId()
        print(f"设备ID: {mx_id}")
        
        # 读取校准数据
        calib_data = device.readCalibration()
        
        # 生成文件名
        timestamp = datetime.now().strftime("_%m_%d_%y_%H_%M")
        filename = f"{mx_id}{timestamp}.json"
        
        # 创建resources目录
        resources_dir = "resources"
        os.makedirs(resources_dir, exist_ok=True)
        
        # 保存路径
        save_path = os.path.join(resources_dir, filename)
        
        # 保存为JSON文件
        calib_data.eepromToJsonFile(save_path)
        
        print(f"校准数据已保存到: {save_path}")
        
        # 验证文件
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path)
            print(f"文件大小: {file_size} 字节")
        else:
            print("文件创建失败")
            
except Exception as e:
    print(f"错误: {e}")