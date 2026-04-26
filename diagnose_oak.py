import depthai as dai
import time

def test_device():
    print("开始诊断测试...")
    
    # 1. 检查可用设备
    device_infos = dai.Device.getAllAvailableDevices()
    if not device_infos:
        print("错误：未找到任何 OAK 设备！请检查 USB 连接。")
        return

    print(f"找到 {len(device_infos)} 个设备:")
    for info in device_infos:
        print(f"- [{info.getMxId()}] 状态: {info.state}")

    # 2. 尝试最简单的连接
    print("\n尝试初始化设备 (限制为 USB 2.0 模式以增强稳定性)...")
    try:
        # 显式指定 USB 2.0 模式，有时可以绕过不稳定的 USB 3.0 握手问题
        conf = dai.Device.Config()
        # 尝试使用最基础的配置
        with dai.Device(dai.OpenVINO.Version.VERSION_2021_4, info, usb2Mode=True) as device:
            print("成功：设备已连接！")
            print(f"USB 速度: {device.getUsbSpeed()}")
            print(f"MXID: {device.getMxId()}")
            print(f"芯片温度: {device.getChipTemperature().average} °C")
            
    except Exception as e:
        print(f"\n连接失败！错误详情: {e}")
        print("\n故障排查建议:")
        print("1. 拔掉 USB 线，等待 5 秒后再重新插入。")
        print("2. 尝试更换 USB 3.0 端口（蓝色插口）。")
        print("3. 如果是 OAK-D，请连接 5V 电源适配器。")
        print("4. 检查是否有其他程序（如之前的 demo 进程）正在占用设备。")

if __name__ == "__main__":
    test_device()
