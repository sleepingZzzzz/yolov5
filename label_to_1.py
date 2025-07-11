import os

label_dir = r"D:\project\DLCV\DeepLearning_Design\datasets\UA_DETRAC\labels\val"
file_list = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

for idx, filename in enumerate(file_list, 1):
    file_path = os.path.join(label_dir, filename)
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0] == "1":
                parts[0] = "0"  # 把类别编号1改为0
                new_lines.append(" ".join(parts) + "\n")
        # 不再删除无编号1的文件，只是覆盖写入（可能变为空文件）
        with open(file_path, "w") as f:
            f.writelines(new_lines)
        if idx % 1000 == 0 or idx == len(file_list):
            print(f"Processed {idx}/{len(file_list)} files")
    except Exception as e:
        print(f"Error processing {filename}: {e}")