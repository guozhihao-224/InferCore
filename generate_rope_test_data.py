import numpy as np
import mlx.core as mx
from pathlib import Path

OUT_DIR = Path("tests/rope_qwen25_data")
OUT_DIR.mkdir(exist_ok=True, parents=True)

BATCH_SIZE = 1
NUM_HEADS = 8
HEAD_DIM = 4
MAX_SEQ_LEN = 20
SEQ_LEN = 10
BASE = 1000000.0   # Qwen2.5 rope_theta

def rope_helper_qwen(precision, with_offset: bool, idx: int):
    # 随机输入
    x = mx.random.uniform(
        shape=(BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM), dtype=precision
    )

    # 偏移
    if with_offset:
        input_pos = np.random.randint(0, MAX_SEQ_LEN - SEQ_LEN)
        input_pos_mx = input_pos
        input_pos_user = slice(input_pos, input_pos + SEQ_LEN)
    else:
        input_pos = None
        input_pos_mx = None
        input_pos_user = None

    # ✅ 保持你原有的 reference_output 计算方式
    reference_output = mx.fast.rope(
        x.transpose(0, 2, 1, 3),
        dims=HEAD_DIM,
        traditional=False,
        base=BASE,
        scale=1.0,
        offset=input_pos_mx or 0,
    ).transpose(0, 2, 1, 3)

    # 保存数据
    tag = f"qwen25_fp32_{'off' if with_offset else 'nooff'}_{idx}"
    np.save(OUT_DIR / f"{tag}_input.npy", x)
    np.save(OUT_DIR / f"{tag}_reference.npy", reference_output)
    np.save(OUT_DIR / f"{tag}_offset.npy", np.array([input_pos_mx or 0], dtype=np.int32))

    print(f"✅ Saved {tag}_*.npy")

def gen_qwen25_tests():
    for with_offset in [True, False]:
        for idx in range(10):  # 减少到10个测试用例以便调试
            rope_helper_qwen(mx.float32, with_offset, idx)

if __name__ == "__main__":
    gen_qwen25_tests() 