import cv2
def draw_feature_bars(frame, mouth_w_n, lip_gap_n, curve_n, brow_n, origin=(10, 120), max_width=200):
    """
    Draw normalized horizontal bars for each feature (visual strength).
    mouth_w_n etc. should be normalized floats (0..1 or >0).
    """
    x, y = origin
    gap = 18
    # clamp to [0,1] for visualization
    def clamped(v):
        try:
            return max(0.0, min(1.0, float(v)))
        except:
            return 0.0

    bars = [
        ("mouth_w", clamped(mouth_w_n)),
        ("lip_gap", clamped(lip_gap_n)),
        ("curve", clamped(curve_n)),
        ("brow", clamped(brow_n)),
    ]

    for i, (name, val) in enumerate(bars):
        bx = x
        by = y + i*gap
        width = int(val * max_width)
        # background
        cv2.rectangle(frame, (bx, by), (bx + max_width, by + 12), (50,50,50), -1)
        # bar
        cv2.rectangle(frame, (bx, by), (bx + width, by + 12), (0,200,0), -1)
        # label
        cv2.putText(frame, f"{name} {val:.2f}", (bx + max_width + 8, by + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170,100,100), 2)
