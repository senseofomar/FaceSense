import cv2


def draw_results(frame, bbox, label, confidence):
    x1,y1,x2,y2 = bbox

    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.rectangle(frame,(x1,y1-30), (x1-8,y1), -1)

    cv2.putText(frame, f"({label},{confidence}",
                x1-5, y1+7,
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                color =(0,255,0), thickness=2

    return frame


"""
Why is top-left = (x1, y1 - 30)?

Because the top-left corner must be:

horizontally aligned with face → x1

30 pixels ABOVE face top → y1 - 30

So:

TOP-LEFT OF TEXT BOX = (x1, y1 - 30)

⭐ Why is bottom-right = (x1 + 200, y1)?

Because:

1️⃣ The bottom of the label box should sit right on top of the face box

→ so same y-coordinate as face top
→ y = y1

2️⃣ The right side should extend 200 pixels to the right

→ x = x1 + 200

So:

BOTTOM-RIGHT OF TEXT BOX = (x1 + 200, y1)"""

"""
In the LABEL BOX:

Both coordinates share an x1 or y1 because that’s how rectangles work.

Let’s compare:

Face box:
top-left = (x1, y1)
bottom-right = (x2, y2)

Label box:
top-left = (x1, y1 - 30)
bottom-right = (x1 + 200, y1)


Do you see the difference?

The LABEL BOX coordinates are designed by YOU to sit above the face.

The FACE BOX coordinates come from Mediapipe.

Top-left x = x1 (align with face)

Top-left y = y1 - 30 (above the face)

Bottom-right x = x1 + some_width (200)

Bottom-right y = y1 (touch the face)"""

