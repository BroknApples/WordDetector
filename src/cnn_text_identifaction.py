"""

1️⃣ Sort all detected boxes by their Y coordinate
Words on the same line will share similar y.
Pseudo-code:
boxes.sort(key=lambda b: b.y1)

2️⃣ Group boxes that are "close" vertically
If |y_i – y_j| < threshold → same text line.
People often use:
vertical_threshold = avg_height * 0.5

3️⃣ Sort each line group left→right (x1)
This puts words in reading order.

4️⃣ Combine them

Example:
Detected words (in order):
["the", "quick", "brown", "fox"]
Merged into:
"the quick brown fox"

✨ Expected Result

EAST detects:
"the" → box 1
"quick" → box 2
"brown" → box 3
"fox" → box 4
Your grouping step outputs:
Sentence #1: "the quick brown fox"
Bounding box: min(all x1,y1), max(all x2,y2)


"""

