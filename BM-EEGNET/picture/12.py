import numpy as np
import matplotlib.pyplot as plt

H = np.array([
[500, 281, 287, 247, 259, 281, 318, 225, 340, 256, 300, 294, 311, 362, 289, 328, 274, 211, 245, 237, 306, 295, 266, 328, 258, 247, 347, 287, 288, 390] ,
[500, 500, 215, 341, 208, 387, 231, 229, 356, 337, 231, 285, 226, 213, 290, 394, 363, 233, 347, 278, 324, 359, 353, 357, 321, 389, 399, 200, 305, 288] ,
[500, 500, 500, 338, 303, 355, 251, 310, 205, 356, 266, 228, 361, 202, 383, 346, 363, 376, 202, 318, 247, 294, 377, 263, 396, 220, 223, 353, 337, 333] ,
[500, 500, 500, 500, 241, 259, 233, 243, 291, 202, 278, 336, 346, 307, 340, 303, 352, 243, 205, 398, 225, 351, 315, 357, 287, 210, 210, 285, 390, 232] ,
[500, 500, 500, 500, 500, 298, 353, 391, 382, 384, 297, 267, 337, 271, 294, 326, 202, 381, 279, 266, 270, 293, 286, 219, 381, 252, 375, 285, 310, 387] ,
[500, 500, 500, 500, 500, 500, 349, 328, 218, 384, 303, 224, 347, 212, 332, 216, 239, 340, 386, 251, 276, 240, 351, 244, 364, 305, 383, 201, 290, 202] ,
[500, 500, 500, 500, 500, 500, 500, 258, 367, 231, 378, 354, 222, 223, 342, 208, 343, 368, 366, 310, 335, 240, 304, 362, 257, 215, 371, 239, 230, 313] ,
[500, 500, 500, 500, 500, 500, 500, 500, 300, 359, 320, 383, 270, 384, 247, 210, 365, 362, 229, 320, 248, 356, 295, 266, 200, 256, 229, 292, 231, 377] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 286, 320, 399, 362, 247, 292, 288, 211, 303, 288, 318, 356, 219, 207, 357, 252, 275, 381, 253, 395, 217] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 393, 270, 396, 386, 232, 320, 260, 322, 229, 261, 260, 220, 279, 354, 264, 260, 351, 381, 357, 316] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 200, 239, 237, 333, 361, 204, 285, 309, 215, 319, 214, 240, 262, 340, 200, 284, 373, 235, 243] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 221, 390, 374, 269, 270, 336, 338, 272, 344, 395, 374, 237, 324, 293, 206, 248, 352, 220] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 222, 371, 317, 351, 265, 282, 279, 392, 329, 231, 321, 255, 311, 343, 253, 366, 259] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 267, 208, 208, 328, 253, 374, 284, 375, 274, 306, 337, 286, 267, 314, 258, 259] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 346, 323, 324, 387, 257, 374, 239, 331, 346, 383, 328, 316, 326, 277, 212] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 236, 251, 305, 362, 389, 298, 287, 230, 209, 328, 359, 261, 202, 218] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 345, 271, 253, 240, 320, 281, 205, 276, 309, 266, 356, 295, 380] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 234, 277, 267, 278, 375, 393, 380, 208, 231, 265, 358, 284] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 242, 223, 362, 273, 372, 253, 358, 327, 296, 248, 313] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 331, 374, 306, 340, 331, 332, 277, 202, 254, 283] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 330, 286, 387, 262, 204, 221, 393, 398, 356] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 365, 322, 245, 294, 206, 399, 224, 220] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 244, 359, 207, 324, 357, 253, 374] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 275, 340, 289, 214, 345, 339] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 381, 331, 251, 385, 229] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 203, 295, 292, 372] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 275, 377, 262] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 373, 217] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 359] ,
[500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500]

              ])  # added some commas and array creation code


fig = plt.figure(figsize=(4, 3.2))

ax = fig.add_subplot(111)


# 绘制热力图
from matplotlib import cm
plt.imshow(H,cmap=cm.YlGnBu)
ax.set_aspect('equal')
plt.colorbar(orientation='vertical')
plt.show()