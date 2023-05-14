from PIL import Image
import matplotlib.pyplot as plt
import pandas as np

Names = []
ValueRA= []
ValueBA= []
ValueGA = []
ValueRO= []
ValueBO= []
ValueGO = []
ValueRP= []
ValueBP= []
ValueGP = []
for i in range(1,37):
    if i < 10:
        i = "0" + str(i)
    else:
        i = str(i)
    image = "FRUITS/PEACHES/P" + i +".JPG"
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    print(rgb_im.size)
    r, g, b = rgb_im.getpixel((int(rgb_im.size[0]/2),int(rgb_im.size[1]/2)))
    Names.append("PEACHES")
    ValueRP.append(r)
    ValueBP.append(b)
    ValueGP.append(g)


for i in range(1,16):
    if i < 10:
        i = "0" + str(i)
    else:
        i = str(i)
    image = "FRUITS/ORANGES/O" + i +".JPG"
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    r, g, b = rgb_im.getpixel((int(rgb_im.size[0]/2),int(rgb_im.size[1]/2)))
    Names.append("ORANGES")
    ValueRO.append(r)
    ValueBO.append(b)
    ValueGO.append(g)

for i in range(1,23):
    if i < 10:
        i = "0" + str(i)
    else:
        i = str(i)
    image = "FRUITS/APPLES/A" + i +".JPG"
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    r, g, b = rgb_im.getpixel((int(rgb_im.size[0]/2),int(rgb_im.size[1]/2)))
    Names.append("APPLES")
    ValueRA.append(r)
    ValueBA.append(b)
    ValueGA.append(g)

ValueG = ValueGP + ValueGO + ValueGA
ValueB = ValueBP + ValueBO + ValueBA
ValueR = ValueRP + ValueRO + ValueRA


df = np.DataFrame((Names,ValueR,ValueG,ValueB)).transpose()
df.columns=["name","Red","Green","Blue"]

colors = {'APPLES':'green', 'ORANGES':'orange', 'PEACHES':'pink'}

grouped = df.groupby("name")
fig, ax = plt.subplots()
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='Red', y='Green', label=key, color=colors[key])
grouped = df.groupby("name")

fig, ax = plt.subplots()
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='Red', y='Blue', label=key, color=colors[key])
    grouped = df.groupby("name")

fig, ax = plt.subplots()
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='Blue', y='Green', label=key, color=colors[key])
plt.show()