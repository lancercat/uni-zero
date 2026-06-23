import random

class colors:
    colors20={'dimgray': '#696969', 'seagreen': '#2e8b57', 'maroon': '#800000', 'olive': '#808000', 'navy': '#000080', 'red': '#ff0000', 'darkorange': '#ff8c00', 'gold': '#ffd700', 'chartreuse': '#7fff00', 'mediumorchid': '#ba55d3', 'mediumspringgreen': '#00fa9a', 'darksalmon': '#e9967a', 'aqua': '#00ffff', 'blue': '#0000ff', 'fuchsia': '#ff00ff', 'dodgerblue': '#1e90ff', 'palegoldenrod': '#eee8aa', 'plum': '#dda0dd', 'deeppink': '#ff1493', 'lightskyblue': '#87cefa'}
    colors64={'mediumseagreen': '#3cb371', 'rosybrown': '#bc8f8f', 'rebeccapurple': '#663399', 'darkgoldenrod': '#b8860b', 'darkkhaki': '#bdb76b', 'darkcyan': '#008b8b', 'peru': '#cd853f', 'steelblue': '#4682b4', 'navy': '#000080', 'chocolate': '#d2691e', 'yellowgreen': '#9acd32', 'limegreen': '#32cd32', 'purple2': '#7f007f', 'darkseagreen': '#8fbc8f', 'maroon3': '#b03060', 'tan': '#d2b48c', 'mediumaquamarine': '#66cdaa', 'orangered': '#ff4500', 'darkturquoise': '#00ced1', 'darkorange': '#ff8c00', 'gold': '#ffd700', 'yellow': '#ffff00', 'mediumvioletred': '#c71585', 'mediumblue': '#0000cd', 'lime': '#00ff00', 'darkviolet': '#9400d3', 'mediumorchid': '#ba55d3', 'mediumspringgreen': '#00fa9a', 'royalblue': '#4169e1', 'crimson': '#dc143c', 'aqua': '#00ffff', 'deepskyblue': '#00bfff', 'mediumpurple': '#9370db', 'blue': '#0000ff', 'lightcoral': '#f08080', 'greenyellow': '#adff2f', 'tomato': '#ff6347', 'thistle': '#d8bfd8', 'fuchsia': '#ff00ff', 'palevioletred': '#db7093', 'khaki': '#f0e68c', 'cornflower': '#6495ed', 'plum': '#dda0dd', 'skyblue': '#87ceeb', 'deeppink': '#ff1493', 'lightsalmon': '#ffa07a', 'paleturquoise': '#afeeee', 'violet': '#ee82ee', 'palegreen': '#98fb98', 'aquamarine': '#7fffd4', 'hotpink': '#ff69b4', 'bisque': '#ffe4c4', 'lightpink': '#ffb6c1'}


def wash(raw):
    cdict={};
    lines=raw.replace(" ","").split("\n");
    kvs=[];
    for l in lines:
        if(len(l)<2):
            continue;
        kvs.append(l);
    for i in range(len(kvs)//2):
        cdict[kvs[i*2]]=kvs[i*2+1];
    return cdict
def random_color():
    return random.randint(100, 255), random.randint(100, 255), random.randint(100, 255);

def get_palette():

    colors={
    "circle9" : ["#CFFFFF", "#6F8BF9", "#F56E41", "#B8DEFF", "#FDF6FE"],
    "madoka" : ["#ffbae4", "#ff8dc7", "#d47295", "#bf3f6c"],
    "nep" : ["#5b5ddf", "#3f54ba", "#274687", "#d5d5ec"],
    "orange" : ["#ffcc90", "#ffb65f", "#ff991e", "#dc7800"],
    "red" : ["#00ff00", "#8ab70e", "#a3c440", "#71f515"],
    "vert": []
    }
    return colors;

if __name__ == '__main__':
    raw = """
    mediumseagreen

#3cb371

rosybrown

#bc8f8f

rebeccapurple

#663399

darkgoldenrod

#b8860b

darkkhaki

#bdb76b

darkcyan

#008b8b

peru

#cd853f

steelblue

#4682b4

navy

#000080

chocolate

#d2691e

yellowgreen

#9acd32

limegreen

#32cd32

purple2

#7f007f

darkseagreen

#8fbc8f

maroon3

#b03060

tan

#d2b48c

mediumaquamarine

#66cdaa

orangered

#ff4500

darkturquoise

#00ced1

darkorange

#ff8c00

gold

#ffd700

yellow

#ffff00

mediumvioletred

#c71585

mediumblue

#0000cd

lime

#00ff00

darkviolet

#9400d3

mediumorchid

#ba55d3

mediumspringgreen

#00fa9a

royalblue

#4169e1

crimson

#dc143c

aqua

#00ffff

deepskyblue

#00bfff

mediumpurple

#9370db

blue

#0000ff

lightcoral

#f08080

greenyellow

#adff2f

tomato

#ff6347

thistle

#d8bfd8

fuchsia

#ff00ff

palevioletred

#db7093

khaki

#f0e68c

cornflower

#6495ed

plum

#dda0dd

skyblue

#87ceeb

deeppink

#ff1493

lightsalmon

#ffa07a

paleturquoise

#afeeee

violet

#ee82ee

palegreen

#98fb98

aquamarine

#7fffd4

hotpink

#ff69b4

bisque

#ffe4c4

lightpink

#ffb6c1
    """
    print(wash(raw))
