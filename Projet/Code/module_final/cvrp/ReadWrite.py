# This module contains functions, which allow to read an xml file, and write  in a text file

from lxml import etree
import os.path


def read(file):  # give the path of the file
    x = []
    y = []
    demand = [0]
    tree = etree.parse("" + file)
    for abs in tree.xpath("/instance/network/nodes/node/cx"):
        x.append((float(abs.text)))
    for ord in tree.xpath("/instance/network/nodes/node/cy"):
        y.append((float(ord.text)))
    inst = [(x[i], y[i]) for i in range(len(x))]
    for dem in tree.xpath("/instance/requests/request/quantity"):
        demand.append((float(dem.text)))
    for c in tree.xpath("/instance/fleet/vehicle_profile/capacity"):
        C = float(c.text)
    return inst, demand, C


def writef(namefile, text):
    if not os.path.isfile(namefile):
        f = open(namefile, 'w')
        f.write(text + '\n')
        f.close()
    else:
        f = open(namefile, 'a')
        f.write(text + '\n')
        f.close()
